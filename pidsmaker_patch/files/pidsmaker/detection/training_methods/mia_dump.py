"""MIA support: split training window-graphs into in-training / held-out sets
and dump per-node inference losses for both, for membership inference attacks."""

import json
import os

import numpy as np

from pidsmaker.utils.utils import get_device, log

from . import inference_loop


def split_train_graphs(train_data, mode: str, frac: float, seed: int):
    """Partition training datasets into (in_datasets, out_datasets).

    train_data is list[dataset]; dataset is an indexable sequence of graphs.
    mode="random":   each graph -> in with probability `frac`, else out (seeded).
    mode="calendar": all graphs in-training; out is empty (caller supplies val/test).
    mode="xrole":    all graphs in-training; out is empty (caller supplies val only). `frac` is
                     unused, so the model trains on 100% of its training days.
    """
    if mode in ("calendar", "xrole"):
        return list(train_data), [[] for _ in train_data]
    if mode != "random":
        raise ValueError(f"Unknown mia_split_mode: {mode!r}")

    rng = np.random.default_rng(seed)
    in_datasets, out_datasets = [], []
    for dataset in train_data:
        mask = rng.random(len(dataset)) < frac
        in_datasets.append([g for g, m in zip(dataset, mask) if m])
        out_datasets.append([g for g, m in zip(dataset, mask) if not m])
    return in_datasets, out_datasets


def write_manifest(edge_losses_dir: str, epoch: int, meta: dict) -> str:
    os.makedirs(edge_losses_dir, exist_ok=True)
    manifest = {
        "epoch": epoch,
        "train_in_dir": os.path.join(edge_losses_dir, "mia_train_in"),
        "heldout_dir": os.path.join(edge_losses_dir, "mia_heldout"),
        **meta,
    }
    path = os.path.join(edge_losses_dir, "mia_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    # Per-epoch copy. `mia_manifest.json` is a fixed path, so a run that dumps at several
    # epochs (training.mia_dump_epochs) overwrites it each time and only the last epoch stays
    # discoverable -- even though every epoch's dump directory survives. Existing readers glob
    # the exact name `mia_manifest.json` and are unaffected by this extra file.
    with open(os.path.join(edge_losses_dir, f"mia_manifest_epoch_{epoch}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _dump_split(cfg, model, datasets, split_name, epoch, device):
    model_epoch_file = f"model_epoch_{epoch}"
    # Reset stateful encoder (TGN memory / last-neighbor) so the in-training and
    # held-out splits are each scored from the SAME post-training state. Without
    # this, held-out graphs would be scored with state warmed by all in-training
    # graphs, asymmetrically shrinking the member/non-member gap. No-op for
    # stateless encoders (e.g. NodLink).
    if hasattr(model, "reset_state"):
        model.reset_state()
    # Dispatch to node- or edge-level inference exactly as inference_loop.main does,
    # so edge-level detectors (e.g. Orthrus) dump edge losses instead of crashing.
    test_fn = (
        inference_loop.test_node_level
        if cfg._is_node_level
        else inference_loop.test_edge_level
    )
    for dataset in datasets:
        for g in dataset:
            g.to(device=device)
            test_fn(
                data=g, model=model, split=split_name,
                model_epoch_file=model_epoch_file, cfg=cfg, device=device,
            )
            g.to("cpu")


def dump_mia_inference(cfg, model, in_datasets, out_datasets, epoch, device) -> dict:
    model.eval()
    model.to_device(device)
    log(f"[MIA] dumping in-training inference (epoch {epoch})...")
    _dump_split(cfg, model, in_datasets, "mia_train_in", epoch, device)
    log(f"[MIA] dumping held-out inference (epoch {epoch})...")
    _dump_split(cfg, model, out_datasets, "mia_heldout", epoch, device)

    # Every detector writes its manifest to <hash>/CADETS_E3/edge_losses/mia_manifest.json,
    # so the detector name must be recorded here: without it the only way to tell whose
    # manifest this is, is file mtime, which silently mixes detectors across sweeps.
    meta = {
        "detector": cfg._model,
        "node_level": bool(cfg._is_node_level),
        "mode": cfg.training.mia_split_mode,
        # frac/seed only mean something in "random" mode. Recording the config default under
        # "calendar"/"xrole" would claim the training set was subsampled when it was not -- those
        # modes train on 100% of the training days and take their non-members from other days.
        "frac": cfg.training.mia_split_frac if cfg.training.mia_split_mode == "random" else None,
        "seed": cfg.training.mia_split_seed if cfg.training.mia_split_mode == "random" else None,
        # Lets the attack side tell a rich (embedding) dump from a loss-only one without
        # globbing for .npz files that may legitimately be absent.
        "rich_dump": bool(getattr(cfg.training, "mia_rich_dump", False)),
    }
    # Opt-in control pass, gated on the environment rather than a config key: a new
    # config Arg changes the config hash, which invalidates the cached preprocessing
    # for every run that has already been done.
    if os.environ.get("PIDS_MIA_SINGLEPASS") == "1":
        log(f"[MIA] dumping single-pass control (epoch {epoch})...")
        ordered = _original_order(
            in_datasets, out_datasets, cfg.training.mia_split_mode,
            cfg.training.mia_split_frac, cfg.training.mia_split_seed,
        )
        meta["singlepass_labels"] = _dump_singlepass(cfg, model, ordered, epoch, device)

    manifest = write_manifest(cfg.training._edge_losses_dir, epoch, meta)
    return json.load(open(manifest))


def _original_order(in_datasets, out_datasets, mode, frac, seed):
    """Rebuild the pre-split graph sequence, pairing each graph with its membership label.

    split_train_graphs discards the original order, so the mask is regenerated from the
    same seed and used to interleave the two lists back. Any disagreement means the
    regenerated mask is not the one that produced the split, which would silently
    mislabel every graph, so it raises instead.
    """
    if mode in ("calendar", "xrole"):
        return [(g, 1) for ds in in_datasets for g in ds]
    if mode != "random":
        raise ValueError(f"Unknown mia_split_mode: {mode!r}")

    rng = np.random.default_rng(seed)
    ordered = []
    for ins, outs in zip(in_datasets, out_datasets):
        mask = rng.random(len(ins) + len(outs)) < frac
        if int(mask.sum()) != len(ins):
            raise RuntimeError(
                f"regenerated mask disagrees with the split: {int(mask.sum())} in-graphs "
                f"from the mask vs {len(ins)} in the split"
            )
        it_in, it_out = iter(ins), iter(outs)
        ordered += [(next(it_in), 1) if m else (next(it_out), 0) for m in mask]
    return ordered


def _dump_singlepass(cfg, model, ordered, epoch, device) -> str:
    """Label-independent control dump: ONE reset, every training graph scored in the
    order it was in before the split.

    _dump_split resets the encoder between the member and the non-member pass, so for a
    stateful encoder (TGN memory) a graph is scored against state warmed only by graphs
    sharing its label -- the feature is then a function of the label, and an attack on it
    measures the dump procedure rather than the model. A single pass cannot do that.
    """
    model_epoch_file = f"model_epoch_{epoch}"
    test_fn = (
        inference_loop.test_node_level
        if cfg._is_node_level
        else inference_loop.test_edge_level
    )
    logs_dir = os.path.join(
        cfg.training._edge_losses_dir, "mia_singlepass", model_epoch_file
    )
    os.makedirs(logs_dir, exist_ok=True)
    # Leftovers from an earlier run share window names with this one, which would make
    # every new-file diff below come back empty.
    for stale in os.listdir(logs_dir):
        os.remove(os.path.join(logs_dir, stale))

    if hasattr(model, "reset_state"):
        model.reset_state()

    labels, seen = [], set()
    for i, (g, member) in enumerate(ordered):
        g.to(device=device)
        test_fn(
            data=g, model=model, split="mia_singlepass",
            model_epoch_file=model_epoch_file, cfg=cfg, device=device,
        )
        g.to("cpu")
        new = set(os.listdir(logs_dir)) - seen
        # A window whose name repeats overwrites the earlier CSV and leaves the diff
        # empty; joining on the name downstream would then attach one score to two
        # labels, so refuse rather than emit an ambiguous manifest.
        if len(new) != 1:
            raise RuntimeError(
                f"graph {i} produced {len(new)} new loss files (expected 1) -- "
                "window names are not unique, cannot label this dump"
            )
        name = new.pop()
        seen.add(name)
        labels.append({"index": i, "file": name, "member": int(member)})

    path = os.path.join(cfg.training._edge_losses_dir, "mia_singlepass_labels.json")
    with open(path, "w") as fh:
        json.dump({"epoch": epoch, "dir": logs_dir, "graphs": labels}, fh, indent=2)
    log(f"[MIA] single-pass control: {len(labels)} graphs -> {path}")
    return path
