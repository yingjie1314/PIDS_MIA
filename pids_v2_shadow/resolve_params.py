#!/usr/bin/env python
"""Resolve the EFFECTIVE training config for every arm of the v2 campaigns.

Grepping config/<detector>.yml is wrong: those files inherit, so a value absent from the file is
not a value absent from the run. This drives the real loader with the exact flag list recorded in
each arm's run_flags_<DATASET>.json, so what it prints is what the arm actually trained under.

Writes JSON to stdout. Artifacts are pointed at a scratch dir so nothing here touches campaign data.
"""

import contextlib
import io
import json
import os
import pathlib
import sys

sys.path.insert(0, "/home/yingjie/PIDSMaker")

from pidsmaker.config.pipeline import get_runtime_required_args, get_yml_cfg  # noqa: E402

SCRATCH = "/data/pids_v2_shadow/_param_scratch"

FIELDS = [
    "lr",
    "node_hid_dim",
    "node_out_dim",
    "emb_dim",
    "dropout",
    "num_epochs",
    "time_window_size",
    "weight_decay",
    "batch_size",
    "patience",
]


def find(node, key, path=""):
    """Every occurrence of `key` anywhere in the config tree, with its dotted path."""
    out = []
    if hasattr(node, "items"):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if k == key and not hasattr(v, "items"):
                out.append((p, v))
            out.extend(find(v, key, p))
    return out


def resolve(detector, dataset, flags):
    argv = ["main.py", detector, dataset] + list(flags)
    old = sys.argv
    sys.argv = argv
    try:
        # the loader is chatty; keep stdout clean so the caller gets pure JSON
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            args, unknown = get_runtime_required_args(return_unknown_args=True)
            cfg = get_yml_cfg(args)
    finally:
        sys.argv = old
    if unknown:
        return {"error": f"unknown args {unknown}"}
    res = {}
    for f in FIELDS:
        hits = find(cfg, f)
        if hits:
            res[f] = {p: v for p, v in hits}
    for f in ["used_methods", "used_method"]:
        res[f] = {p: v for p, v in find(cfg, f)}
    return res


def main():
    pathlib.Path(SCRATCH).mkdir(parents=True, exist_ok=True)
    campaigns = {
        "cadets_e3": "/data/pids_v2_shadow/cadets_e3",
        "theia_e3": "/data/pids_v2_shadow/theia_e3",
        "clearscope_e3": "/data/pids_v2_shadow/clearscope_e3",
    }
    out = {}
    for camp, root in campaigns.items():
        out[camp] = {}
        for p in sorted(pathlib.Path(root).glob("*/run_flags_*.json")):
            rec = json.load(open(p))
            det, ds = rec["detector"], rec["dataset"]
            flags = [f for f in rec["flags"]]
            # redirect artifacts away from the live campaign tree
            if "--artifact_dir" in flags:
                flags[flags.index("--artifact_dir") + 1] = SCRATCH
            try:
                out[camp][f"{det}|{ds}"] = {
                    "recorded_flags": rec["flags"],
                    "epochs": rec["epochs"],
                    "dump_epochs": rec["dump_epochs"],
                    "effective": resolve(det, ds, flags),
                }
            except Exception as e:  # noqa: BLE001
                out[camp][f"{det}|{ds}"] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=2, default=str)


if __name__ == "__main__":
    main()
