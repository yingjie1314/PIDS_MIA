import json
import os

from pidsmaker.detection.training_methods.mia_dump import split_train_graphs, write_manifest


def _fake_train_data(n=100):
    # train_data is list[dataset]; dataset is list[graph]. Use ints as graph stand-ins.
    return [list(range(n))]


def test_random_split_is_deterministic_and_partitions():
    data = _fake_train_data(200)
    in1, out1 = split_train_graphs(data, mode="random", frac=0.5, seed=7)
    in2, out2 = split_train_graphs(data, mode="random", frac=0.5, seed=7)
    assert in1 == in2 and out1 == out2  # deterministic

    in_flat = [g for ds in in1 for g in ds]
    out_flat = [g for ds in out1 for g in ds]
    assert set(in_flat).isdisjoint(out_flat)             # partition
    assert sorted(in_flat + out_flat) == list(range(200))  # covers all
    assert 60 < len(in_flat) < 140                        # ~50% with tolerance


def test_random_split_seed_changes_partition():
    data = _fake_train_data(200)
    in1, _ = split_train_graphs(data, mode="random", frac=0.5, seed=1)
    in2, _ = split_train_graphs(data, mode="random", frac=0.5, seed=2)
    assert [g for ds in in1 for g in ds] != [g for ds in in2 for g in ds]


def test_calendar_mode_keeps_all_in():
    data = _fake_train_data(50)
    in_ds, out_ds = split_train_graphs(data, mode="calendar", frac=0.5, seed=0)
    assert [g for ds in in_ds for g in ds] == list(range(50))
    assert [g for ds in out_ds for g in ds] == []


def test_write_manifest(tmp_path):
    path = write_manifest(
        str(tmp_path), epoch=11,
        meta={"detector": "nodlink", "mode": "random", "frac": 0.5, "seed": 3, "role": "shadow"},
    )
    assert os.path.exists(path)
    data = json.load(open(path))
    assert data["epoch"] == 11
    assert data["train_in_dir"].endswith("mia_train_in")
    assert data["heldout_dir"].endswith("mia_heldout")
    assert data["detector"] == "nodlink"
    assert data["role"] == "shadow"
