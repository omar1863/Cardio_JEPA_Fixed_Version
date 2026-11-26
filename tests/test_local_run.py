"""Tests for the local-run additions: resumable checkpoints, explicit dataset paths,
partial PTB-XL copies, SHDB-AF rhythm annotations and the environment check."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import cardio_jepa as cj
from cardio_jepa.config import smoke_config
from cardio_jepa.data import discover_explicit
from cardio_jepa.data.build import build_pretrain_loaders
from cardio_jepa.data.longform import _episodes_from_annotation
from cardio_jepa.data.wfdb_lite import Annotation
from cardio_jepa.models.cardio_jepa import CardioJEPA


def _cfg(epochs=3):
    cfg = smoke_config()
    cfg.data.allow_synthetic_fallback = True
    cfg.data.input_root = "/nonexistent-input"
    cfg.optim.epochs = epochs
    cfg.optim.on_collapse = "ignore"
    cfg.optim.max_steps_per_epoch = 6
    return cfg


def _model(cfg):
    return CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)


class _Boom(RuntimeError):
    pass


class _CrashingLoader:
    """Wraps a loader and raises after ``n`` batches in total -- a simulated power cut."""

    def __init__(self, inner, n):
        self.inner, self.left = inner, n

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        for b in self.inner:
            if self.left <= 0:
                raise _Boom()
            self.left -= 1
            yield b


# ------------------------------------------------------------------ resumable training
def test_interrupted_run_resumes_to_completion(tmp_path):
    cfg = _cfg(epochs=3)
    save = str(tmp_path / "m.pth")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)

    # die in the middle of epoch 1 (6 steps/epoch -> after 9 steps)
    with pytest.raises(_Boom):
        cj.pretrain(_model(cfg), _CrashingLoader(loader, 9), cfg, arm_name="t",
                    save_path=save, verbose=False, log_every=0, ckpt_every_steps=2)
    assert os.path.isfile(str(tmp_path / "m_last.pt"))
    assert not os.path.isfile(save)                       # not complete yet

    res = cj.pretrain(_model(cfg), loader, cfg, arm_name="t", save_path=save,
                      verbose=False, log_every=0, ckpt_every_steps=2)
    ck = torch.load(save, map_location="cpu", weights_only=False)
    assert ck["complete"]
    assert len(res.history) == cfg.optim.epochs           # all three epochs, none lost
    assert len(res.monitor.history) == cfg.optim.epochs
    assert not os.path.isfile(str(tmp_path / "m_last.pt"))  # superseded by the final model


def test_resume_does_not_repeat_finished_steps(tmp_path):
    """A resumed run must execute exactly the remaining steps, not start over."""
    cfg = _cfg(epochs=3)
    save = str(tmp_path / "m.pth")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    total = 3 * 6

    with pytest.raises(_Boom):
        cj.pretrain(_model(cfg), _CrashingLoader(loader, 8), cfg, arm_name="t",
                    save_path=save, verbose=False, log_every=0, ckpt_every_steps=1)

    counter = _CrashingLoader(loader, 10 ** 9)
    cj.pretrain(_model(cfg), counter, cfg, arm_name="t", save_path=save,
                verbose=False, log_every=0, ckpt_every_steps=1)
    ran = 10 ** 9 - counter.left
    assert ran == total - 8, f"resumed run did {ran} steps, expected {total - 8}"


def test_finished_run_is_reloaded_not_retrained(tmp_path):
    cfg = _cfg(epochs=2)
    save = str(tmp_path / "m.pth")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    m1 = _model(cfg)
    cj.pretrain(m1, loader, cfg, arm_name="t", save_path=save, verbose=False, log_every=0)

    counter = _CrashingLoader(loader, 10 ** 9)
    m2 = _model(cfg)
    res = cj.pretrain(m2, counter, cfg, arm_name="t", save_path=save, verbose=False,
                      log_every=0)
    assert counter.left == 10 ** 9                        # no batch was drawn
    assert res.seconds == 0.0 and len(res.monitor.history) == 2
    for (k, a), (_k, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a.cpu(), b.cpu()), k


def test_corrupt_last_checkpoint_falls_back_to_backup(tmp_path):
    """A hard power cut can leave _last.pt truncated; resume must use _last.pt.bak
    instead of silently starting over."""
    cfg = _cfg(epochs=3)
    save = str(tmp_path / "m.pth")
    last = str(tmp_path / "m_last.pt")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    with pytest.raises(_Boom):
        cj.pretrain(_model(cfg), _CrashingLoader(loader, 8), cfg, arm_name="t",
                    save_path=save, verbose=False, log_every=0, ckpt_every_steps=1)
    assert os.path.isfile(last + ".bak")
    with open(last, "r+b") as fh:                          # simulate the lost data
        fh.truncate(1024)

    counter = _CrashingLoader(loader, 10 ** 9)
    cj.pretrain(_model(cfg), counter, cfg, arm_name="t", save_path=save,
                verbose=False, log_every=0, ckpt_every_steps=1)
    ran = 10 ** 9 - counter.left
    assert ran == 3 * 6 - 7, f"resumed run did {ran} steps, expected {3 * 6 - 7}"
    assert not os.path.isfile(last) and not os.path.isfile(last + ".bak")


def test_checkpoint_from_a_different_config_is_not_resumed(tmp_path):
    cfg = _cfg(epochs=3)
    save = str(tmp_path / "m.pth")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    with pytest.raises(_Boom):
        cj.pretrain(_model(cfg), _CrashingLoader(loader, 5), cfg, arm_name="t",
                    save_path=save, verbose=False, log_every=0, ckpt_every_steps=1)

    cfg2 = _cfg(epochs=3)
    cfg2.model.encoder_embed_dim = 128                     # incompatible width
    cfg2.model.encoder_num_heads = 4
    counter = _CrashingLoader(loader, 10 ** 9)
    cj.pretrain(_model(cfg2), counter, cfg2, arm_name="t", save_path=save, verbose=False,
                log_every=0)
    assert 10 ** 9 - counter.left == 3 * 6                 # trained from scratch, no crash


def test_epoch_snapshots_are_kept(tmp_path):
    cfg = _cfg(epochs=4)
    save = str(tmp_path / "m.pth")
    loader, _ = build_pretrain_loaders({}, cfg, verbose=False)
    cj.pretrain(_model(cfg), loader, cfg, arm_name="t", save_path=save, verbose=False,
                log_every=0, keep_every_epochs=2)
    assert os.path.isfile(str(tmp_path / "m_ep002.pth"))
    assert os.path.isfile(str(tmp_path / "m_ep004.pth"))
    assert not os.path.isfile(str(tmp_path / "m_ep001.pth"))


# ------------------------------------------------------------------------ data layer
def _write_wfdb_dir(d, names, fs=250, n=2500):
    os.makedirs(d, exist_ok=True)
    for nm in names:
        with open(os.path.join(d, nm + ".hea"), "w") as fh:
            fh.write(f"{nm} 2 {fs} {n}\n{nm}.dat 16 200 16 0 0 0 0 I\n"
                     f"{nm}.dat 16 200 16 0 0 0 0 II\n")
        np.zeros(n * 2, dtype="<i2").tofile(os.path.join(d, nm + ".dat"))


def test_discover_explicit_maps_arbitrary_folder_names(tmp_path):
    _write_wfdb_dir(str(tmp_path / "data"), ["00", "01"], fs=128)
    _write_wfdb_dir(str(tmp_path / "Training_WFDB"), ["A0001"], fs=500)
    found = discover_explicit({"ltaf": str(tmp_path / "data"),
                               "cpsc": str(tmp_path / "Training_WFDB"),
                               "afdb": str(tmp_path / "missing")}, verbose=False)
    assert set(found) == {"ltaf", "cpsc"}                  # missing dir is skipped, not fatal
    assert found["ltaf"].kind == "long" and found["ltaf"].fmt == "wfdb"
    assert found["cpsc"].kind == "short"
    with pytest.raises(KeyError):
        discover_explicit({"not-a-corpus": str(tmp_path)}, verbose=False)


def test_new_long_corpus_keys_are_distinct_from_mitbih():
    from cardio_jepa.data.discovery import _match_key
    assert _match_key("mit-bih-atrial-fibrillation-database")[0] == "afdb"
    assert _match_key("mit-bih-arrhythmia-database")[0] == "mitbih"
    assert _match_key("shdb-af-a-japanese-holter-ecg-database")[0] == "shdb"


def test_shdb_style_rhythm_annotations_are_recognised():
    """SHDB-AF stores rhythm changes as '"' comments with a parenthesised note, and its
    first annotations sit at negative sample indices."""
    ann = Annotation(
        sample=np.array([0, -1, 100, 500, 900, 1300]),
        symbol=[" ", '"', '"', '"', '"', '"'],
        aux_note=["## time resolution: 200", "(N", "", "(AFIB", "", "(N"], fs=200.0)
    eps = _episodes_from_annotation(ann, 2000)
    assert eps == [(500, 1300, "AFIB"), (1300, 2000, "N")]


def test_long_records_without_a_signal_file_are_skipped(tmp_path):
    from cardio_jepa.data.discovery import FoundDataset
    from cardio_jepa.data.longform import load_long_records
    d = str(tmp_path / "afdb")
    _write_wfdb_dir(d, ["04015"])
    with open(os.path.join(d, "00735.hea"), "w") as fh:     # header only, like data_2
        fh.write("00735 2 250 2500\n00735.dat 16 200 16 0 0 0 0 ECG1\n"
                 "00735.dat 16 200 16 0 0 0 0 ECG2\n")
    ds = FoundDataset(key="afdb", label="afdb", kind="long", root=d, data_root=d,
                      fmt="wfdb")
    recs = load_long_records(ds, limit=10)
    assert [r.name for r in recs] == ["04015"]


def test_partial_ptbxl_only_indexes_records_present_on_disk(tmp_path):
    import pandas as pd
    from cardio_jepa.data.discovery import FoundDataset
    from cardio_jepa.data.short import index_ptbxl
    root = str(tmp_path)
    rows = [{"ecg_id": i, "patient_id": i, "strat_fold": 1 + i % 10,
             "scp_codes": "{'NORM': 100.0}",
             "filename_lr": f"records100/00000/{i:05d}_lr",
             "filename_hr": f"records500/00000/{i:05d}_hr"} for i in range(1, 6)]
    pd.DataFrame(rows).to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    pd.DataFrame({"diagnostic": [1], "diagnostic_class": ["NORM"]},
                 index=["NORM"]).to_csv(os.path.join(root, "scp_statements.csv"))
    _write_wfdb_dir(os.path.join(root, "records100", "00000"), ["00001_lr", "00003_lr"],
                    fs=100, n=1000)
    ds = FoundDataset(key="ptbxl", label="PTB-XL", kind="short", root=root,
                      data_root=root, fmt="ptbxl")
    ix = index_ptbxl(ds, limit=10 ** 9)
    assert len(ix) == 2 and ix.meta["n_missing_on_disk"] == 3
    assert len(ix.meta["labels"]) == 2 and len(ix.meta["strat_fold"]) == 2


# --------------------------------------------------------------------- environment
def test_envcheck_reports_missing_datasets_and_passes_on_smoke(tmp_path):
    from cardio_jepa.envcheck import check_environment
    rep = check_environment({"ltaf": str(tmp_path), "afdb": str(tmp_path / "nope")},
                            run_mode="smoke", out_dir=str(tmp_path), verbose=False)
    assert any("afdb" in w for w in rep["warnings"])
    assert rep["ok"], rep["errors"]
    bad = check_environment({"x": str(tmp_path / "nope")}, run_mode="smoke",
                            out_dir=str(tmp_path), verbose=False)
    assert not bad["ok"]
