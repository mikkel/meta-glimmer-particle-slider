"""CLI orchestration with a synthetic host; never a native Glimmer claim."""
import copy
import json
import sys

import torch

from glimmer_slider.__main__ import main
from glimmer_slider.storage import load_adapter
from test_pipeline import IDENTITY, TinyCausal


class FixtureTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": torch.tensor([[int(token) for token in messages[0]["content"].split()]])}

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def test_train_resume_sample_distill_commands(tmp_path, monkeypatch, original_api, capsys):
    import glimmer_slider.native as native
    torch.manual_seed(83)
    base = TinyCausal().eval()
    calls = []
    def load(_path, *, preflight, **kwargs):
        tokenizer = FixtureTokenizer()
        preflight(tokenizer)
        calls.append("synthetic_host")
        return copy.deepcopy(base), tokenizer, IDENTITY
    monkeypatch.setattr(native, "load_local", load)
    def messages(text):
        return [{"role": "user", "content": text}]
    pairs = [dict(neutral=messages("1 3"), positive=messages("2 4"), history=[6, 7]),
             dict(neutral=messages("1 5"), positive=messages("2 8"), history=[9])]
    evaluation = [dict(neutral=messages("1 10"), positive=messages("2 11"), history=[12]),
                  dict(neutral=messages("1 13"), positive=messages("2 14"), history=[15])]
    output = tmp_path / "train"
    cfg = dict(particle_sliders=str(original_api.root), snapshot="synthetic-fixture",
               output_dir=str(output), max_tokens=16, max_new_tokens=3,
               horizon=8, updates=2, checkpoint_every=2, seed=7,
               training=pairs, evaluation=evaluation, generation_evaluation=[messages("1 10")])
    path = tmp_path / "command.json"
    def run(command):
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr(sys, "argv", ["glimmer_slider", command, "--config", str(path)])
        main()
    run("train")
    assert (output / "resume-000002.pt").exists()
    initial = (output / "metrics-000002.json").read_bytes()
    cfg["resume"] = str(output / "resume-000002.pt")
    run("train")
    assert (output / "resume-000004.pt").exists()
    assert (output / "metrics-000002.json").read_bytes() == initial
    metrics = json.loads((output / "metrics-000004.json").read_text())
    assert set(metrics["generation"]["0"]) == {"off", "raw", "ema"}
    cfg.update(adapter=str(output / "teacher-raw-000004.safetensors"), messages=messages("1 10"))
    run("sample")
    cfg.update(output_dir=str(tmp_path / "distill"), calibration=[messages("1 3 6"), messages("1 5 9")])
    run("distill")
    student, record = load_adapter(copy.deepcopy(base), IDENTITY, tmp_path / "distill/ordinary-lora.safetensors")
    assert student.ordinary
    assert len(record["calibration"]["teacher_sha256"]) == 64
    assert calls == ["synthetic_host"] * 4
    capsys.readouterr()
