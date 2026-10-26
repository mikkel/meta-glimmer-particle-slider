import copy
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from glimmer_slider.host import Slider, distill
from glimmer_slider.native import load_local, prompt_ids, sample
from glimmer_slider.storage import OutputStore, STUDENT, load_adapter, save_adapter
from glimmer_slider.training import Trainer, prepare
from glimmer_slider.upstream import REVISION, load_upstream

IDENTITY = dict(model_id="synthetic/tiny-causal-test", revision="0" * 40)
PAIRS = [dict(neutral=[1, 3], positive=[2, 4], history=[6, 7]),
         dict(neutral=[1, 5], positive=[2, 8], history=[9])]
HELD_OUT = [dict(neutral=[1, 10], positive=[2, 11], history=[12]),
            dict(neutral=[1, 13], positive=[2, 14], history=[15, 16])]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(12, 12, bias=False)
        self.k_proj = nn.Linear(12, 4, bias=False)
        self.v_proj = nn.Linear(12, 4, bias=False)
        self.o_proj = nn.Linear(12, 12, bias=False)

    def forward(self, x):
        mixed = self.q_proj(x) + self.k_proj(x).repeat(1, 1, 3) + self.v_proj(x).repeat(1, 1, 3)
        # Causal prefix aggregation, with grouped KV output dimensions.
        causal = mixed.cumsum(1) / torch.arange(1, x.shape[1] + 1).view(1, -1, 1)
        return x + self.o_proj(causal.tanh())


class TinyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = Attention()
        self.model.language_model.layers = nn.ModuleList([layer])
        self.model.vision_model = nn.Module()
        self.model.vision_model.q_proj = nn.Linear(12, 12)
        self.head = nn.Linear(12, 32, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        for layer in self.model.language_model.layers:
            h = layer.self_attn(h)
        return SimpleNamespace(logits=self.head(h), hidden_states=(h,))

    def generate(self, input_ids, max_new_tokens, **kwargs):
        result = input_ids
        for _ in range(max_new_tokens):
            token = self(result).logits[:, -1].argmax(-1, keepdim=True)
            result = torch.cat([result, token], dim=1)
        return result


def setup(seed=41, *, protocol=None):
    torch.manual_seed(seed)
    model = TinyCausal().eval()
    baseline = copy.deepcopy(model)
    slider = Slider(model, IDENTITY)
    train = prepare(model, slider, PAIRS)
    dev = prepare(model, slider, HELD_OUT)
    trainer = Trainer(model, slider, train, dev, horizon=8, seed=3, evaluation_protocol=protocol)
    return model, baseline, slider, trainer


def test_live_bank_strength_and_zero(original_api):
    model, baseline, slider, _ = setup()
    assert slider.particles.shape == (128, 4)
    assert len(slider.branches) == 4
    assert [b.lora_up.out_features for b in slider.branches.values()] == [12, 4, 4, 12]
    assert not hasattr(model.model.vision_model.q_proj, "_glimmer_slider")
    ids = torch.tensor([[1, 3, 6]])
    with torch.no_grad():
        for branch in slider.branches.values():
            branch.lora_up.weight.normal_()
    with slider.scaled(0.):
        assert torch.equal(model(ids).logits, baseline(ids).logits)
    with slider.scaled(1.):
        first = model(ids).logits.detach().clone()
        assert torch.equal(first, model(ids).logits)
        assert not torch.equal(first, baseline(ids).logits)
        with torch.no_grad():
            slider.particles.add_(2.)
        assert not torch.equal(first, model(ids).logits)
    assert slider.strength == 0.
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in slider.parameters())
    slider.detach()
    assert torch.equal(model(ids).logits, baseline(ids).logits)


def test_direct_game_and_training_updates(original_api, monkeypatch):
    model, baseline, slider, trainer = setup()
    assert Path(original_api.game.update.__code__.co_filename).is_relative_to(original_api.root)
    assert Path(inspect.unwrap(original_api.fit_routed_down).__code__.co_filename).is_relative_to(original_api.root)
    called = []
    update = original_api.game.update
    def observed(*args, **kwargs):
        called.append(kwargs["step"])
        return update(*args, **kwargs)
    monkeypatch.setattr(original_api.game, "update", observed)
    before = {k: v.clone() for k, v in slider.state_dict().items()}
    result = trainer.step()
    assert called == [0]
    assert result["particle_gan_grad_norm"] >= 0
    assert all(torch.isfinite(torch.tensor(result[k])) for k in ("g_adv", "d_loss", "d_pen"))
    assert any(not torch.equal(before[k], v) for k, v in slider.state_dict().items())
    assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in model.state_dict().items())
    assert set(trainer.metrics()) == {"step", "off_rms", "raw_rms", "ema_rms", "evaluation_sha256"}


def test_full_resume_is_exact(tmp_path):
    model, _, slider, trainer = setup()
    for _ in range(4):
        trainer.step()
    expected = copy.deepcopy(slider.state_dict())
    critic = copy.deepcopy(trainer.critic.state_dict())
    expected_metrics = trainer.metrics()
    _, _, _, split = setup()
    split.step(); split.step()
    store = OutputStore(tmp_path)
    split.save(store, "resume.pt")
    model2, _, slider2, resumed = setup(seed=99)
    # Same frozen base is required; a different random base cannot fake resume.
    model2.load_state_dict(model.state_dict())
    resumed = Trainer(model2, slider2, prepare(model2, slider2, PAIRS),
                      prepare(model2, slider2, HELD_OUT), horizon=8, seed=3)
    resumed.restore(tmp_path / "resume.pt")
    assert resumed.step_index == 2
    resumed.step(); resumed.step()
    assert all(torch.equal(expected[k], v) for k, v in slider2.state_dict().items())
    assert all(torch.equal(critic[k], v) for k, v in resumed.critic.state_dict().items())
    assert all(torch.equal(trainer.ema[k], v) for k, v in resumed.ema.items())
    assert resumed.metrics() == expected_metrics
    assert torch.equal(trainer.sampler.state_dict()["rngs"]["noise"], resumed.sampler.state_dict()["rngs"]["noise"])


def test_trained_teacher_reload_zero_and_identity(tmp_path):
    model, baseline, slider, trainer = setup()
    trainer.step(); trainer.step()
    store = OutputStore(tmp_path)
    save_adapter(store, "teacher.safetensors", slider)
    ids = torch.tensor([[1, 3, 6]])
    with slider.scaled(1.):
        expected = model(ids).logits.detach().clone()
    slider.detach()
    with pytest.raises(ValueError, match="mismatch"):
        load_adapter(model, dict(IDENTITY, revision="1"*40), tmp_path / "teacher.safetensors")
    restored, record = load_adapter(model, IDENTITY, tmp_path / "teacher.safetensors")
    with restored.scaled(0.):
        assert torch.equal(model(ids).logits, baseline(ids).logits)
    with restored.scaled(1.):
        assert torch.equal(model(ids).logits, expected)
    assert record["upstream_revision"] == REVISION
    assert torch.equal(slider.particles, restored.particles)


def test_upstream_ridge_keeps_up_and_ordinary_reload(tmp_path, original_api, monkeypatch):
    model, baseline, slider, trainer = setup()
    trainer.step(); trainer.step()
    calls = []
    fit = original_api.fit_routed_down
    def observed(x, y, **kw):
        calls.append((x.clone(), y.clone()))
        return fit(x, y, **kw)
    monkeypatch.setattr(original_api, "fit_routed_down", observed)
    inputs = [r["ids"] for r in trainer.training]
    state, metrics = distill(slider, model, inputs, max_rows=5)
    assert len(calls) == 4
    assert all(2 <= info["rows"] <= 5 for info in metrics.values())
    for key, branch in slider.branches.items():
        assert torch.equal(state[f"branches.{key}.lora_up.weight"], branch.lora_up.weight)
    store = OutputStore(tmp_path)
    save_adapter(store, "student.safetensors", slider, state=state, kind=STUDENT, calibration=metrics)
    assert all("bridge" not in k and k != "particles" for k in load_file(tmp_path / "student.safetensors"))
    slider.detach()
    student, _ = load_adapter(model, IDENTITY, tmp_path / "student.safetensors")
    assert student.ordinary and not hasattr(student, "particles")
    x, _ = calls[0]
    branch = student.branches["0"]
    with student.scaled(1.):
        actual = model.model.language_model.layers[0].self_attn.q_proj(x)
        expected = branch.org_forward(x) + (x @ branch.lora_down.weight.T) @ branch.lora_up.weight.T
        assert torch.allclose(actual, expected)
    with student.scaled(0.):
        assert torch.equal(model(inputs[0]).logits, baseline(inputs[0]).logits)


def test_data_alignment_and_resume_contract(tmp_path):
    model, _, slider, trainer = setup()
    row = trainer.training[0]
    assert row["ids"].tolist() == [[1, 3, 6, 7]]
    assert row["positive_ids"].tolist() == [[2, 4, 6, 7]]
    trainer.save(OutputStore(tmp_path), "state.pt")
    trainer.contract["horizon"] += 1
    with pytest.raises(ValueError, match="contract"):
        trainer.restore(tmp_path / "state.pt")
    with pytest.raises(ValueError, match="budget"):
        prepare(model, slider, PAIRS, max_tokens=2)
    with pytest.raises(ValueError, match="already"):
        Slider(model, IDENTITY)


@pytest.mark.parametrize("value", [[[1, 2]], [1, 2], {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}])
def test_tokenizer_return_contract(value):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["reasoning_strength"] == "low"
            return value
    assert prompt_ids(Tokenizer(), []).tolist() == [[1, 2]]
    with pytest.raises(ValueError, match="budget"):
        prompt_ids(Tokenizer(), [], max_tokens=1)


def test_sampling_is_live_and_keeps_raw_tokens():
    model, _, slider, _ = setup()
    class Tokenizer:
        response_template = {"start_anchor": "<|start|>assistant"}
        def decode(self, ids, **kwargs):
            assert kwargs == dict(skip_special_tokens=False)
            return "to=self<|message|>unfinished"
        def parse_response(self, raw):
            assert raw == "<|start|>assistantto=self<|message|>unfinished"
            return {"content": None}
    result = sample(model, Tokenizer(), slider, torch.tensor([[1, 2]]), strength=0., max_new_tokens=3)
    assert result["raw"].startswith("to=self")
    assert result["reached_budget"] and result["parsed"]["content"] is None
    assert len(result["token_ids"]) == 3


def test_storage_guards(tmp_path):
    store = OutputStore(tmp_path)
    for name in ("../escape", "/tmp/escape", "sub/file"):
        with pytest.raises(ValueError):
            with store.writing(name):
                pass
    with store.writing("safe.json") as path:
        path.write_text("{}")
    with pytest.raises(FileExistsError):
        with store.writing("safe.json"):
            pass
    assert (tmp_path / "safe.json").read_text() == "{}"
    (tmp_path / "link").symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        with store.writing("link"):
            pass
    with pytest.raises(OSError):
        with OutputStore(tmp_path, reserve_bytes=10**20).writing("other"):
            pass


def test_native_loader_refuses_unpinned_before_transformers_import(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        load_local(tmp_path)


def test_no_local_loss_fallback(original_api, tmp_path):
    (tmp_path / "not-a-checkout").mkdir()
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        load_upstream(tmp_path / "not-a-checkout")


def test_current_stamp_and_original_ar_game_are_equivalent(original_api):
    stamp, game = original_api.stamp, original_api.game
    stamp.require(stamp.as_dict())
    model, _, slider, trainer = setup()
    assert trainer.sampler.batch_size == stamp.spec["adv_batch"] == 8
    assert trainer.critic.noise_decay_steps == stamp.spec["noise_decay_steps"] == 1600
    assert [g["lr"] for g in trainer.g.param_groups] == [stamp.spec["g_lr"], stamp.spec["particle_lr"]]
    assert trainer.d.param_groups[0]["lr"] == stamp.spec["d_lr"]
    assert type(slider.branches["0"].bridge) is type(stamp.bridge())
    assert type(trainer.critic).__name__ == "GlobalMixErrorCritic"
    stamped_critic = stamp.critic(trainer.targets, torch.cat([r["neutral"] for r in trainer.training]))
    stamped_critic.load_state_dict(trainer.critic.state_dict())
    x = torch.randn(8, 12)
    assert torch.equal(trainer.critic(x), stamped_critic(x))
    assert torch.equal(trainer.critic.normalize(x), stamped_critic.normalize(x))
    d_loss, g_loss, vic = stamp.losses()
    real, fake = torch.randn(8), torch.randn(8)
    assert torch.equal(game.rp_d_loss(real, fake), d_loss(real, fake))
    assert torch.equal(game.rp_g_loss(real, fake), g_loss(real, fake))
    assert torch.equal(game.particle_vic(slider.particles), vic(slider.particles))
    from analysis.slider2d.adv import make_grad_regularizer
    regularizer = make_grad_regularizer(arm="b_cap", coeff=1., kappa=1., norm="l2", lazy_k=4, target_anneal="none")
    for step in (0, 1, 4):
        a, _ = regularizer.penalty(trainer.critic, x, x + .1, step=step, collect_stats=False)
        b, _ = stamp.regularizer().penalty(stamped_critic, x, x + .1, step=step, collect_stats=False)
        assert torch.equal(a, b)


def test_corrupt_teacher_rolls_back_attachment(tmp_path):
    from safetensors import safe_open
    model, _, slider, _ = setup()
    store = OutputStore(tmp_path)
    save_adapter(store, "good.safetensors", slider)
    slider.detach()
    with safe_open(tmp_path / "good.safetensors", framework="pt") as handle:
        metadata = handle.metadata()
    state = load_file(tmp_path / "good.safetensors")
    state["particles"][0, 0] = float("nan")
    save_file(state, tmp_path / "corrupt.safetensors", metadata=metadata)
    with pytest.raises(ValueError, match="Nonfinite"):
        load_adapter(model, IDENTITY, tmp_path / "corrupt.safetensors")
    restored, _ = load_adapter(model, IDENTITY, tmp_path / "good.safetensors")
    restored.detach()


def test_native_ram_gate_and_preflight_before_model(tmp_path, monkeypatch):
    import sys
    from glimmer_slider import MODEL_REVISION
    root = tmp_path / MODEL_REVISION
    root.mkdir()
    (root / "config.json").write_text(json.dumps(dict(model_type="muse_glimmer", architectures=["MuseGlimmerForConditionalGeneration"])))
    (root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"x": "part.safetensors"})))
    (root / "part.safetensors").write_bytes(b"fixture")
    with pytest.raises(MemoryError, match="refused"):
        load_local(root, max_weight_bytes=1)
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append("tokenizer")
            assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
            return object()
    class ModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("Must not load weights before prompt preflight")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Loader, MuseGlimmerForConditionalGeneration=ModelLoader))
    def reject(_tokenizer):
        calls.append("preflight")
        raise ValueError("fixture prompt over budget")
    with pytest.raises(ValueError, match="over budget"):
        load_local(root, preflight=reject)
    assert calls == ["tokenizer", "preflight"]
