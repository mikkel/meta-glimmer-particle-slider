"""Causal hidden-state host wiring to the original paired-error PG API."""
import hashlib
import json
import random
from pathlib import Path

import torch

from .storage import validate_state
from .upstream import api, REVISION


def hidden(model, ids):
    outputs = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    if not getattr(outputs, "hidden_states", None):
        raise RuntimeError("Differentiable text hidden states are unavailable")
    return outputs.hidden_states[-1][:, -1].float()


def token_tensor(ids):
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("Expected nonempty nonnegative token IDs")
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def prepare(model, slider, pairs, *, max_tokens=128):
    """Both prompts include exactly the same caller-provided causal history.

    Neutral/positive prefixes may differ; matching continuation IDs are required.
    Target is frozen raw positive final hidden state, as in the upstream AR host.
    No token KL, supervised loss or diffusion surrogate is introduced.
    """
    rows = []
    with slider.scaled(0.):
        for pair in pairs:
            history = pair.get("history", [])
            if not isinstance(history, list):
                raise ValueError("History must be token IDs")
            neutral = token_tensor(pair["neutral"] + history)
            positive = token_tensor(pair["positive"] + history)
            if max(neutral.shape[1], positive.shape[1]) > max_tokens:
                raise ValueError("Prompt plus history exceeds token budget")
            rows.append(dict(ids=neutral, positive_ids=positive,
                             neutral=hidden(model, neutral).detach().clone(),
                             target=hidden(model, positive).detach().clone()))
    if len(rows) < 2:
        raise ValueError("At least two fixed paired rows required")
    return rows


def data_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        for key in ("ids", "positive_ids", "neutral", "target"):
            value = row[key].detach().cpu().contiguous()
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


class Trainer:
    def __init__(self, model, slider, training, evaluation, *, horizon=1600, seed=7, evaluation_protocol=None):
        if slider.ordinary or horizon < 1:
            raise ValueError("PG training requires particle branches and a fixed positive horizon")
        if model.training:
            raise ValueError("Frozen host must be in eval mode (no dropout)")
        if not evaluation:
            raise ValueError("Fixed evaluation rows are required")
        self.model, self.slider = model, slider
        self.training, self.evaluation = training, evaluation
        self.targets = torch.cat([r["target"] for r in training])
        neutrals = torch.cat([r["neutral"] for r in training])
        self.horizon, self.step_index, self.seed = int(horizon), 0, int(seed)
        stamp = api().stamp
        spec = stamp.as_dict()
        stamp.require(spec)
        critic_config = {key: spec["critic_" + key] for key in ("tokens", "width", "layers", "heads", "score_bound")}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            self.critic, self.g, self.d = api().game.build_game(
                slider, self.targets, critic=spec["critic"], critic_config=critic_config, neutrals=neutrals)
        # The run's stop budget does not change the shared formulation's schedule.
        self.critic.noise_decay_steps = spec["noise_decay_steps"]
        self.sampler = api().game.BridgeSampler(len(training), seed, batch_size=spec["adv_batch"])
        self.ema = api().game.initialize_ema(slider)
        self.contract = dict(base=slider.spec, upstream_revision=REVISION,
                             train_sha256=data_digest(training), eval_sha256=data_digest(evaluation),
                             horizon=self.horizon, seed=self.seed,
                             recipe=spec, batch_size=spec["adv_batch"],
                             reference=json.loads(json.dumps(api().game.REFERENCE)),
                             evaluation_protocol=evaluation_protocol,
                             torch_version=str(torch.__version__))

    def step(self):
        if self.step_index >= self.horizon:
            raise ValueError("Declared training horizon reached")
        def predict(index, phase):
            with self.slider.scaled(1.):
                return self.critic.normalize(hidden(self.model, self.training[index]["ids"]))
        result = api().game.update(self.slider, self.critic, self.g, self.d, self.targets,
                                   predict, sampler=self.sampler, step=self.step_index)
        api().game.update_ema(self.ema, self.slider)
        self.step_index += 1
        return dict(step=self.step_index, **result)

    @torch.no_grad()
    def metrics(self):
        def measure(strength):
            with self.slider.scaled(strength):
                errors = [self.critic.normalize(hidden(self.model, row["ids"])) -
                          self.critic.normalize(row["target"]) for row in self.evaluation]
                return float(torch.cat(errors).square().mean().sqrt())
        raw = measure(1.)
        off = measure(0.)
        with self.slider.using_state(self.ema):
            ema = measure(1.)
        return dict(step=self.step_index, off_rms=off, raw_rms=raw, ema_rms=ema,
                    evaluation_sha256=self.contract["eval_sha256"])

    def save(self, store, name):
        record = dict(format="glimmer-pg-continuation-v1", contract=self.contract,
                      step=self.step_index, adapter=self.slider.state_dict(),
                      critic=self.critic.state_dict(), generator_optimizer=self.g.state_dict(),
                      discriminator_optimizer=self.d.state_dict(), sampler=self.sampler.state_dict(),
                      ema=self.ema, torch_rng=torch.get_rng_state(), python_rng=random.getstate())
        with store.writing(name, 1024**3) as path:
            with path.open("wb") as handle:
                torch.save(record, handle)

    def restore(self, path):
        if Path(path).stat().st_size > 1024**3:
            raise ValueError("Training checkpoint exceeds 1 GiB limit")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state.get("format") != "glimmer-pg-continuation-v1" or state.get("contract") != self.contract
                or type(state.get("step")) is not int or not 0 <= state["step"] <= self.horizon):
            raise ValueError("Training continuation contract differs")
        validate_state(state["adapter"], self.slider.state_dict())
        validate_state(state["ema"], self.ema)
        validate_state(state["critic"], self.critic.state_dict())
        self.slider.load_state_dict(state["adapter"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.g.load_state_dict(state["generator_optimizer"])
        self.d.load_state_dict(state["discriminator_optimizer"])
        self.sampler.load_state_dict(state["sampler"])
        self.ema = state["ema"]
        self.step_index = state["step"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
