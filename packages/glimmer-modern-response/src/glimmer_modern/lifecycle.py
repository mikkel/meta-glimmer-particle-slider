"""Registered complete response factory; the shared core owns every update."""
import hashlib
import json
from pathlib import Path

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, EvaluationRows, SessionConfig
from particle_sliders.pg09 import ConditionalPG09, PG_VERSION, PG_SOURCE_PROFILE
from particle_sliders.pg09_audio import AudioRow, audio_conditions
from particle_sliders.response import ResponseBackend, ResponseCapabilities

from . import BACKEND_NAME, PROTOCOL
from .fixture import TinyGlimmerCPU, make_model
from .native import Native, ReadoutCritic, validate_ids, model_identity
from .provenance import verify_sources

CAPS = ResponseCapabilities("routed_particle_gan", "glimmer_causal_text_hidden",
    "native_final_text_hidden_cpu_fixture", "continuous_vector", (1, 12),
    PG_VERSION, PG_SOURCE_PROFILE, base_precisions=("float32",),
    qualification="Synthetic causal CPU fixture only; pretrained Glimmer and wide vectors unqualified")


def _configuration(options):
    if options.get("mode") != "cpu-fixture":
        raise ValueError("native/wide Glimmer training awaits central source admission and model qualification")
    if options.get("width", 12) != 12 or options.get("base_precision", "float32") != "float32":
        raise ValueError("CPU fixture geometry/precision differs; no pooling or response reshape")
    if type(options.get("model_seed", 41)) is not int or not 0 <= options.get("model_seed", 41) < 2**31:
        raise ValueError("bounded integer model_seed required")
    if options.get("pg_source_profile", PG_SOURCE_PROFILE) != PG_SOURCE_PROFILE:
        raise ValueError("unqualified source profile")
    if options.get("penalty_input_domain", "auto") != "auto":
        raise ValueError("unqualified continuous override")


def read_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("bounded local paired-token JSON file required")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"train", "evaluation"}:
        raise ValueError("distinct train and evaluation row lists required")
    result = {}
    for split, rows in value.items():
        if not isinstance(rows, list) or not 2 <= len(rows) <= 32:
            raise ValueError("each split requires 2..32 fixed rows")
        result[split] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"neutral", "positive", "history"}:
                raise ValueError("each pair requires neutral, positive and shared history token IDs")
            for key in ("neutral", "positive"):
                validate_ids(row[key])
            history = row["history"]
            if not isinstance(history, list):
                raise ValueError("shared history must be token IDs")
            if history:
                validate_ids(history)
            neutral, positive = row["neutral"] + history, row["positive"] + history
            validate_ids(neutral)
            validate_ids(positive)
            if neutral == positive:
                raise ValueError("neutral and positive context must differ")
            result[split].append(AudioRow(tuple(neutral), tuple(positive), len(history)))
    training = {(r.neutral_ids, r.positive_ids) for r in result["train"]}
    if any((r.neutral_ids, r.positive_ids) in training for r in result["evaluation"]):
        raise ValueError("evaluation rows overlap training rows")
    return result


def validate(options):
    allowed = {"mode", "width", "base_precision", "model_seed", "data_file", "steps", "batch",
               "seed", "ema_decay", "pg_version", "pg_source_profile", "penalty_input_domain"}
    if set(options) - allowed:
        raise ValueError("unknown modern options")
    _configuration(options)
    read_rows(options["data_file"])
    verify_sources()


def _model(options, model=None):
    if model is not None and type(model) is not TinyGlimmerCPU:
        raise ValueError("only the explicit CPU protocol model is admitted; native qualification pending")
    expected = make_model(options.get("model_seed", 41))
    if model is not None and model_identity(model) != model_identity(expected):
        raise ValueError("borrowed fixture base differs from its reconstructible seed identity")
    return expected if model is None else model


def preflight(options, request, staged):
    if set(staged) - {"model"}:
        raise ValueError("unknown staged objects")
    if staged.get("model") is not None:
        _model(options, staged["model"])
    if request["native"]:
        raise ValueError("training response rows own fixed native token conditioning")


def prepare(options, *, model=None):
    rows = read_rows(options["data_file"])
    native = Native(_model(options, model))
    try:
        prepared = {key: [native.backend.prepare(row) for row in values] for key, values in rows.items()}
        conditions = audio_conditions(prepared["train"])
        critic = ReadoutCritic(12, conditions.shape[1])
        game = ConditionalPG09(native, prepared["train"], critic, conditions=conditions,
            particles=native.slider.particles, response_domain="continuous_vector",
            total_steps=options.get("steps", 4), batch_size=options.get("batch", 2),
            seed=options.get("seed", 7), pg_source_profile=PG_SOURCE_PROFILE)
        configuration = dict(mode="cpu-fixture", width=12, model_seed=options.get("model_seed", 41))
        spec = INFERENCE.specification(native, model_identity=native.identity,
            protocol=PROTOCOL, configuration=configuration)
        identity = dict(protocol="paired-text-held-out-v1", sha256=hashlib.sha256(
            json.dumps([dict(neutral=list(row.neutral_ids), positive=list(row.positive_ids),
                             history_tokens=row.history_tokens) for row in rows["evaluation"]],
                       sort_keys=True).encode()).hexdigest())
        return Binding(game, native.close, inference=spec, observation=native.observation,
            evaluation_rows={"evaluation": EvaluationRows(prepared["evaluation"], identity)})
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if options:
        raise ValueError("CPU fixture native-only reload requires no training/runtime inputs")
    if set(spec.configuration) != {"mode", "width", "model_seed"}:
        raise ValueError("native-only fixture construction contract differs")
    _configuration(spec.configuration)
    verify_sources()


def inference_preflight(options, request, spec):
    if set(request["native"]) != {"ids"}:
        raise ValueError("native response requires explicit bounded token IDs")
    validate_ids(request["native"]["ids"])


def prepare_inference(options, spec):
    native = Native(_model(spec.configuration))
    try:
        identity = native_identity(native, model_identity=native.identity, protocol=PROTOCOL)
        return InferenceBinding(native.network, identity, native.close,
                                response=native.response, observation=native.observation)
    except BaseException:
        native.close()
        raise


BACKEND = ResponseBackend(BACKEND_NAME, CAPS, prepare, validate, preflight=preflight)
INFERENCE = InferenceBackend(BACKEND_NAME, CAPS, prepare_inference, validate_inference,
                             preflight=inference_preflight)


def cpu_config(data_file, *, steps=4, seed=7, model_seed=41):
    return SessionConfig(BACKEND_NAME, PROTOCOL, mode="cpu-fixture", width=12,
        data_file=str(Path(data_file).resolve()), steps=steps, batch=2, seed=seed,
        model_seed=model_seed, ema_decay=.8)
