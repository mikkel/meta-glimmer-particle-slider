"""Explicit configuration-driven local training, live sampling and ridge export."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .upstream import load_upstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "sample", "distill"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    load_upstream(cfg["particle_sliders"])
    from .host import Slider, distill
    from .native import load_local, prompt_ids, sample
    from .storage import OutputStore, STUDENT, load_adapter, save_adapter
    from .training import Trainer, prepare
    def preflight(tokenizer):
        messages = []
        if args.command == "train":
            if not 1 <= cfg["updates"] <= cfg["horizon"] or cfg["checkpoint_every"] < 1:
                raise ValueError("Require positive bounded update/checkpoint budgets")
            for row in cfg["training"] + cfg["evaluation"]:
                limit = cfg["max_tokens"] - len(row.get("history", []))
                prompt_ids(tokenizer, row["neutral"], max_tokens=limit)
                prompt_ids(tokenizer, row["positive"], max_tokens=limit)
            messages = cfg.get("generation_evaluation", [])
        elif args.command == "sample":
            messages = [cfg["messages"]]
        else:
            messages = cfg["calibration"]
        for item in messages:
            prompt_ids(tokenizer, item, max_tokens=cfg["max_tokens"])
    model, tokenizer, identity = load_local(cfg["snapshot"], max_weight_bytes=cfg.get("max_weight_bytes", 2*1024**3), preflight=preflight)
    store = OutputStore(cfg["output_dir"])
    torch.manual_seed(cfg.get("seed", 7))
    if args.command == "train":
        slider = Slider(model, identity)
        def pairs(records):
            return [dict(neutral=prompt_ids(tokenizer, r["neutral"], max_tokens=cfg["max_tokens"])[0].tolist(),
                         positive=prompt_ids(tokenizer, r["positive"], max_tokens=cfg["max_tokens"])[0].tolist(),
                         history=r.get("history", [])) for r in records]
        train = prepare(model, slider, pairs(cfg["training"]), max_tokens=cfg["max_tokens"])
        evaluation = prepare(model, slider, pairs(cfg["evaluation"]), max_tokens=cfg["max_tokens"])
        protocol = dict(generation_evaluation=cfg.get("generation_evaluation", []),
                        max_new_tokens=cfg["max_new_tokens"], decoding="greedy", reasoning_strength="low")
        trainer = Trainer(model, slider, train, evaluation, horizon=cfg["horizon"], seed=cfg.get("seed", 7), evaluation_protocol=protocol)
        if cfg.get("resume"):
            trainer.restore(cfg["resume"])
        def report():
            receipt = trainer.metrics()
            receipt["generation"] = {}
            for index, record in enumerate(cfg.get("generation_evaluation", [])):
                ids = prompt_ids(tokenizer, record, max_tokens=cfg["max_tokens"])
                outputs = {"off": sample(model, tokenizer, slider, ids, strength=0., max_new_tokens=cfg["max_new_tokens"]),
                           "raw": sample(model, tokenizer, slider, ids, strength=1., max_new_tokens=cfg["max_new_tokens"])}
                with slider.using_state(trainer.ema):
                    outputs["ema"] = sample(model, tokenizer, slider, ids, strength=1., max_new_tokens=cfg["max_new_tokens"])
                receipt["generation"][str(index)] = outputs
            name = f"metrics-{trainer.step_index:06d}.json"
            existing = store.root / name
            if existing.is_symlink():
                raise ValueError("Refuse a symlinked metric receipt")
            if existing.exists():
                if json.loads(existing.read_text()) != receipt:
                    raise ValueError("Existing metric receipt differs; refusing overwrite")
            else:
                with store.writing(name) as path:
                    path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)
        report()
        stop = min(trainer.step_index + cfg["updates"], trainer.horizon)
        while trainer.step_index < stop:
            print(json.dumps(trainer.step()), flush=True)
            if trainer.step_index % cfg["checkpoint_every"] == 0 or trainer.step_index == stop:
                report()
                trainer.save(store, f"resume-{trainer.step_index:06d}.pt")
        provenance = dict(step=trainer.step_index, contract=trainer.contract)
        save_adapter(store, f"teacher-raw-{trainer.step_index:06d}.safetensors", slider, training=dict(provenance, weights="raw"))
        save_adapter(store, f"teacher-ema-{trainer.step_index:06d}.safetensors", slider, state=trainer.ema, training=dict(provenance, weights="ema"))
    else:
        slider, _ = load_adapter(model, identity, cfg["adapter"])
        if args.command == "sample":
            ids = prompt_ids(tokenizer, cfg["messages"], max_tokens=cfg["max_tokens"])
            result = sample(model, tokenizer, slider, ids, strength=cfg.get("strength", 1.), max_new_tokens=cfg["max_new_tokens"])
            print(json.dumps(result), flush=True)
        else:
            inputs = [prompt_ids(tokenizer, m, max_tokens=cfg["max_tokens"]) for m in cfg["calibration"]]
            state, metrics = distill(slider, model, inputs, max_rows=cfg.get("calibration_rows", 128))
            provenance = dict(teacher_sha256=hashlib.sha256(Path(cfg["adapter"]).read_bytes()).hexdigest(),
                              input_ids=[x.tolist() for x in inputs], projections=metrics)
            save_adapter(store, "ordinary-lora.safetensors", slider, state=state, kind=STUDENT, calibration=provenance)


if __name__ == "__main__":
    main()
