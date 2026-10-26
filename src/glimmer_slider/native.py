"""Optional local open-weight Transformers entry points; no network fallback."""
from collections.abc import Mapping
import json
from pathlib import Path

import torch

from . import MODEL_ID, MODEL_REVISION


def prompt_ids(tokenizer, messages, *, max_tokens=128, reasoning_strength="low"):
    if reasoning_strength not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("Unsupported Glimmer reasoning strength")
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          reasoning_strength=reasoning_strength, return_tensors="pt")
    if isinstance(value, Mapping):
        value = value["input_ids"]
    ids = torch.as_tensor(value, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= max_tokens:
        raise ValueError("Expected one nonempty prompt within the token budget")
    return ids.cpu()


def load_local(snapshot, *, max_weight_bytes=2*1024**3, reserve_bytes=16*1024**3, preflight=None):
    """Only a locally staged, explicitly pinned snapshot, with conservative RAM gate.

    No quantized or CUDA path is qualified by this release. Default byte budget
    intentionally refuses the 30B model; a provisioned CPU host needs an explicit
    larger budget. No auto-download, credentials, remote code, or API training.
    """
    root = Path(snapshot).resolve(strict=True)
    if root.name != MODEL_REVISION:
        raise ValueError("Local snapshot directory must name the pinned model revision")
    config = json.loads((root / "config.json").read_text())
    if (config.get("model_type") != "muse_glimmer"
            or config.get("architectures") != ["MuseGlimmerForConditionalGeneration"]):
        raise ValueError("Unsupported Glimmer model architecture")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    if not files or any(Path(name).name != name or not name.endswith(".safetensors") for name in files):
        raise ValueError("Unsafe or empty checkpoint index")
    total = sum((root / name).stat().st_size for name in files)
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    # Float32 CPU base plus conservative staging headroom; no mmap-reclaim assumption.
    required = total * 3 + reserve_bytes
    if total > max_weight_bytes or available is None or available < required:
        raise MemoryError(f"CPU load refused: weights={total}, required_available={required}, available={available}")
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if preflight is not None:
        preflight(tokenizer)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        device_map={"": "cpu"}, low_cpu_mem_usage=True)
    model.eval()
    return model, tokenizer, dict(model_id=MODEL_ID, revision=MODEL_REVISION)


@torch.no_grad()
def sample(model, tokenizer, slider, ids, *, strength=1., max_new_tokens=128):
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("Generation budget must be 1..512 tokens")
    with slider.scaled(strength):
        tokens = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    new = tokens[0, ids.shape[1]:].tolist()
    raw = tokenizer.decode(new, skip_special_tokens=False)
    # Keep native format intact. A truncated reasoning trace is not a final answer.
    parsed = None
    if callable(getattr(tokenizer, "parse_response", None)):
        # The prompt already contains this open assistant header; generated IDs
        # start inside it. The pinned native response template needs its anchor.
        template = getattr(tokenizer, "response_template", None)
        if template is None:
            template = getattr(tokenizer, "init_kwargs", {}).get("response_template", {})
        anchor = template.get("start_anchor", "<|start|>assistant") if isinstance(template, dict) else "<|start|>assistant"
        parsed = tokenizer.parse_response(anchor + raw)
    return dict(token_ids=new, raw=raw, parsed=parsed,
                reached_budget=len(new) == max_new_tokens, strength=float(strength))
