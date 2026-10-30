"""Local NF4 demo loader with user-directed RAM telemetry, not RAM admission."""
import json
import os
from pathlib import Path

import torch

from . import MODEL_ID, MODEL_REVISION
from .admission import require_cuda_identity
from .weights import pread_state


def load_demo_local(snapshot, *, expected_gpu_uuid, guardian_pid, preflight=None,
                    max_weight_bytes=62_000_000_000):
    """Same pinned low-CPU-memory NF4 load; no MemAvailable/RSS threshold.

    A live external guardian still owns the device leases, deadline and VRAM
    ceiling. Actual allocation/loader errors propagate for owned cleanup.
    Existing conservative loaders and other projects are unchanged.
    """
    if (guardian_pid != os.getppid() or os.environ.get('GLIMMER_GUARDIAN') != str(guardian_pid)
            or os.environ.get('GLIMMER_RAM_POLICY') != 'telemetry_only'):
        raise ValueError('Demo NF4 requires its live explicit-policy guardian parent')
    root=Path(snapshot).resolve(strict=True)
    origin=root/'glimmer-origin.json'
    declared=json.loads(origin.read_text()) if origin.exists() else None
    if root.name!=MODEL_REVISION and declared!=dict(model_id=MODEL_ID,revision=MODEL_REVISION):
        raise ValueError('Local snapshot must identify the pinned Glimmer revision')
    config=json.loads((root/'config.json').read_text())
    if config.get('model_type')!='muse_glimmer' or config.get('architectures')!=['MuseGlimmerForConditionalGeneration']:
        raise ValueError('Unsupported Glimmer architecture')
    index=json.loads((root/'model.safetensors.index.json').read_text())
    files=sorted(set(index['weight_map'].values()))
    if not files or any(Path(f).name!=f or not f.endswith('.safetensors') for f in files):
        raise ValueError('Unsafe or empty checkpoint index')
    total=sum((root/f).stat().st_size for f in files)
    if total>max_weight_bytes:
        raise ValueError('Cached checkpoint exceeds the unchanged weight-byte budget')
    require_cuda_identity(expected_gpu_uuid)
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig, BitsAndBytesConfig, MuseGlimmerForConditionalGeneration
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if preflight is not None:preflight(tokenizer)
    config=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    config._name_or_path=str(root)
    kwargs=dict(local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16))
    with pread_state(root,files) as state:
        model,info=MuseGlimmerForConditionalGeneration.from_pretrained(None,config=config,state_dict=state,
            output_loading_info=True,**kwargs)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'Native weight loading incomplete: {info}')
    model.generation_config=GenerationConfig.from_pretrained(root,local_files_only=True)
    model.eval()
    return model,tokenizer,dict(model_id=MODEL_ID,revision=MODEL_REVISION)
