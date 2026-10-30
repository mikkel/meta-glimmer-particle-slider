"""Shared explicit-neutral request and verified particle-catalog protocol."""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import json
import torch
from . import sampler_server as studio
from .sampler import describe

ORIGINAL = 'Write in plain, modern, neutral prose. No archaic words, no drama.'
BLANK_ID = 'medieval-blank-1024'


def particle_artifacts(artifacts):
    """Keep distilled exports on disk; never admit them to this sampler."""
    return [artifact for artifact in artifacts if artifact.kind == 'glimmer-particle-teacher-v1']


def catalog_particles(path):
    catalog_path = Path(path).resolve()
    records = json.loads(catalog_path.read_text())
    artifacts = []
    for row in records['entries']:
        artifact = describe(catalog_path.parent / row['path'], id=row['id'], title=row['title'], qualification=row['qualification'])
        if (artifact.kind != 'glimmer-particle-teacher-v1' or artifact.sha256 != row['sha256']
                or artifact.step != row['step']):
            raise ValueError('Particle catalog source differs; no distilled fallback')
        artifacts.append(replace(artifact, quality_note=row['quality_note'], definition_version=row.get('definition_version')))
    return artifacts


@dataclass(frozen=True)
class LabeledRequest(studio.Request):
    system_condition: str = 'explicit_blank'
    scopes: tuple = ()

    @classmethod
    def parse(cls, body, available):
        if not isinstance(body, dict):
            raise ValueError('Expected an object')
        values = dict(body)
        values.setdefault('system_condition', 'custom' if values.get('system') else 'explicit_blank')
        scopes = values.pop('scopes', {})
        if not isinstance(scopes, dict) or set(scopes)-set(available) or any(v not in ('thinking','response','both') for v in scopes.values()):
            raise ValueError('Scopes require known slider IDs and thinking/response/both')
        result = replace(super().parse(values, available), scopes=tuple(sorted(scopes.items())))
        if result.system_condition not in ('explicit_blank', 'original', 'omitted', 'custom'):
            raise ValueError('Unknown inference system condition')
        return result


def conversation(request):
    condition = request.system_condition
    value = '' if condition == 'explicit_blank' else ORIGINAL if condition == 'original' else request.system
    return ([] if condition == 'omitted' else [dict(role='system', content=value)]) + [dict(role='user', content=request.prompt)]


def prompt_tokens(tokenizer, request, context_cap=studio.CONTEXT_BUDGET):
    messages = conversation(request)
    if request.mode == 'direct_final':
        messages.append(dict(role='assistant', recipient='user', content=''))
        kwargs = dict(add_generation_prompt=False, continue_final_message=True)
    else:
        kwargs = dict(add_generation_prompt=True)
    value = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt',
        reasoning_strength=request.reasoning, current_date='2026-10-27', **kwargs)
    if isinstance(value, Mapping): value = value['input_ids']
    ids = torch.as_tensor(value, dtype=torch.long)
    if ids.ndim == 1: ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= min(studio.MAX_PROMPT_TOKENS, context_cap-request.max_new_tokens):
        raise ValueError('Prompt plus output exceeds context budget')
    return ids


studio.Request = LabeledRequest
studio.prompt_tokens = prompt_tokens
