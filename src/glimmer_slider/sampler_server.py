"""Private sampler HTTP UI and bounded, serialized generation queue."""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import copy
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import torch

from .sampler import ResidentSliders, discover, verify_final_ordinary


MAX_NEW_TOKENS = 4096
MAX_PROMPT_TOKENS = 4096
CONTEXT_BUDGET = 8192


@dataclass(frozen=True)
class Request:
    system: str
    prompt: str
    strengths: tuple
    seed: int = 7
    max_new_tokens: int = 256
    temperature: float = .8
    top_p: float = .95
    top_k: int = 50
    repetition_penalty: float = 1.
    mode: str = 'direct_final'
    reasoning: str = 'low'

    @classmethod
    def parse(cls, body, available):
        if not isinstance(body, dict) or set(body)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown request setting')
        values = dict(body)
        system, prompt = values.get('system', ''), values.get('prompt', '')
        if not isinstance(system, str) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('A nonempty text prompt is required')
        if len(system) > 12000 or len(prompt) > 16000:
            raise ValueError('Prompt character budget exceeded')
        strengths = values.get('strengths', {})
        if not isinstance(strengths, dict) or set(strengths)-set(available):
            raise ValueError('Unknown slider ID')
        for value in strengths.values():
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or abs(value)>8:
                raise ValueError('Signed strengths must be finite within -8..8')
        values.update(system=system, prompt=prompt, strengths=tuple(sorted((k,float(v)) for k,v in strengths.items() if v)))
        result = cls(**values)
        for field, low, high in [('seed',0,2**32-1),('max_new_tokens',1,MAX_NEW_TOKENS),('top_k',0,200)]:
            value = getattr(result,field)
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f'{field} must be an integer in {low}..{high}')
        for field, low, high in [('temperature',0,2),('top_p',.01,1),('repetition_penalty',.8,2)]:
            value = getattr(result,field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{field} must be finite in {low}..{high}')
        if result.mode not in ('direct_final','reasoning') or result.reasoning not in ('low','medium','high','xhigh'):
            raise ValueError('Unsupported native prompt mode or reasoning strength')
        return result


def prompt_tokens(tokenizer, request, context_cap=CONTEXT_BUDGET):
    messages = ([dict(role='system',content=request.system)] if request.system else []) + [dict(role='user',content=request.prompt)]
    if request.mode == 'direct_final':
        messages.append(dict(role='assistant',recipient='user',content=''))
        kwargs = dict(add_generation_prompt=False,continue_final_message=True)
    else:
        kwargs = dict(add_generation_prompt=True)
    value = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt',
        reasoning_strength=request.reasoning, **kwargs)
    if isinstance(value, Mapping):
        value = value['input_ids']
    ids = torch.as_tensor(value,dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim!=2 or ids.shape[0]!=1 or not 0<ids.shape[1]<=min(MAX_PROMPT_TOKENS, context_cap-request.max_new_tokens):
        raise ValueError(f'Prompt plus output exceeds the {context_cap} token context budget')
    return ids


class NativeBackend:
    def __init__(self, model, tokenizer, sliders):
        self.model, self.tokenizer, self.sliders = model, tokenizer, sliders
        self.model_loads = 1
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def generate(self, request, cancelled, *, snapshot=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        started = time.monotonic()
        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set() or time.monotonic()-started > 240
        ids = prompt_tokens(self.tokenizer,request).to(self.device)
        kwargs = dict(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=request.max_new_tokens,
            do_sample=request.temperature>0,use_cache=True,repetition_penalty=request.repetition_penalty,
            stopping_criteria=StoppingCriteriaList([Stop()]))
        if request.temperature>0:
            kwargs.update(temperature=request.temperature,top_p=request.top_p,top_k=request.top_k)
        devices = [self.device.index or 0] if self.device.type=='cuda' else []
        # Only the queue worker calls this method. Preserve CPU and our one visible
        # device's RNG; never seed or enumerate unrelated physical GPUs.
        using=(self.sliders.using(dict(request.strengths),snapshot=snapshot) if snapshot is not None
               else self.sliders.using(dict(request.strengths)))
        with using, torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(request.seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(request.seed)
            tokens = self.model.generate(**kwargs)
        new = tokens[0,ids.shape[1]:].tolist()
        raw = self.tokenizer.decode(new,skip_special_tokens=False)
        parsed = None
        if callable(getattr(self.tokenizer,'parse_response',None)):
            parsed = self.tokenizer.parse_response(new,prefix=ids[0].tolist())
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos or []
        termination = ('cancelled' if cancelled.is_set() else 'eos' if new and new[-1] in eos
                       else 'token_budget' if len(new)>=request.max_new_tokens else 'time_budget')
        content = (parsed or {}).get('content')
        return dict(raw=raw, parsed=parsed, text=content if isinstance(content,str) else raw,
                    token_ids=new, input_tokens=ids.shape[1],output_tokens=len(new),
                    elapsed_seconds=round(time.monotonic()-started,3),termination=termination,
                    model_loads=self.model_loads,service_base_loads=int(os.environ.get('GLIMMER_SERVICE_BASE_LOADS','1')))

    @torch.inference_mode()
    def register_artifact(self, artifact, *, reload_adapter=False):
        request=Request.parse(dict(system='Answer briefly.',prompt='Describe a quiet morning.',strengths={},
            max_new_tokens=8),[a.id for a in self.sliders.artifacts])
        ids=prompt_tokens(self.tokenizer,request).to(self.device)
        pointers=tuple(parameter.data_ptr() for parameter in self.model.parameters())
        def probe():
            return self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits.cpu()
        started=time.monotonic()
        result=self.sliders.register(artifact,probe=probe,reload_adapter=reload_adapter)
        unchanged=pointers==tuple(parameter.data_ptr() for parameter in self.model.parameters())
        if not unchanged:
            raise RuntimeError('Base parameter storage changed during adapter registration')
        result.setdefault('checks',{})['base_storage_unchanged']=unchanged
        return dict(result,elapsed_seconds=round(time.monotonic()-started,3),model_loads=self.model_loads,
            service_base_loads=int(os.environ.get('GLIMMER_SERVICE_BASE_LOADS','1')))


class BusyError(Exception): pass
class NotReadyError(Exception): pass


class SamplerQueue:
    def __init__(self, artifacts, backend=None, *, max_pending=8, history_limit=64, qualification_root=None):
        self.artifacts = tuple(artifacts)
        self.backend = backend
        self.status = 'ready' if backend else 'Waiting for coordinated GPU handoff; CPU interface only'
        self.jobs = OrderedDict()
        self.pending = queue.Queue(max_pending)
        self.history_limit = history_limit
        self.qualification_root=Path(qualification_root).resolve() if qualification_root else None
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self._run,name='glimmer-generation',daemon=True)
        self.thread.start()

    def state(self):
        with self.lock:
            bank=getattr(self.backend,'sliders',None)
            snapshot=bank.snapshot() if callable(getattr(bank,'snapshot',None)) else None
            artifacts=snapshot.artifacts if snapshot else self.artifacts
            return dict(ready=self.backend is not None,status=self.status,model_loads=getattr(self.backend,'model_loads',0),
                queued=self.pending.qsize(),running=sum(j['status']=='running' for j in self.jobs.values()),
                sliders=[asdict(a) | {'path':Path(a.path).name} for a in artifacts],
                catalog_revision=snapshot.revision if snapshot else 0,
                adapter_registration=self.qualification_root is not None and hasattr(self.backend,'register_artifact'),
                ui_live_file=True,
                service_base_loads=int(os.environ.get('GLIMMER_SERVICE_BASE_LOADS','1')),
                lease_end_UTC=os.environ.get('GLIMMER_LEASE_END_UTC'),
                limits=dict(max_new_tokens=MAX_NEW_TOKENS,max_prompt_tokens=MAX_PROMPT_TOKENS,context=CONTEXT_BUDGET,strength=[-8,8],generation_seconds=240))

    def _add(self, job):
        # Caller holds lock. Both generation and refresh share the same queue.
        if self.backend is None:
            raise NotReadyError(self.status)
        if self.pending.full():
            raise BusyError('Generation queue is full; retry after a job completes')
        # Never evict a running or pending job, so cancellation remains valid.
        for key in list(self.jobs):
            if len(self.jobs)<self.history_limit:
                break
            if self.jobs[key]['status'] in ('done','failed','cancelled'):
                del self.jobs[key]
        if len(self.jobs)>=self.history_limit:
            raise BusyError('Job history is full of active requests')
        jid = uuid.uuid4().hex
        self.jobs[jid] = dict(id=jid,status='queued',cancelled=threading.Event(),**job)
        self.pending.put_nowait(jid)
        return self.get(jid)

    def submit(self, body):
        with self.lock:
            request = Request.parse(body,[a.id for a in self.artifacts])
            bank=getattr(self.backend,'sliders',None)
            snapshot=bank.snapshot() if callable(getattr(bank,'snapshot',None)) else None
            artifacts=snapshot.artifacts if snapshot else self.artifacts
            return self._add(dict(kind='generation',request=request,_snapshot=snapshot,
                catalog_revision=snapshot.revision if snapshot else 0,
                artifacts=[asdict(a)|{'path':Path(a.path).name} for a in artifacts]))

    def submit_refresh(self, body):
        if (not isinstance(body,dict) or set(body)-{'id','reload_adapter'} or not isinstance(body.get('id'),str)
                or type(body.get('reload_adapter',False)) is not bool):
            raise ValueError('Refresh accepts an existing slider id and optional boolean reload_adapter')
        if self.qualification_root is None or not hasattr(self.backend,'register_artifact'):
            raise NotReadyError('Adapter registration is not configured')
        with self.lock:
            if body['id'] not in {a.id for a in self.artifacts}:
                raise ValueError('Unknown slider id')
        try:
            candidate=next(a for a in discover(self.qualification_root) if a.id==body['id'])
            verify_final_ordinary(candidate)
        except (OSError,StopIteration) as error:
            raise ValueError('Qualified final ordinary export is not available') from error
        with self.lock:
            return self._add(dict(kind='adapter_refresh',request=dict(body),_candidate=candidate,
                candidate=asdict(candidate)|{'path':Path(candidate.path).name}))

    def get(self, jid):
        with self.lock:
            job = self.jobs[jid]
            return copy.deepcopy({k:(asdict(v) if isinstance(v,Request) else v) for k,v in job.items()
                                  if k!='cancelled' and not k.startswith('_')})

    def cancel(self, jid):
        with self.lock:
            job = self.jobs[jid]
            if job.get('kind')=='adapter_refresh' and job['status']=='running':
                raise BusyError('Adapter publication is running; observe its final receipt')
            job['cancelled'].set()
            if job['status']=='queued':
                job['status']='cancelled'
            return self.get(jid)

    def _run(self):
        while True:
            jid = self.pending.get()
            try:
                with self.lock:
                    job = self.jobs[jid]
                    if job['cancelled'].is_set():
                        continue
                    job['status']='running'
                if job.get('kind')=='adapter_refresh':
                    result=self.backend.register_artifact(job['_candidate'],reload_adapter=job['request'].get('reload_adapter',False))
                    with self.lock:
                        self.artifacts=self.backend.sliders.artifacts
                elif job.get('_snapshot') is not None:
                    result=self.backend.generate(job['request'],job['cancelled'],snapshot=job['_snapshot'])
                else:
                    result = self.backend.generate(job['request'],job['cancelled'])
                with self.lock:
                    job.update(status='cancelled' if job['cancelled'].is_set() else 'done',result=result)
            except Exception as error:
                with self.lock:
                    job.update(status='failed',error=f'{type(error).__name__}: {error}')
            finally:
                with self.lock:
                    job.pop('_snapshot',None);job.pop('_candidate',None)
                self.pending.task_done()


def handler_for(runtime, host, port):
    authority = f'{host}:{port}'
    page_path = Path(__file__).with_name('studio.html')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass  # Prompts are not persisted or logged.
        def reply(self,status,value,content_type='application/json'):
            data = value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.end_headers()
            self.wfile.write(data)
        def dispatch(self,method):
            # Same-origin browser actions only, with no wildcard CORS or public bind.
            if self.headers.get('Host') != authority or self.headers.get('Origin') not in (None,f'http://{authority}'):
                return self.reply(403,dict(error='Expected the private sampler origin'))
            try:
                if method=='GET' and self.path=='/':
                    return self.reply(200,page_path.read_bytes(),'text/html; charset=utf-8')
                if method=='GET' and self.path=='/api/state':
                    return self.reply(200,runtime.state())
                if self.path.startswith('/api/jobs/'):
                    jid = self.path.removeprefix('/api/jobs/')
                    if len(jid)!=32 or any(c not in '0123456789abcdef' for c in jid):
                        raise KeyError(jid)
                    if method=='GET':
                        return self.reply(200,runtime.get(jid))
                    if method=='DELETE':
                        return self.reply(200,runtime.cancel(jid))
                if method=='POST' and self.path in ('/api/jobs','/api/artifacts/refresh'):
                    if self.headers.get('Content-Type')!='application/json':
                        raise ValueError('Expected application/json')
                    length = int(self.headers.get('Content-Length','0'))
                    if not 0<length<=65536:
                        raise ValueError('Request size must be 1..65536 bytes')
                    body = json.loads(self.rfile.read(length))
                    return self.reply(202,runtime.submit_refresh(body) if self.path.endswith('/refresh') else runtime.submit(body))
                return self.reply(404,dict(error='Unknown endpoint'))
            except KeyError:
                return self.reply(404,dict(error='Job not found'))
            except (ValueError,TypeError) as error:
                return self.reply(400,dict(error=str(error)))
            except BusyError as error:
                return self.reply(429,dict(error=str(error)))
            except NotReadyError as error:
                return self.reply(503,dict(error=str(error)))
        def do_GET(self): self.dispatch('GET')
        def do_POST(self): self.dispatch('POST')
        def do_DELETE(self): self.dispatch('DELETE')
    return Handler
