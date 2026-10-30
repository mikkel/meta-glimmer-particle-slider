"""Native token callbacks and disconnect-aware streaming on the resident sampler."""
from dataclasses import asdict
import hashlib
import json
import os
import threading
import time
import uuid

import torch
from .restored_sampler import studio, prompt_tokens, conversation
from .channel_recovery import recover_explicit_channels
from .generation_scope import ScopeGate, active_channel

TERMINAL = ('done', 'failed', 'cancelled')


class AnswerStreamer:
    """Decode actual generated tokens; the official parser separates reasoning and final-answer regions.

    Full public-text snapshots permit detokenization corrections without duplicate
    characters. Incomplete UTF-8 replacement suffixes wait for the next token.
    """
    def __init__(self, tokenizer, prefix, emit, emit_reasoning=None, emit_diagnostic=None, on_channel=None):
        self.tokenizer, self.prefix, self.emit = tokenizer, prefix, emit
        self.on_channel=on_channel or (lambda channel: None)
        self.tokens, self.prompt_pending, self.decoded, self.text = [], True, '', ''
        self.reasoning_text, self.last_reasoning = '', ''
        self.official_text, self.official_reasoning = '', ''
        self.prefix_text=tokenizer.decode(prefix,skip_special_tokens=False)
        self.emit_diagnostic=emit_diagnostic or (lambda *_: None)
        self.framing=dict(status='pending_channel_header',warnings=[])
        self.last_diagnostic=None
        self.emit_reasoning = emit_reasoning or (lambda *_: None)
        self.parser = tokenizer.get_response_parser(prefix=prefix)

    def _events(self, events):
        for event in events:
            if event.get('field') == 'content' and event.get('type') == 'region_chunk':
                self.official_text += event['text']
            elif event.get('field') == 'reasoning_content' and event.get('type') == 'region_chunk':
                self.official_reasoning += event['text']

    def put(self, value):
        values = value.detach().cpu()
        if values.ndim == 2:
            if values.shape[0] != 1: raise ValueError('Streaming requires batch1')
            values = values[0]
        values = values.reshape(-1).tolist()
        if self.prompt_pending:
            self.prompt_pending = False
            if values != self.prefix: raise ValueError('Streamer prompt differs')
            return
        self.tokens.extend(values)
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False,
            clean_up_tokenization_spaces=False).rstrip('\ufffd')
        if not decoded.startswith(self.decoded):
            # A tokenizer may revise previous text while a multi-token character
            # or normalization completes. Replay the parser, then replace text.
            self.parser = self.tokenizer.get_response_parser(prefix=self.prefix)
            self.official_text = ''; self.official_reasoning = ''
            self._events(self.parser.feed(decoded))
        else:
            self._events(self.parser.feed(decoded[len(self.decoded):]))
        self.decoded = decoded
        self.on_channel(active_channel(self.prefix_text,decoded))
        self._display(decoded)

    def _display(self, decoded):
        recovery=recover_explicit_channels(decoded,self.prefix_text)
        self.text=self.official_text or recovery['content']
        self.reasoning_text=self.official_reasoning or recovery['reasoning']
        self.framing=dict(status=recovery['status'],warnings=recovery['warnings'],
            official_reasoning_characters=len(self.official_reasoning),official_answer_characters=len(self.official_text),
            recovered_reasoning_characters=len(self.reasoning_text) if not self.official_reasoning else 0,
            recovered_answer_characters=len(self.text) if not self.official_text else 0)
        self.emit(self.text,len(self.tokens))
        if self.reasoning_text!=self.last_reasoning:
            self.emit_reasoning(self.reasoning_text,len(self.tokens));self.last_reasoning=self.reasoning_text
        if recovery['status']!='standard':
            self.emit_diagnostic(decoded,len(self.tokens),self.framing)

    def end(self):
        decoded = self.tokenizer.decode(self.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != self.decoded:
            self.parser = self.tokenizer.get_response_parser(prefix=self.prefix)
            self.official_text = ''; self.official_reasoning = ''; self._events(self.parser.feed(decoded))
        _, events = self.parser.finalize()
        self._events(events)
        self._display(decoded)


class StreamingBackend(studio.NativeBackend):
    @torch.inference_mode()
    def generate(self, request, cancelled, *, snapshot=None, on_text=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        started = time.monotonic()
        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set() or time.monotonic() - started > 240
        cpu_ids = prompt_tokens(self.tokenizer, request)
        prefix = cpu_ids[0].tolist()
        ids = cpu_ids.to(self.device)
        gate = ScopeGate(self.sliders, dict(request.strengths), getattr(request,'scopes',()), snapshot, active_channel(self.tokenizer.decode(prefix,skip_special_tokens=False),''))
        streamer = AnswerStreamer(self.tokenizer, prefix, on_text or (lambda *_: None),
            lambda text,count: on_text(text,count,channel='reasoning') if on_text else None,
            lambda raw,count,framing: (on_text(raw,count,channel='raw'),on_text(framing,count,channel='protocol')) if on_text else None, on_channel=gate.select)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=request.max_new_tokens,
            do_sample=request.temperature > 0, use_cache=True, repetition_penalty=request.repetition_penalty,
            stopping_criteria=StoppingCriteriaList([Stop()]), streamer=streamer)
        if request.temperature > 0:
            kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=request.top_k)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        using = self.sliders.using(dict(request.strengths), snapshot=snapshot) if snapshot is not None else self.sliders.using(dict(request.strengths))
        with using, gate.using(), torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(request.seed)
            if devices: torch.cuda.default_generators[devices[0]].manual_seed(request.seed)
            tokens = self.model.generate(**kwargs)
        new = tokens[0, ids.shape[1]:].tolist()
        raw = self.tokenizer.decode(new, skip_special_tokens=False)
        parsed = self.tokenizer.parse_response(new, prefix=prefix)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        termination = ('cancelled' if cancelled.is_set() else 'eos' if new and new[-1] in eos
            else 'token_budget' if len(new) >= request.max_new_tokens else 'time_budget')
        content = (parsed or {}).get('content')
        return dict(raw=raw, parsed=parsed, text=content if isinstance(content,str) else streamer.text,
            reasoning_text=(parsed or {}).get('reasoning_content') or streamer.reasoning_text,framing=streamer.framing,
            token_ids=new, input_tokens=ids.shape[1], output_tokens=len(new),
            elapsed_seconds=round(time.monotonic()-started,3), termination=termination, model_loads=self.model_loads,
            service_base_loads=int(os.environ.get('GLIMMER_SERVICE_BASE_LOADS','1')),
            inference_system_condition=request.system_condition, effective_system_messages=conversation(request)[:-1],
            scope_gating=gate.report(), input_token_ids=prefix, artifact_format='glimmer-particle-teacher-v1', weights='raw')


class StreamingQueue(studio.SamplerQueue):
    def __init__(self, *args, **kwargs):
        self.stream_condition = threading.Condition(threading.RLock())
        self.request_ids = {}
        super().__init__(*args, **kwargs)
        self.stream_condition = threading.Condition(self.lock)

    def state(self):
        return dict(super().state(), streaming=True, stream_protocol='ndjson-v1', stream_text='reasoning-and-answer-separated', stream_channels=['reasoning','text','raw','protocol'], framing_recovery='explicit-recipient-only')

    def _add(self, job):
        if self.backend is None: raise studio.NotReadyError(self.status)
        if self.pending.full(): raise studio.BusyError('Generation queue is full; retry after a job completes')
        for key in list(self.jobs):
            if len(self.jobs)<self.history_limit: break
            previous=self.jobs[key]
            if previous['status'] in TERMINAL and not previous.get('_stream_clients'):
                del self.jobs[key]
        if len(self.jobs)>=self.history_limit: raise studio.BusyError('Job history is full of active requests')
        jid=uuid.uuid4().hex
        self.jobs[jid]=dict(id=jid,status='queued',cancelled=threading.Event(),**job)
        self.pending.put_nowait(jid)
        return self.get(jid)

    def submit(self, body):
        with self.lock:
            value=super().submit(body)
            job=self.jobs[value['id']]
            strengths=dict(job['request'].strengths)
            job['effective_sliders']=[dict(id=a['id'],name=a['title'],strength=strengths[a['id']],
                scope=dict(getattr(job['request'],'scopes',())).get(a['id'],'both'),
                step=a['step'],sha256=a['sha256'],definition_version=a.get('definition_version'),kind=a['kind'])
                for a in job['artifacts'] if strengths.get(a['id'],0)!=0]
            job['effective_mix']='ON' if job['effective_sliders'] else 'OFF / base model'
            return self.get(value['id'])

    def submit_stream(self, body, request_id):
        if not isinstance(request_id,str) or len(request_id)!=32 or any(c not in '0123456789abcdef' for c in request_id):
            raise ValueError('X-Request-ID must be32 lowercase hex characters')
        fingerprint = hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()
        with self.lock:
            self.request_ids = {key:value for key,value in self.request_ids.items() if value[1] in self.jobs}
            existing = self.request_ids.get(request_id)
            if existing:
                if existing[0] != fingerprint: raise ValueError('Request ID reused with different settings')
                return self.get(existing[1])
            job = self.submit(body)
            self.request_ids[request_id] = (fingerprint, job['id'])
            return job

    def events(self, jid, cursor, timeout=2.):
        with self.stream_condition:
            self.stream_condition.wait_for(lambda:len(self.jobs[jid].get('_events',[]))>cursor or self.jobs[jid]['status'] in TERMINAL, timeout)
            job=self.jobs[jid]
            events=job.get('_events',[])[cursor:]
            return events, cursor+len(events), self.get(jid)

    def _run(self):
        while True:
            jid = self.pending.get()
            try:
                with self.lock:
                    job = self.jobs[jid]
                    if job['cancelled'].is_set(): continue
                    job.update(status='running', _events=[])
                def emit(text, count, channel='text'):
                    with self.stream_condition:
                        job['_events'].append(dict(type=channel, text=text, output_tokens=count,
                            elapsed_seconds=round(time.monotonic()-started,3)))
                        key={'reasoning':'partial_reasoning','text':'partial_text','raw':'partial_raw','protocol':'stream_framing'}[channel]
                        job.update({key:text,'partial_output_tokens':count})
                        self.stream_condition.notify_all()
                started=time.monotonic()
                result=self.backend.generate(job['request'],job['cancelled'],snapshot=job.get('_snapshot'),on_text=emit)
                with self.stream_condition:
                    job.update(status='cancelled' if job['cancelled'].is_set() else 'done',result=result)
                    self.stream_condition.notify_all()
            except Exception as error:
                with self.stream_condition:
                    job.update(status='cancelled' if job['cancelled'].is_set() else 'failed',error=f'{type(error).__name__}: {error}')
                    self.stream_condition.notify_all()
            finally:
                with self.lock:
                    job.pop('_snapshot',None); job.pop('_candidate',None)
                    if not job.get('_stream_clients') and job['status'] in TERMINAL:
                        job['_events']=[]  # Final metadata/partial text remain; no token-snapshot history leak.
                self.pending.task_done()


def streaming_handler_for(runtime, host, port):
    parent=studio.handler_for(runtime,host,port)
    authority=f'{host}:{port}'
    class Handler(parent):
        def do_POST(self):
            if self.path!='/api/stream': return super().do_POST()
            if self.headers.get('Host')!=authority or self.headers.get('Origin') not in (None,'http://'+authority):
                return self.reply(403,dict(error='Expected private sampler origin'))
            try:
                if self.headers.get('Content-Type')!='application/json': raise ValueError('Expected application/json')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536: raise ValueError('Request size must be1..65536bytes')
                job=runtime.submit_stream(json.loads(self.rfile.read(length)),self.headers.get('X-Request-ID'))
            except (ValueError,TypeError) as error: return self.reply(400,dict(error=str(error)))
            except studio.BusyError as error: return self.reply(429,dict(error=str(error)))
            except studio.NotReadyError as error: return self.reply(503,dict(error=str(error)))
            jid=job['id']
            disconnected=False
            with runtime.lock:
                runtime.jobs[jid]['_stream_clients']=runtime.jobs[jid].get('_stream_clients',0)+1
            def send(value):
                self.wfile.write((json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'));self.wfile.flush()
            try:
                self.connection.settimeout(5.)
                self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
                self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Connection','close');self.end_headers();self.close_connection=True
                send(dict(type='start',job=job));cursor=0
                while True:
                    events,cursor,current=runtime.events(jid,cursor)
                    for event in events: send(event)
                    if current['status'] in TERMINAL:
                        send(dict(type='complete',job=current));break
                    if not events:send(dict(type='heartbeat',status=current['status']))
            except (BrokenPipeError,ConnectionError,TimeoutError,OSError): disconnected=True
            finally:
                with runtime.lock:
                    current=runtime.jobs[jid];current['_stream_clients']-=1
                    if disconnected and current['_stream_clients']==0 and current['status'] not in TERMINAL:
                        runtime.cancel(jid)
                    if current['_stream_clients']==0 and current['status'] in TERMINAL:
                        current['_events']=[]
    return Handler
