from contextlib import contextmanager
from dataclasses import replace
from http.server import ThreadingHTTPServer
import json
import threading
import time
import urllib.request

import pytest
import torch

from glimmer_slider.sampler_stream import AnswerStreamer, StreamingBackend, StreamingQueue, streaming_handler_for
from glimmer_slider.restored_sampler import LabeledRequest
from glimmer_slider.sampler import Artifact, AdapterSnapshot


SCHEMA = dict(defaults=dict(role='assistant'), start_anchor='<|start|>assistant', fields={
    'content':dict(open_pattern=r'to=user<\|message\|>', close=['<|eot|>','<|eom|>'],content='text'),
    'reasoning_content':dict(open_pattern=r'to=self<\|message\|>',close='<|eom|>',content='text')})


class ByteTokenizer:
    def decode(self, values, **kwargs):
        pieces=[];buffer=bytearray()
        for value in values:
            if value>=1000:
                pieces.append(buffer.decode('utf-8',errors='replace'));buffer.clear()
                pieces.append({1000:'<|start|>assistant to=user<|message|>',1001:'<|start|>assistant',1002:'<|eot|>'}[value])
            else: buffer.append(value)
        return ''.join(pieces)+buffer.decode('utf-8',errors='replace')
    def get_response_parser(self, *, prefix):
        from transformers.utils.chat_parsing import ResponseParser
        return ResponseParser(SCHEMA,prefix=self.decode(prefix))
    def parse_response(self, values, *, prefix):
        parser=self.get_response_parser(prefix=prefix);parser.feed(self.decode(values));return parser.finalize()[0]
    def apply_chat_template(self, *args, **kwargs): return dict(input_ids=[[1000]])


def test_actual_incremental_parser_holds_unicode_and_hides_reasoning():
    seen=[];stream=AnswerStreamer(ByteTokenizer(),[1000],lambda text,n:seen.append((text,n)))
    stream.put(torch.tensor([[1000]]))
    data='café 🐈!'.encode()
    for i,value in enumerate(data):
        stream.put(torch.tensor([value]));assert '\ufffd' not in seen[-1][0]
        if i==0: assert seen[-1][0]=='c'  # Before generation ends.
    stream.put(torch.tensor([1002]));stream.end()
    assert seen[-1][0]=='café 🐈!' and seen[-1][1]==len(data)+1
    seen=[];stream=AnswerStreamer(ByteTokenizer(),[1001],lambda text,n:seen.append(text))
    stream.put(torch.tensor([[1001]]))
    text=' to=self<|message|>PRIVATE<|eom|><|start|>assistant to=user<|message|>Public café'
    for value in text.encode():stream.put(torch.tensor([value]))
    stream.put(torch.tensor([1002]));stream.end()
    assert seen[-1]=='Public café' and all('PRIVATE' not in text for text in seen)


class Bank:
    def __init__(self, artifacts): self.artifacts=tuple(artifacts);self.active=0;self.seen=[]
    def snapshot(self): return AdapterSnapshot(3,self.artifacts,{})
    @contextmanager
    def using(self, strengths, snapshot=None):
        self.active+=1;self.seen.append((strengths,snapshot))
        try:yield
        finally:self.active-=1


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1))
        self.generation_config=type('Config',(),dict(eos_token_id=1002))()
    def generate(self, **kwargs):
        streamer=kwargs['streamer'];ids=kwargs['input_ids'];streamer.put(ids)
        output=[]
        for value in list('Real café 🐈'.encode())+[1002]:
            output.append(value);streamer.put(torch.tensor([value]))
            if kwargs['stopping_criteria'](torch.tensor([output]),None):break
        streamer.end()
        return torch.cat([ids,torch.tensor([output])],dim=1)


def test_backend_callback_is_in_generate_not_post_completion_and_cancel_releases_snapshot():
    model=TokenModel();bank=Bank([]);backend=StreamingBackend(model,ByteTokenizer(),bank)
    request=LabeledRequest.parse(dict(prompt='Hi',strengths={},temperature=0,max_new_tokens=32),[])
    cancelled=threading.Event();seen=[];snapshot=bank.snapshot()
    def emit(text,count):
        assert bank.active==1
        seen.append(text)
        if text=='Real':cancelled.set()
    result=backend.generate(request,cancelled,snapshot=snapshot,on_text=emit)
    assert seen and result['termination']=='cancelled' and bank.active==0
    assert bank.seen[0][1] is snapshot and result['weights']=='raw'


def artifacts():
    return [Artifact('one','One · raw256','/fixture/one','a'*64,'glimmer-particle-teacher-v1',256,'fixture'),
            Artifact('two','Two · raw512','/fixture/two','b'*64,'glimmer-particle-teacher-v1',512,'fixture')]


class Backend:
    model_loads=1
    def __init__(self):self.sliders=Bank(artifacts());self.calls=0;self.release=threading.Event()
    def generate(self, request, cancelled, *, snapshot, on_text):
        self.calls+=1
        for count,text in enumerate(['R','Real','Real café 🐈'],1):
            on_text(text,count)
            if request.prompt=='error': raise RuntimeError('fixture failure')
            if cancelled.wait(.02):break
        while not self.release.is_set() and not cancelled.wait(.01):pass
        return dict(text=text,output_tokens=count,termination='cancelled' if cancelled.is_set() else 'eos')


def wait_done(runtime,jid):
    end=time.monotonic()+3
    while time.monotonic()<end:
        job=runtime.get(jid)
        if job['status'] in ('done','failed','cancelled'):return job
        time.sleep(.01)
    pytest.fail('Fixture generation did not finish')


def test_frozen_multi_slider_provenance_repeated_submit_off_error_and_cancel():
    backend=Backend();runtime=StreamingQueue(artifacts(),backend)
    body=dict(prompt='Hello',strengths={'one':1.5,'two':-.75})
    job=runtime.submit_stream(body,'a'*32);body['strengths']['one']=7
    duplicate=runtime.submit_stream(dict(prompt='Hello',strengths={'one':1.5,'two':-.75}),'a'*32)
    assert duplicate['id']==job['id']
    with pytest.raises(ValueError):runtime.submit_stream(body,'a'*32)
    # A future catalog and mutable client controls cannot relabel this job.
    backend.sliders.artifacts=(replace(artifacts()[0],title='Replacement',step=1024),artifacts()[1])
    runtime.artifacts=backend.sliders.artifacts
    runtime.cancel(job['id']);finished=wait_done(runtime,job['id'])
    assert finished['effective_mix']=='ON'
    assert [(a['name'],a['strength'],a['step']) for a in finished['effective_sliders']]==[
        ('One · raw256',1.5,256),('Two · raw512',-.75,512)]
    assert backend.calls<=1
    off=runtime.submit_stream(dict(prompt='error',strengths={}),'b'*32)
    failed=wait_done(runtime,off['id'])
    assert failed['status']=='failed' and failed['partial_text']=='R'
    assert failed['effective_sliders']==[] and failed['effective_mix']=='OFF / base model'


def serve(runtime):
    server=ThreadingHTTPServer(('127.0.0.1',0),streaming_handler_for(runtime,'127.0.0.1',0))
    port=server.server_port;server.RequestHandlerClass=streaming_handler_for(runtime,'127.0.0.1',port)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,f'http://127.0.0.1:{port}'


def connect(url, rid):
    body=json.dumps(dict(prompt='Hello',strengths={'one':1.5})).encode()
    return urllib.request.urlopen(urllib.request.Request(url+'/api/stream',data=body,
        headers={'Content-Type':'application/json','X-Request-ID':rid}),timeout=5)


def test_http_stream_arrives_before_completion_and_has_terminal_metadata():
    backend=Backend();runtime=StreamingQueue(artifacts(),backend);server,url=serve(runtime)
    try:
        with connect(url,'c'*32) as response:
            start=json.loads(response.readline());assert start['type']=='start'
            first=json.loads(response.readline());assert first['type']=='text' and first['text']=='R'
            assert runtime.get(start['job']['id'])['status']=='running'
            backend.release.set();events=[json.loads(line) for line in response]
            assert events[-1]['type']=='complete' and events[-1]['job']['result']['termination']=='eos'
            assert any(e.get('text')=='Real café 🐈' for e in events)
    finally:backend.release.set();server.shutdown();server.server_close()


def test_http_disconnect_cancels_generation_without_killing_queue():
    backend=Backend();runtime=StreamingQueue(artifacts(),backend);server,url=serve(runtime)
    try:
        response=connect(url,'d'*32);job=json.loads(response.readline())['job'];response.readline();response.close()
        stopped=wait_done(runtime,job['id']);assert stopped['status']=='cancelled'
        backend.release.set();next_job=runtime.submit_stream(dict(prompt='next',strengths={}),'e'*32)
        assert wait_done(runtime,next_job['id'])['status']=='done'
    finally:backend.release.set();server.shutdown();server.server_close()
