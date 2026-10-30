import json,threading,time
from pathlib import Path
import torch
from glimmer_slider.sampler_stream import AnswerStreamer, StreamingBackend, StreamingQueue
from glimmer_slider.studio import LabeledRequest
from test_sampler_stream import ByteTokenizer, TokenModel, Bank, serve, connect, artifacts, wait_done


def test_split_channel_headers_unicode_and_final_answer_are_separate():
    final=[];reasoning=[]
    stream=AnswerStreamer(ByteTokenizer(),[1001],lambda text,n:final.append(text),lambda text,n:reasoning.append(text))
    stream.put(torch.tensor([[1001]]))
    raw=' to=self<|message|>Count café carefully.<|eom|><|start|>assistant to=user<|message|>Four 🐈.'
    for value in raw.encode():stream.put(torch.tensor([value]))
    stream.put(torch.tensor([1002]));stream.end()
    assert reasoning[-1]=='Count café carefully.' and final[-1]=='Four 🐈.'
    assert all('Count café' not in value for value in final)
    assert all('Four' not in value for value in reasoning)
    assert all('\ufffd' not in value for value in final+reasoning)


class ThinkingTokenizer(ByteTokenizer):
    def apply_chat_template(self,*args,**kwargs):return {'input_ids':[[1001]]}


class ThinkingModel(TokenModel):
    def generate(self,**kwargs):
        ids=kwargs['input_ids'];stream=kwargs['streamer'];stream.put(ids)
        output=[]
        raw=' to=self<|message|>Two plus two is four.<|eom|><|start|>assistant to=user<|message|>4.'
        for value in list(raw.encode())+[1002]:
            output.append(value);stream.put(torch.tensor([value]))
            if kwargs['stopping_criteria'](torch.tensor([output]),None):break
        stream.end();return torch.cat([ids,torch.tensor([output])],dim=1)


def test_native_backend_retains_reasoning_and_streams_real_channel_callbacks():
    backend=StreamingBackend(ThinkingModel(),ThinkingTokenizer(),Bank([]));seen=[]
    req=LabeledRequest.parse(dict(prompt='2+2',strengths={},mode='reasoning',max_new_tokens=256),[])
    result=backend.generate(req,threading.Event(),on_text=lambda text,n,channel='text':seen.append((channel,text)))
    assert result['reasoning_text']=='Two plus two is four.' and result['text']=='4.'
    assert ('reasoning','Two plus two is four.') in seen
    assert result['termination']=='eos'


class ReasoningBackend:
    model_loads=1
    def __init__(self):self.sliders=Bank(artifacts());self.release=threading.Event()
    def generate(self,request,cancelled,*,snapshot,on_text):
        on_text('Model reasoning café',1,channel='reasoning')
        while not self.release.is_set() and not cancelled.wait(.01):pass
        on_text('Final answer',2)
        return dict(text='Final answer',reasoning_text='Model reasoning café',termination='cancelled' if cancelled.is_set() else 'eos')


def test_http_reasoning_arrives_live_and_provenance_is_frozen():
    backend=ReasoningBackend();runtime=StreamingQueue(artifacts(),backend);server,url=serve(runtime)
    try:
        with connect(url,'f'*32) as response:
            start=json.loads(response.readline());reason=json.loads(response.readline())
            assert reason['type']=='reasoning' and reason['text']=='Model reasoning café'
            assert runtime.get(start['job']['id'])['status']=='running'
            backend.release.set();events=[json.loads(line) for line in response]
            assert any(e.get('type')=='text' and e['text']=='Final answer' for e in events)
            job=events[-1]['job'];assert job['effective_sliders']==start['job']['effective_sliders']
            assert job['partial_reasoning']=='Model reasoning café'
    finally:backend.release.set();server.shutdown();server.server_close()


def test_reasoning_ui_has_one_existing_mode_control_and_retains_saved_runs():
    import glimmer_slider.studio as service
    html=Path(service.__file__).with_name('studio.html').read_text()
    assert html.count('id="mode"')==1 and html.count('id="reasoning"')==1
    assert 'id="reasoning-panel"' in html and 'renderReasoning(run)' in html
    assert "event.type==='reasoning'" in html and 'partial_reasoning' in html
    assert 'Both share the output-token budget' in html
