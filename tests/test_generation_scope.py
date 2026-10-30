import copy
from dataclasses import replace
import pytest
import torch
from glimmer_slider.generation_scope import active_channel,ScopeGate
from glimmer_slider.studio import LabeledRequest
from test_sampling_speed import raw_banks

S='<|start|>assistant'
@pytest.mark.parametrize('prefix,raw,want',[(S,'','unknown'),(S,' to=se','unknown'),(S,' to=self','thinking'),(S,' to=self<|mess','thinking'),(S,' to=self...hmm','thinking'),(S+' to=user<|message|>','','response'),(S,' to=self<|message|>a<|eom|>','unknown'),(S,' to=self<|message|>a<|eom|>'+S+' to=us','unknown'),(S,' to=self<|message|>a<|eom|>'+S+' to=user...answer','response'),('user to=self','','unknown')])
def test_explicit_channel_boundaries(prefix,raw,want):assert active_channel(prefix,raw)==want

def test_validation_default_and_invalid():
 req=LabeledRequest.parse(dict(prompt='Hi',strengths={'a':1}),['a']);assert req.scopes==()
 assert dict(LabeledRequest.parse(dict(prompt='Hi',scopes={'a':'thinking'}),['a']).scopes)=={'a':'thinking'}
 for scopes in [{'bad':'both'},{'a':'wrong'},[]]:
  with pytest.raises(ValueError):LabeledRequest.parse(dict(prompt='Hi',scopes=scopes),['a'])

def test_real_projection_gating_mixed_scopes_and_restoration(raw_banks):
 _,_,model,_,bank=raw_banks;ids=torch.tensor([[1,3,6]]);snap=bank.snapshot();mix={'raw1':1.5,'raw2':-.75}
 def logits(values):
  with torch.inference_mode(),bank.using(values,snap):return model(ids).logits.clone()
 expected={c:logits(v) for c,v in [('thinking',{'raw1':1.5}),('response',{'raw2':-.75}),('unknown',{})]}
 with torch.inference_mode(),bank.using(mix,snap):
  gate=ScopeGate(bank,mix,{'raw1':'thinking','raw2':'response'},snap,'unknown')
  with gate.using():
   for channel in ['unknown','thinking','response','unknown']:
    gate.select(channel);assert torch.equal(model(ids).logits,expected[channel])
 assert bank._active.get() is None and bank._readers==0
 # Both remains active for unknown and uses the exact original plan.
 with torch.inference_mode(),bank.using(mix,snap):
  before=model(ids).logits.clone();plan=bank._active.get();gate=ScopeGate(bank,mix,{},snap,'unknown')
  with gate.using():
   gate.select('response');assert bank._active.get() is plan;assert torch.equal(model(ids).logits,before)
 # Alpha changes reuse immutable compiled raw programs.
 with torch.inference_mode(),bank.using({'raw1':1},snap):pass
 program=snap.adapters['raw1']._raw_programs[0]
 with torch.inference_mode(),bank.using({'raw1':2},snap):pass
 assert snap.adapters['raw1']._raw_programs[0] is program

def test_backend_gates_actual_forwards_across_split_and_malformed_headers(raw_banks):
 import threading
 from glimmer_slider.sampler_stream import StreamingBackend
 from test_sampler_reasoning_channels import ThinkingTokenizer,ThinkingModel
 _,_,model,_,bank=raw_banks;ids=torch.tensor([[1,3,6]]);snap=bank.snapshot();mix={'raw1':1.5,'raw2':-.75}
 expected={}
 for channel,values in [('thinking',{'raw1':1.5}),('response',{'raw2':-.75}),('unknown',{})]:
  with torch.inference_mode(),bank.using(values,snap):expected[channel]=model(ids).logits.clone()
 class CheckingModel(ThinkingModel):
  def generate(self,**kwargs):
   prompt=kwargs['input_ids'];stream=kwargs['streamer'];stream.put(prompt);raw='';output=[]
   # Missing separators exercise the same explicit recipient recovery as native Sleepy.
   body=' to=self...hmm<|eom|><|start|>assistant to=user...4.'
   for value in list(body.encode())+[1002]:
    channel=active_channel('<|start|>assistant',raw)
    assert torch.equal(model(ids).logits,expected[channel])
    output.append(value);stream.put(torch.tensor([value]));raw+=chr(value) if value<1000 else '<|eot|>'
   stream.end();return torch.cat([prompt,torch.tensor([output])],dim=1)
 backend=StreamingBackend(CheckingModel(),ThinkingTokenizer(),bank)
 req=LabeledRequest.parse(dict(prompt='2+2',strengths=mix,scopes={'raw1':'thinking','raw2':'response'},mode='reasoning',max_new_tokens=256),list(mix))
 result=backend.generate(req,threading.Event(),snapshot=snap)
 assert 'thinking' in result['scope_gating']['transitions'] and 'response' in result['scope_gating']['transitions']
 assert bank._active.get() is None and bank._readers==0
