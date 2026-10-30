from pathlib import Path
import pytest
import torch
from glimmer_slider import studio as service
from glimmer_slider import restored_sampler
from glimmer_slider import sampler_server as studio


def request(n=4096):
    return service.LabeledRequest.parse(dict(prompt='Hello', strengths={}, max_new_tokens=n), [])


def test_output_range_default_and_no_silent_clamping():
    assert request(1).max_new_tokens == 1
    assert request(1025).max_new_tokens == 1025
    assert request(4096).max_new_tokens == 4096
    assert service.LabeledRequest.parse(dict(prompt='Hello',strengths={}),[]).max_new_tokens == 256
    for value in (0,4097,True,1.5):
        with pytest.raises(ValueError,match='max_new_tokens'):
            request(value)


class Tokenizer:
    def __init__(self,n): self.n=n
    def apply_chat_template(self,*args,**kwargs):
        return {'input_ids':torch.zeros((1,self.n),dtype=torch.long)}


@pytest.mark.parametrize('module',[service,restored_sampler])
def test_prompt_and_output_share_context_budget(module):
    assert module.prompt_tokens(Tokenizer(4096),request()).shape==(1,4096)
    with pytest.raises(ValueError,match='context budget'):
        module.prompt_tokens(Tokenizer(4097),request())
    assert module.prompt_tokens(Tokenizer(4096),request(1)).shape==(1,4096)
    with pytest.raises(ValueError,match='context budget'):
        module.prompt_tokens(Tokenizer(4097),request(1))


def test_advertised_limits_and_ui_match_without_changing_default():
    value=studio.SamplerQueue([],None).state()['limits']
    assert value['max_new_tokens']==4096 and value['context']==8192
    assert value['generation_seconds']==240 and value['max_prompt_tokens']==4096
    ui=Path(service.__file__).with_name('studio.html').read_text()
    assert 'id="tokens" type="number" min="1" max="4096" value="256"' in ui
    assert 'prompt + output budget: 8192 tokens' in ui
