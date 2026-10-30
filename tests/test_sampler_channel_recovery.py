import threading
import torch
from glimmer_slider.channel_recovery import recover_explicit_channels
from glimmer_slider.sampler_stream import AnswerStreamer,StreamingBackend,StreamingQueue
from glimmer_slider.studio import LabeledRequest
from test_sampler_stream import ByteTokenizer,TokenModel,Bank

PREFIX='<|start|>assistant'
RAW=" to=self...mmm... 12 plus 9, minus 4, plus 3 is 20...<|eom|><|start|>assistant to=user...20 marbles.<|end_of_text|>"


def test_sleepy_27_missing_message_delimiters_are_recovered_and_flagged():
    value=recover_explicit_channels(RAW,PREFIX)
    assert value['status']=='recovered_explicit_channels'
    assert value['reasoning']=='...mmm... 12 plus 9, minus 4, plus 3 is 20...'
    assert value['content']=='...20 marbles.'
    assert len(value['warnings'])==2


def test_unclassified_text_and_special_tokens_never_become_reasoning():
    value=recover_explicit_channels('mmm... sleepy text with no channel',PREFIX)
    assert value['status']=='unclassified' and value['reasoning']==value['content']==''
    value=recover_explicit_channels('<|eom|>',PREFIX)
    assert value['status']=='special_tokens_only' and value['reasoning']==value['content']==''
    value=recover_explicit_channels(' to=selfish...hello',PREFIX)
    assert value['reasoning']==''


def test_incremental_split_missing_headers_and_unicode_do_not_leak_into_wrong_channel():
    final=[];reason=[];diagnostic=[]
    stream=AnswerStreamer(ByteTokenizer(),[1001],lambda text,n:final.append(text),lambda text,n:reason.append(text),lambda raw,n,framing:diagnostic.append((raw,n,framing)))
    stream.put(torch.tensor([[1001]]))
    raw=' to=self...café before answer.<|eom|><|start|>assistant to=user...Final 🐈.'
    for value in raw.encode():stream.put(torch.tensor([value]))
    stream.put(torch.tensor([1002]));stream.end()
    assert reason[-1]=='...café before answer.' and final[-1]=='...Final 🐈.'
    assert all('<|start' not in text and 'Final' not in text for text in reason)
    assert all('café' not in text for text in final)
    assert all('\ufffd' not in text for text in final+reason)
    assert diagnostic[-1][2]['status']=='recovered_explicit_channels'


class Tokenizer(ByteTokenizer):
    def apply_chat_template(self,*args,**kwargs):return {'input_ids':[[1001]]}


class MalformedSleepyModel(TokenModel):
    def generate(self,**kwargs):
        stream=kwargs['streamer'];ids=kwargs['input_ids'];stream.put(ids);output=[]
        for value in list(RAW.encode())+[1002]:
            output.append(value);stream.put(torch.tensor([value]))
        stream.end();return torch.cat([ids,torch.tensor([output])],dim=1)


def test_high_effort_sleepy_path_keeps_strict_parse_raw_and_live_recovery_distinct():
    model=MalformedSleepyModel();bank=Bank([]);backend=StreamingBackend(model,Tokenizer(),bank);events=[]
    req=LabeledRequest.parse(dict(prompt='Explain the marble puzzle',system='You are helpful',system_condition='custom',mode='reasoning',reasoning='high',strengths={'sleepy':2.7},seed=7,max_new_tokens=256,temperature=0),['sleepy'])
    result=backend.generate(req,threading.Event(),on_text=lambda text,n,channel='text':events.append((channel,text,n)))
    assert bank.seen[0][0]=={'sleepy':2.7}
    assert result['text']=='...20 marbles.' and result['reasoning_text'].startswith('...mmm')
    assert not (result['parsed'] or {}).get('reasoning_content')
    assert result['framing']['status']=='recovered_explicit_channels'
    assert result['framing']['official_reasoning_characters']==0
    assert any(e[0]=='raw' for e in events) and any(e[0]=='protocol' for e in events)
    reason=[e for e in events if e[0]=='reasoning'];assert len(reason)>2 and reason[0][2]<reason[-1][2]


def test_normal_valid_protocol_is_unchanged():
    value=recover_explicit_channels(' to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>Answer.<|eot|>',PREFIX)
    assert value==dict(reasoning='Think.',content='Answer.',status='standard',warnings=[])


def test_incomplete_valid_separator_waits_without_claiming_bad_framing():
    for raw in [' to=self<',' to=self<|m',' to=self<|message|>']:
        assert recover_explicit_channels(raw,PREFIX)['status'] in ('pending_channel_header','standard')


def test_recorded_real_sleepy_output_recovers_both_regions_without_changing_raw():
    import json
    from pathlib import Path
    data=json.loads((Path(__file__).parent/'fixtures/sleepy_27_missing_message.json').read_text())
    raw=data['raw'];value=recover_explicit_channels(raw,PREFIX)
    expected_reason,expected_final=raw[len(' to=self'):].split('<|eom|><|start|>assistant to=user',1)
    expected_final=expected_final.removesuffix('<|end_of_text|>')
    assert value['reasoning']==expected_reason and value['content']==expected_final
    assert value['status']=='recovered_explicit_channels' and data['output_tokens']==213
    assert not (data['strict_parsed'] or {}).get('content')
