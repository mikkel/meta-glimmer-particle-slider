import copy
from dataclasses import replace
import threading

import pytest
import torch

from glimmer_slider.host import Slider
from glimmer_slider.particle_catalog_runtime import replace_particles
from glimmer_slider.sampler import ResidentSliders, describe
from glimmer_slider.storage import OutputStore, save_adapter
from test_pipeline import TinyCausal, IDENTITY


@pytest.fixture
def raw_banks(tmp_path):
    torch.manual_seed(763)
    pristine = TinyCausal().eval()
    artifacts = []
    for sid, ordinary in [('raw1', False), ('raw2', False), ('ordinary', True)]:
        model = copy.deepcopy(pristine)
        slider = Slider(model, IDENTITY, ordinary=ordinary)
        with torch.no_grad():
            for branch in slider.branches.values():
                branch.lora_up.weight.normal_(std=.1)
        save_adapter(OutputStore(tmp_path, reserve_bytes=0), sid+'.safetensors', slider,
                     training=dict(step=512, weights='raw'))
        slider.detach()
        artifacts.append(describe(tmp_path/(sid+'.safetensors'), id=sid, title=sid, qualification='CPU fixture'))
    eager, scripted = copy.deepcopy(pristine), copy.deepcopy(pristine)
    banks = [ResidentSliders(eager, IDENTITY, artifacts),
             ResidentSliders(scripted, IDENTITY, artifacts, particle_execution='scripted')]
    yield pristine, eager, scripted, *banks
    for bank in banks:
        bank.detach()


@pytest.mark.parametrize('mix', [{}, {'raw1': 0, 'raw2': -0.}, {'raw1': 1.5},
    {'raw1': -.75}, {'raw1': 1.5, 'raw2': -.75}, {'raw1': .25, 'raw2': 2, 'ordinary': -.5}])
@pytest.mark.parametrize('length', [1, 3, 64])
def test_raw_program_is_bitwise_equal_for_prefill_decode_and_signed_mixes(raw_banks, mix, length):
    pristine, eager, scripted, eb, sb = raw_banks
    ids = torch.arange(length).remainder(32).unsqueeze(0)
    pointers = [p.data_ptr() for p in scripted.parameters()]
    weights = {k:v.clone() for k,v in scripted.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    with torch.inference_mode(), eb.using(mix), sb.using(mix):
        a, b = eager(ids).logits, scripted(ids).logits
        assert torch.equal(a, b)
        if not any(mix.values()):
            assert torch.equal(b, pristine(ids).logits)
    assert pointers == [p.data_ptr() for p in scripted.parameters()]
    assert all(torch.equal(v,scripted.state_dict()[k]) for k,v in weights.items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sb.snapshot().artifacts == eb.snapshot().artifacts


def test_raw_program_generation_tokens_and_exception_restore(raw_banks):
    pristine, eager, scripted, eb, sb = raw_banks
    ids = torch.tensor([[1,3,6]])
    with torch.inference_mode():
        for mix in ({'raw1': 1.}, {'raw1': 1.5, 'raw2': -.75}):
            with eb.using(mix), sb.using(mix):
                assert torch.equal(eager.generate(ids, max_new_tokens=16), scripted.generate(ids, max_new_tokens=16))
        with pytest.raises(RuntimeError, match='cancel'):
            with sb.using({'raw1':1,'raw2':-1}):
                scripted(ids)
                raise RuntimeError('cancel fixture')
        assert sb._readers == 0 and sb._active.get() is None
        assert torch.equal(scripted(ids).logits, pristine(ids).logits)


def test_catalog_swap_does_not_reuse_program_from_old_hash(raw_banks):
    _, _, model, _, bank = raw_banks
    ids = torch.tensor([[1,3,6]])
    old = bank.snapshot()
    with torch.inference_mode(), bank.using({'raw1':1}, old):
        before = model(ids).logits.clone()
    program = old.adapters['raw1']._raw_programs[0]
    # Replace raw1 by a distinct, already hash-verified raw2 checkpoint.
    candidate = replace(old.artifacts[1], id='raw1', title='Replacement')
    replace_particles(bank, [candidate])
    with torch.inference_mode(), bank.using({'raw1':1}):
        assert not torch.equal(before, model(ids).logits)
    assert bank.adapters['raw1']._raw_programs[0] is not program
    with torch.inference_mode(), bank.using({'raw1':1}, old):
        assert torch.equal(before, model(ids).logits)
    assert bank.snapshot().artifacts == (candidate,)


def test_streaming_cancellation_keeps_multi_slider_snapshot_and_raw_provenance(raw_banks):
    from glimmer_slider.sampler_stream import StreamingBackend
    from glimmer_slider.restored_sampler import LabeledRequest
    from test_sampler_stream import ByteTokenizer, TokenModel
    pristine, _, model, _, bank = raw_banks
    backend = StreamingBackend(TokenModel(), ByteTokenizer(), bank)
    mix = {'raw1':1.5,'raw2':-.75}
    req = LabeledRequest.parse(dict(prompt='Hi',strengths=mix,temperature=0,max_new_tokens=32), list(mix))
    snapshot = bank.snapshot()
    cancelled = threading.Event();seen=[]
    def emit(text, count):
        assert bank._readers == 1
        # Exercise the actual raw programs while the backend owns this plan.
        with torch.inference_mode():
            assert torch.isfinite(model(torch.tensor([[1,3,6]])).logits).all()
        seen.append((text,count))
        if text == 'Real': cancelled.set()
    rng = torch.random.get_rng_state().clone()
    result = backend.generate(req, cancelled, snapshot=snapshot, on_text=emit)
    assert seen and result['termination'] == 'cancelled'
    assert result['artifact_format'] == 'glimmer-particle-teacher-v1' and result['weights'] == 'raw'
    assert result['input_token_ids'] == [1000]
    assert bank._readers == 0 and bank._active.get() is None
    assert bank.snapshot().artifacts == snapshot.artifacts
    assert torch.equal(rng,torch.random.get_rng_state())
    with torch.inference_mode():
        assert torch.equal(model(torch.tensor([[1,3,6]])).logits, pristine(torch.tensor([[1,3,6]])).logits)
