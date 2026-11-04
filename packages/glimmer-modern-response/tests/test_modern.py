from copy import deepcopy
from dataclasses import replace
import importlib.metadata
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from glimmer_modern.fixture import make_model
from glimmer_modern.lifecycle import BACKEND, INFERENCE, cpu_config
from glimmer_modern.native import Native, ReadoutCritic, targets
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation
from particle_sliders.pg09 import _hash
from particle_sliders.response import open_response


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def config(tmp_path):
    pairs = dict(train=[dict(neutral=[1,3], positive=[2,4], history=[6,7]),
                        dict(neutral=[1,5], positive=[2,8], history=[9])],
                 evaluation=[dict(neutral=[1,10], positive=[2,11], history=[12]),
                             dict(neutral=[1,13], positive=[2,14], history=[15,16])])
    path = tmp_path / 'pairs.json'
    path.write_text(json.dumps(pairs))
    return cpu_config(path)


def test_installed_identity_and_dependency_split():
    import glimmer_modern
    import particle_sliders.pg09 as game
    assert 'site-packages' in str(Path(glimmer_modern.__file__))
    assert 'site-packages' in inspect.getsourcefile(game.ConditionalPG09)
    assert importlib.metadata.version('particlegan') == '0.9.4'
    assert importlib.metadata.version('hndl') == '0.10.0'
    assert 'glimmer_slider.upstream' not in sys.modules
    assert 'glimmer_slider.training' not in sys.modules
    from glimmer_modern.provenance import verify_sources
    _, identity = verify_sources()
    assert identity['core_commit'] == '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'


@pytest.mark.parametrize('output_width', [4, 12])
@pytest.mark.parametrize('shape', [(2,12), (2,3,12)])
def test_routed_outputs_all_vjps_and_strength(output_width, shape):
    from particle_sliders.routed_projection import TinyParticleProjection
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    torch.manual_seed(42)
    host = nn.Linear(12, output_width, bias=False).requires_grad_(False)
    twin = deepcopy(host)
    bank = nn.Parameter(torch.randn(128,4))
    other_bank = nn.Parameter(bank.detach().clone())
    old = TinyParticleProjection('ref', host, bank, 8, 8)
    new = HNDLParticleProjection('hndl', twin, other_bank, 8, 8)
    pairs = [(new.graph['down'], old.lora_down), (new.graph['up'], old.lora_up)]
    nn.init.normal_(old.lora_up.weight, std=.2)
    for family, sequence in [('router', old.bridge.router), ('bridge', old.bridge.net)]:
        pairs.extend((new.graph[family + str(i)], sequence[i]) for i in (0,2,4,6))
    with torch.no_grad():
        for destination, source in pairs:
            destination.weight.copy_(source.weight)
            if source.bias is not None:
                destination.bias.copy_(source.bias)
    for strength in (0., .5, 1.):
        old.multiplier = new.multiplier = strength
        x = torch.randn(shape, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        expected, actual = host(x), twin(y)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        if strength == 0:
            assert torch.equal(actual, new.org_forward(y))
            continue
        old_params, new_params = [x, bank], [y, other_bank]
        for destination, source in pairs:
            old_params.extend(source.parameters())
            new_params.extend(destination.parameters())
        vjp = torch.randn_like(expected)
        for a, b in zip(torch.autograd.grad(actual, new_params, vjp),
                        torch.autograd.grad(expected, old_params, vjp)):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def test_critic_outputs_vjp_and_second_derivatives():
    from particle_sliders.pg09_audio import AudioResponseCritic
    torch.manual_seed(11)
    old, new = AudioResponseCritic(12,16), ReadoutCritic(12,16)
    with torch.no_grad():
        for destination, source in [(new.graph['hidden'], old.net[0]), (new.graph['logits'], old.net[2])]:
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)
    x = torch.randn(2,12,requires_grad=True)
    y = x.detach().clone().requires_grad_()
    condition = torch.randn(2,16)
    a,b = old(x, condition),new(y,condition)
    torch.testing.assert_close(a,b)
    ga = torch.autograd.grad(a.sum(),x,create_graph=True)[0]
    gb = torch.autograd.grad(b.sum(),y,create_graph=True)[0]
    torch.testing.assert_close(ga,gb)
    da = torch.autograd.grad(ga.square().sum(),tuple(old.parameters()),allow_unused=True)
    # Final bias has no input derivative and is mathematically unused.
    db = torch.autograd.grad(gb.square().sum(),tuple(new.parameters()),allow_unused=True)
    for a,b in zip(da,db):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a,b)


def test_shared_updates_exact_resume_ema_raw_and_native_only(config,tmp_path):
    before_rng = torch.get_rng_state().clone()
    with open_response(config) as full:
        native = full.game.backend
        assert len(native.slider.branches) == 4
        assert [b.output_width for b in native.slider.branches.values()] == [12,4,4,12]
        assert sum(p is native.slider.particles for p in full.game.network.parameters()) == 1
        initial = deepcopy(full.game.network.state_dict())
        base = deepcopy(native.model.state_dict())
        metrics = []
        full.train(on_step=lambda m: metrics.append(m['step']))
        assert metrics == [1,2,3,4]
        assert all(torch.equal(v,native.model.state_dict()[k]) for k,v in base.items())
        assert any(not torch.equal(v,full.game.network.state_dict()[k]) for k,v in initial.items())
        assert not torch.equal(initial['weights.0'],native.slider.particles)
        expected = full.checkpoint()['sha256']
        ids = [1,10,12]
        raw = full.evaluate(index=0,split='evaluation',strength=.5)
        full.export_inference(tmp_path/'raw.pt')
        with torch.no_grad():
            saved=deepcopy(full.game.network.state_dict())
            full.game.network.load_state_dict(full.ema.values)
            ema = native.response(ids=ids,strength=.5)
            full.game.network.load_state_dict(saved)
        full.export_inference(tmp_path/'ema.pt',selection='ema')
        full.export(tmp_path/'v1.pt')
    with open_response(config) as split:
        split.train(stop_after=2)
        split.checkpoint(tmp_path/'training.pt')
    with open_response(config,resume=tmp_path/'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
    # Data-independent native factory cannot call any training constructor.
    Path(config.options['data_file']).unlink()
    import glimmer_modern.lifecycle as product
    original = product.ConditionalPG09
    product.ConditionalPG09 = lambda *a,**kw: pytest.fail('inference constructed a game')
    try:
        for name,expected_value in [('raw',raw),('ema',ema)]:
            with open_inference(tmp_path/(name+'.pt'),operation='response',native_request={'ids':ids}) as inference:
                assert torch.equal(inference.response(ids=ids,strength=.5),expected_value)
                assert not hasattr(inference,'game')
                with pytest.raises(UnsupportedOperation):
                    inference.train()
    finally:
        product.ConditionalPG09 = original
    assert torch.equal(before_rng,torch.get_rng_state())


def test_native_only_fresh_process(config,tmp_path):
    artifact=tmp_path/'native.pt'
    with open_response(config) as run:
        run.step()
        run.export_inference(artifact)
    Path(config.options['data_file']).unlink()
    code = '''import sys,torch
from particle_sliders.inference import open_inference
with open_inference(sys.argv[1], operation="response", native_request={"ids":[1,3]}) as run:
 assert run.response(ids=[1,3]).shape == (1,12)
 assert not hasattr(run,"game")
assert not torch.cuda.is_initialized()
print("native-only fresh process: PASS")
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(artifact)],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'PASS' in result.stdout


def test_borrowed_model_cleanup_and_zero(config):
    model=make_model(41).train()
    model.embedding.requires_grad_(False)
    model.model.language_model.layers[0].eval()
    modes=[m.training for m in model.modules()]
    flags=[p.requires_grad for p in model.parameters()]
    forwards=[m.forward for m in targets(model).values()]
    baseline=model(torch.tensor([[1,3]])).hidden_states[-1].detach().clone()
    with BACKEND.open(config,model=model) as run:
        native=run.game.backend
        with native.slider.scaled(0.):
            assert torch.equal(model(torch.tensor([[1,3]])).hidden_states[-1],baseline)
        run.step()
        grads=[None if p.grad is None else p.grad.clone() for p in native.network.parameters()]
        run.evaluate(strength=.5)
        assert all((a is None and p.grad is None) or torch.equal(a,p.grad)
                   for a,p in zip(grads,native.network.parameters()))
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [m.forward for m in targets(model).values()]==forwards


@pytest.mark.parametrize('options', [dict(mode='native'),dict(width=1024),
    dict(penalty_input_domain='continuous'),dict(pg_source_profile='develop'),dict(base_precision='bfloat16')])
def test_native_and_mismatch_gate_before_prepare(config,monkeypatch,options):
    values=dict(config.options,**options)
    changed=SessionConfig(config.backend,config.cohort,**values)
    guarded=replace(BACKEND,prepare=lambda *a,**k: pytest.fail('native mutation reached'))
    with pytest.raises((ValueError,UnsupportedOperation)):
        guarded.open(changed)


def test_source_tamper_before_mutation(config,monkeypatch):
    import glimmer_modern.provenance as provenance
    model=make_model(41)
    flags=[p.requires_grad for p in model.parameters()]
    forwards=[m.forward for m in targets(model).values()]
    original=provenance.hashlib.sha256
    monkeypatch.setattr(provenance.hashlib,'sha256',lambda *a,**kw: original(b'wrong source'))
    with pytest.raises(ValueError,match='shared-core source'):
        BACKEND.open(config,model=model)
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [m.forward for m in targets(model).values()]==forwards


def test_invalid_restore_is_transactional_and_artifact_split(config):
    with open_response(config) as run:
        run.step()
        packet=run.checkpoint()
        before=packet['sha256']
        bad=deepcopy(packet)
        bad['payload']['game']['network']['weights.0'][0,0]=float('nan')
        with pytest.raises((CheckpointError,ValueError)):
            run.restore(bad)
        assert run.checkpoint()['sha256']==before
        with pytest.raises(CheckpointError):
            run.restore(dict(format='glimmer-pg-continuation-v1'))
        other=deepcopy(packet)
        other['manifest']['config']['cohort']='historical-raw1024'
        other['sha256']=_hash({k:v for k,v in other.items() if k!='sha256'})
        with pytest.raises(CheckpointError):
            run.restore(other)
        assert run.checkpoint()['sha256']==before


@pytest.mark.parametrize('corruption',['late-sampler','ema-shape','bank-alias'])
def test_rehashed_invalid_state_cannot_partially_mutate_adapter(config,corruption):
    with open_response(config) as run:
        run.step()
        packet=run.checkpoint()
        before=packet['sha256']
        bad=deepcopy(packet)
        bad['payload']['game']['network']['weights.1'].add_(123.)
        if corruption=='late-sampler':
            bad['payload']['game']['row_cursors']['d']=-1
        elif corruption=='ema-shape':
            bad['payload']['ema']['values']['weights.0']=torch.zeros(1)
        else:
            # Break the intentionally shared storage instead of mutating both aliases.
            bad['payload']['game']['prior']['z']=bad['payload']['game']['prior']['z']+1.
        bad['sha256']=_hash({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(CheckpointError):
            run.restore(bad)
        assert run.checkpoint()['sha256']==before


def test_failure_during_preparation_detaches_and_restores(config,monkeypatch):
    import glimmer_modern.lifecycle as product
    model=make_model(41).train()
    modes=[m.training for m in model.modules()]
    flags=[p.requires_grad for p in model.parameters()]
    forwards=[m.forward for m in targets(model).values()]
    def fail(*a,**kw):
        raise RuntimeError('fixture critic setup failure')
    monkeypatch.setattr(product,'ReadoutCritic',fail)
    with pytest.raises(RuntimeError,match='fixture critic'):
        BACKEND.open(config,model=model)
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [m.forward for m in targets(model).values()]==forwards


def test_failure_during_update_closes_all_hooks(config,monkeypatch):
    model=make_model(41).train()
    modes=[m.training for m in model.modules()]
    flags=[p.requires_grad for p in model.parameters()]
    forwards=[m.forward for m in targets(model).values()]
    run=BACKEND.open(config,model=model)
    def fail(*a,**kw):
        raise RuntimeError('fixture update failure')
    monkeypatch.setattr(run.game.backend,'generated',fail)
    with pytest.raises(RuntimeError,match='fixture update'):
        run.step()
    assert run.closed
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [m.forward for m in targets(model).values()]==forwards


def test_cleanup_attempts_slider_when_checkpoint_cleanup_fails(config,monkeypatch):
    model=make_model(41).train()
    modes=[m.training for m in model.modules()]
    flags=[p.requires_grad for p in model.parameters()]
    forwards=[m.forward for m in targets(model).values()]
    run=BACKEND.open(config,model=model)
    def fail():
        raise RuntimeError('checkpoint cleanup fixture failure')
    monkeypatch.setattr(run.game.backend.backend,'close_checkpointing',fail)
    with pytest.raises(RuntimeError,match='checkpoint cleanup fixture'):
        run.close()
    assert [m.training for m in model.modules()]==modes
    assert [p.requires_grad for p in model.parameters()]==flags
    assert [m.forward for m in targets(model).values()]==forwards


def test_every_projection_cleanup_attempted_after_ownership_conflict(config):
    model=make_model(41).train()
    run=BACKEND.open(config,model=model)
    native=run.game.backend
    modules=list(native.slider._targets.values())
    owned_forward=modules[0].forward
    modules[0].forward=lambda x: x
    try:
        with pytest.raises(RuntimeError,match='ownership changed'):
            native.slider.close()
        assert all(not hasattr(m,'_glimmer_modern') for m in modules[1:])
        assert all(p.requires_grad for p in model.parameters())
    finally:
        modules[0].forward=owned_forward
        native.slider.close()
        run.close()
    assert not hasattr(modules[0],'_glimmer_modern')


def test_graph_setup_failure_precedes_all_native_mutation(config,monkeypatch):
    import glimmer_modern.native as module
    model=make_model(41).train()
    forwards=[m.forward for m in targets(model).values()]
    flags=[p.requires_grad for p in model.parameters()]
    real=module.HNDLParticleProjection
    calls=[]
    def fail_second(*a,**kw):
        calls.append(1)
        if len(calls)==2:
            raise RuntimeError('second graph failed')
        return real(*a,**kw)
    monkeypatch.setattr(module,'HNDLParticleProjection',fail_second)
    with pytest.raises(RuntimeError,match='second graph'):
        BACKEND.open(config,model=model)
    assert [m.forward for m in targets(model).values()]==forwards
    assert [p.requires_grad for p in model.parameters()]==flags
    assert model.training


def test_failed_observation_restores_modes_buffers_gradients_and_scale(config,monkeypatch):
    model=make_model(41)
    model.register_buffer('probe',torch.ones(2),persistent=False)
    with BACKEND.open(config,model=model) as run:
        native=run.game.backend
        original=native.generated
        native.network.weights[0].grad=torch.ones_like(native.network.weights[0])
        def fail(*a,**kw):
            model.probe.add_(3)
            model.train()
            native.network.weights[0].grad.zero_()
            with native.slider.scaled(.5):
                raise RuntimeError('native observation failed')
        monkeypatch.setattr(native,'generated',fail)
        with pytest.raises(RuntimeError,match='native observation'):
            run.evaluate()
        assert torch.equal(model.probe,torch.ones(2))
        assert not model.training
        assert torch.equal(native.network.weights[0].grad,torch.ones_like(native.network.weights[0]))
        assert native.slider.strength==0.
        monkeypatch.setattr(native,'generated',original)


def test_invalid_history_and_overlap_refused_before_mutation(config):
    from glimmer_modern.lifecycle import read_rows
    path=Path(config.options['data_file'])
    value=json.loads(path.read_text())
    value['evaluation']=value['train']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='overlap'):
        read_rows(path)


def test_shared_wide_vector_guard_is_unchanged():
    from particle_sliders.pg09 import validate_pg09_domain
    with pytest.raises(ValueError,match='width >=170'):
        validate_pg09_domain((1,1024),response_domain='continuous_vector')


def test_low_level_native_width_gate_precedes_source_or_attachment(monkeypatch):
    import glimmer_modern.native as module
    model=make_model(41).train()
    forwards=[m.forward for m in targets(model).values()]
    flags=[p.requires_grad for p in model.parameters()]
    model.config.hidden_size=1024
    monkeypatch.setattr(module,'verify_sources',lambda: pytest.fail('graph construction reached'))
    with pytest.raises(ValueError,match='native/wide'):
        Native(model)
    model.config.hidden_size=12
    assert [m.forward for m in targets(model).values()]==forwards
    assert [p.requires_grad for p in model.parameters()]==flags
    assert model.training


@pytest.mark.parametrize('failure',[False,True])
def test_cleanup_restores_instance_attribute_absence(config,monkeypatch,failure):
    import glimmer_modern.lifecycle as product
    from glimmer_modern.fixture import Attention
    # Inherited checkpoint attributes must not become instance shadows when
    # the provisional shared cleanup writes back its getattr() snapshot.
    inherited_checkpoint=lambda *a,**kw: None
    monkeypatch.setattr(Attention,'gradient_checkpointing',False,raising=False)
    monkeypatch.setattr(Attention,'_gradient_checkpointing_func',inherited_checkpoint,raising=False)
    model=make_model(41).train()
    modules=list(model.modules())
    keys={id(module):set(vars(module)) for module in modules}
    forwards=[module.forward for module in targets(model).values()]
    if failure:
        def fail(*a,**kw):
            raise RuntimeError('post-attachment failure')
        monkeypatch.setattr(product,'ReadoutCritic',fail)
        with pytest.raises(RuntimeError,match='post-attachment'):
            BACKEND.open(config,model=model)
    else:
        with BACKEND.open(config,model=model) as run:
            run.step()
            run.evaluate()
            attention=model.model.language_model.layers[0].self_attn
            assert 'gradient_checkpointing' not in vars(attention)
            assert '_gradient_checkpointing_func' not in vars(attention)
        # Reopening the same borrowed model must remain valid after cleanup.
        with BACKEND.open(config,model=model):
            pass
    assert [module.forward for module in targets(model).values()]==forwards
    assert all(set(vars(module))==keys[id(module)] for module in modules)
    assert all('forward' not in vars(module) for module in modules)
    assert all('_glimmer_modern' not in vars(module) for module in modules)


@pytest.mark.parametrize('route',['public','low-level'])
@pytest.mark.parametrize('kind',[
    'forward-hook','forward-pre-hook','backward-hook','backward-pre-hook',
    'state-dict-hook','state-dict-pre-hook','load-state-dict-pre-hook','load-state-dict-post-hook',
    'parameter-hook','forward-override','call-override','serialization-override',
    'parameters-override','compiled-call','checkpoint-hook'])
def test_preexisting_local_hooks_and_instance_overrides_refused_before_mutation(config,monkeypatch,route,kind):
    model=make_model(41).train()
    projection=next(iter(targets(model).values()))
    handles=[]
    def forbidden(*a,**kw):
        pytest.fail('preexisting native hook/override executed')
    if kind=='forward-hook':
        handles.append(projection.register_forward_hook(forbidden))
    elif kind=='forward-pre-hook':
        handles.append(projection.register_forward_pre_hook(forbidden))
    elif kind=='backward-hook':
        handles.append(projection.register_full_backward_hook(forbidden))
    elif kind=='backward-pre-hook':
        handles.append(projection.register_full_backward_pre_hook(forbidden))
    elif kind=='state-dict-hook':
        handles.append(projection.register_state_dict_post_hook(forbidden))
    elif kind=='state-dict-pre-hook':
        handles.append(projection.register_state_dict_pre_hook(forbidden))
    elif kind=='load-state-dict-pre-hook':
        handles.append(projection.register_load_state_dict_pre_hook(forbidden))
    elif kind=='load-state-dict-post-hook':
        handles.append(projection.register_load_state_dict_post_hook(forbidden))
    elif kind=='parameter-hook':
        handles.append(projection.weight.register_hook(forbidden))
    elif kind=='forward-override':
        projection.forward=forbidden
    elif kind=='call-override':
        projection._call_impl=forbidden
    elif kind=='serialization-override':
        model.state_dict=forbidden
    elif kind=='parameters-override':
        model.parameters=forbidden
    elif kind=='compiled-call':
        projection._compiled_call_impl=forbidden
    else:
        projection.gradient_checkpointing=True
    from glimmer_modern.native import _native_modules
    modules=list(_native_modules(model))
    snapshots=[(module,dict(vars(module))) for _,module in modules]
    flags=[p.requires_grad for p in model.embedding.parameters()]
    rng=torch.get_rng_state().clone()
    try:
        with pytest.raises(ValueError,match='preexisting native'):
            if route=='public':
                BACKEND.open(config,model=model)
            else:
                Native(model)
        for module,saved in snapshots:
            assert vars(module).keys()==saved.keys()
            assert all(vars(module)[key] is value for key,value in saved.items())
        assert [p.requires_grad for p in model.embedding.parameters()]==flags
        assert torch.equal(rng,torch.get_rng_state())
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('name,digest',[
    ('src/glimmer_slider/upstream.py','2d53cd8f314f796da8bbe0ffa4d50a6f4fb09dd3018d0fe01d6b18c781f1aee4'),
    ('src/glimmer_slider/training.py','cc5aeecb7ddc6ede4abd707c8c17279a7f6d0bcdbc54ff4578f09f8d5c23bad0'),
    ('pyproject.toml','34e4b27439890f29a90a120c7811c8dfc1134ef178e22666be3f475a400b5bf3')])
def test_historical_pg82_repository_files_unchanged(name,digest):
    root=Path(__file__).resolve().parents[3]
    if not (root/'src/glimmer_slider/upstream.py').is_file():
        pytest.skip('historical integrity check requires the source repository alongside the tests')
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest


def _caller_cpu_seed(seed):
    torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())


def _forbid_cuda_rng(monkeypatch):
    def forbidden(*a,**kw):
        pytest.fail('CPU initialization must not initialize, query or seed CUDA RNG')
    for name in ('init','_lazy_init','manual_seed','manual_seed_all','get_rng_state',
                 'get_rng_state_all','set_rng_state','set_rng_state_all'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


def test_initial_game_bytes_independent_of_caller_rng_and_seed_positive_control(config,monkeypatch):
    from glimmer_modern import INITIALIZATION_POLICY, PROTOCOL
    _forbid_cuda_rng(monkeypatch)
    saved=torch.get_rng_state().clone()
    snapshots=[]
    try:
        for caller_seed,run_seed in [(111,7),(222,7),(333,8)]:
            _caller_cpu_seed(caller_seed)
            before=torch.get_rng_state().clone()
            selected=cpu_config(config.options['data_file'],seed=run_seed,model_seed=41)
            with open_response(selected) as run:
                native=run.game.backend
                assert run.config.cohort==PROTOCOL=='glimmer-modern-final-hidden-hg-hndl-v2-cpu'
                assert native.metadata()['initialization']==dict(policy=INITIALIZATION_POLICY,seed=run_seed)
                inference_config=run.binding.inference.configuration
                assert inference_config['initialization_policy']==INITIALIZATION_POLICY
                assert inference_config['seed']==run_seed
                snapshots.append(dict(packet=run.checkpoint()['sha256'],
                    adapter=deepcopy(run.game.network.state_dict()),
                    bank=native.slider.particles.detach().clone(),
                    critic=deepcopy(run.game.critic.state_dict()),
                    base=deepcopy(native.model.state_dict())))
            assert torch.equal(torch.get_rng_state(),before)
        a,b,c=snapshots
        assert a['packet']==b['packet']
        for key in ('adapter','critic','base'):
            assert all(torch.equal(value,b[key][name]) for name,value in a[key].items())
        assert torch.equal(a['bank'],b['bank'])
        assert not torch.equal(a['bank'],c['bank'])
        assert any(not torch.equal(value,c['adapter'][name]) for name,value in a['adapter'].items())
        assert any(not torch.equal(value,c['critic'][name]) for name,value in a['critic'].items())
        assert all(torch.equal(value,c['base'][name]) for name,value in a['base'].items())
    finally:
        torch.set_rng_state(saved)


def test_low_level_initialization_scopes_restore_cpu_rng(config,monkeypatch):
    _forbid_cuda_rng(monkeypatch)
    saved=torch.get_rng_state().clone()
    snapshots=[]
    try:
        for caller_seed in (444,555):
            _caller_cpu_seed(caller_seed)
            before=torch.get_rng_state().clone()
            native=Native(make_model(41),seed=9)
            try:
                critic=ReadoutCritic(12,16,seed=9)
                snapshots.append((_hash(native.network.state_dict()),_hash(critic.state_dict())))
            finally:
                native.close()
            assert torch.equal(before,torch.get_rng_state())
        assert snapshots[0]==snapshots[1]
    finally:
        torch.set_rng_state(saved)


@pytest.mark.parametrize('stage',['bank','projection','critic'])
def test_failed_random_constructor_restores_caller_rng(monkeypatch,stage):
    import glimmer_modern.native as product
    _forbid_cuda_rng(monkeypatch)
    model=make_model(41).train()
    original_keys=[set(vars(m)) for m in model.modules()]
    before=torch.get_rng_state().clone()
    def fail(*a,**kw):
        torch.rand(13,device='cpu')
        raise RuntimeError('random constructor failed')
    if stage=='bank':
        monkeypatch.setattr(product.torch,'randn',fail)
    elif stage=='projection':
        monkeypatch.setattr(product,'HNDLParticleProjection',fail)
    else:
        monkeypatch.setattr(product,'verify_sources',lambda: (fail,{}))
    with pytest.raises(RuntimeError,match='random constructor failed'):
        if stage=='critic':
            ReadoutCritic(12,16,seed=10)
        else:
            Native(model,seed=10)
    assert torch.equal(before,torch.get_rng_state())
    assert [set(vars(m)) for m in model.modules()]==original_keys
    assert model.training


def test_old_initialization_cohort_rejected_without_relabeling(config,tmp_path,monkeypatch):
    import glimmer_modern.lifecycle as product
    from particle_sliders.lifecycle import atomic_save
    from types import SimpleNamespace
    old_options=dict(config.options)
    old_options.pop('initialization_policy')
    old=SessionConfig(config.backend,'glimmer-modern-final-hidden-hg-hndl-v1-cpu',**old_options)
    with open_response(config) as run:
        packet=run.checkpoint()
    packet['manifest']['config']=old.as_dict()
    packet['sha256']=_hash({k:v for k,v in packet.items() if k!='sha256'})
    atomic_save(tmp_path/'old-cohort.pt',packet)
    def forbidden(*a,**kw):
        pytest.fail('old initialization cohort reached preparation')
    monkeypatch.setattr(product,'BACKEND',replace(BACKEND,prepare=forbidden))
    with pytest.raises(CheckpointError,match='kind/config/cohort'):
        open_response(config,resume=tmp_path/'old-cohort.pt')
    with pytest.raises(ValueError,match='initialization policy'):
        product.BACKEND.open(old)
    with pytest.raises(ValueError,match='construction contract'):
        product.validate_inference({},SimpleNamespace(configuration=dict(mode='cpu-fixture',width=12,model_seed=41)))
