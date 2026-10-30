import json
from pathlib import Path
import pytest
from glimmer_slider.owned_memory_guard import check_memory,ADMISSION_AVAILABLE
from glimmer_slider.studio_config import load_config,validate_address
from glimmer_slider import studio_guard as guard


def test_actual_available_reserve_and_oom_policy():
    check_memory(30*1024**3,6,6,admission=True)  # Host/shared PSI never enters this decision.
    check_memory(16*1024**3,6,6)
    for available,old,new,admission in [(23*1024**3-1,6,6,True),(16*1024**3-1,6,6,False),(30*1024**3,6,7,False),(None,6,6,False)]:
        with pytest.raises(RuntimeError):check_memory(available,old,new,admission=admission)


def test_config_relative_paths_private_binding_and_duration(tmp_path):
    f=tmp_path/'studio.json';f.write_text(json.dumps(dict(particle_catalog='catalog.json',run_dir='runs/demo',duration_seconds=600)))
    value=load_config(f);assert value['host']=='127.0.0.1' and value['particle_catalog']==str(tmp_path/'catalog.json')
    for host in ['0.0.0.0','8.8.8.8']:
        with pytest.raises(ValueError):validate_address(host,8789)
    f.write_text(json.dumps(dict(particle_catalog='catalog.json',duration_seconds=0)))
    with pytest.raises(ValueError):load_config(f)


def test_guard_terminates_owned_worker_before_releasing_locks(tmp_path,monkeypatch):
    config=dict(host='127.0.0.1',port=8789,snapshot='snapshot',upstream='upstream',particle_catalog='catalog.json',gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',run_dir=str(tmp_path/'run'),duration_seconds=600,leases=[str(tmp_path/f'lease{i}') for i in range(3)],protected_processes=[],particle_execution='scripted')
    events=[]
    class Worker:
        pid=43210;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):events.append('worker_wait');self.returncode=-15
    worker=Worker()
    monkeypatch.setattr(guard.subprocess,'Popen',lambda *a,**kw:worker)
    monkeypatch.setattr(guard,'gpu',lambda uuid:dict(pids=set(),used_bytes=0))
    monkeypatch.setattr(guard,'available',lambda:32*1024**3);monkeypatch.setattr(guard,'oom',lambda:6)
    monkeypatch.setattr(guard,'cgroup',lambda pid:'/shared-app');monkeypatch.setattr(guard,'descendants',lambda pid:[43210])
    monkeypatch.setattr(guard,'process_memory',lambda pid:dict(VmRSS=100));monkeypatch.setattr(guard,'owned_indicators',lambda pids:{})
    monkeypatch.setattr(guard.signal,'signal',lambda *a:None);monkeypatch.setattr(guard.signal,'setitimer',lambda *a:None)
    monkeypatch.setattr(guard.os,'killpg',lambda *a:events.append('owned_group_stop'))
    def stop_after_one_tick(seconds):raise TimeoutError('test deadline')
    monkeypatch.setattr(guard.time,'sleep',stop_after_one_tick)
    guard.run(config)
    assert events==['owned_group_stop','worker_wait']
    receipt=json.loads((tmp_path/'run/receipt.json').read_text());assert receipt['status']=='stopped' and receipt['worker_gone'] and receipt['gpu_allocation_released']
    import fcntl
    for name in config['leases']:
        with open(name,'rb') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_busy_listener_fails_before_model_loading_or_ready_receipt(tmp_path,monkeypatch):
    import socket
    from glimmer_slider import studio
    monkeypatch.setattr(studio,'validate_runtime',lambda plan:None)
    monkeypatch.setattr(studio,'catalog_particles',lambda path:[object()])
    def forbidden(*args,**kwargs):raise AssertionError('A busy listener must fail before loading')
    monkeypatch.setattr(studio,'load_demo_local',forbidden)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        plan=dict(host='127.0.0.1',port=occupied.getsockname()[1],attempt=str(tmp_path/'attempt'),particle_catalog='unused')
        with pytest.raises(OSError):studio.run(plan)
    assert not (tmp_path/'attempt'/'ready.json').exists()
