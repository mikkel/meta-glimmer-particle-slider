"""Linux guardian for one explicitly leased, private, inference-only studio."""
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse,ctypes,fcntl,hashlib,json,os,signal,subprocess,sys,time
from .admission import canonical_uuid,descendants,process_memory,VRAM_CAP
from .owned_memory_guard import POLICY,check_memory,pressure,cgroup,owned_indicators
from .worker_telemetry import progress_fallback
from .studio_files import atomic_json
from .studio_config import load_config


def birth(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def oom():return next(int(l.split()[1]) for l in Path('/proc/vmstat').read_text().splitlines() if l.startswith('oom_kill '))
def available():return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
def service_pid(name):
    if not isinstance(name,str) or not name.endswith('.service') or '/' in name:raise ValueError('Expected a systemd service name')
    values=dict(l.split('=',1) for l in subprocess.check_output(['systemctl','show',name,'--property=MainPID','--property=ActiveState','--property=SubState'],text=True,timeout=5).splitlines() if '=' in l)
    if values.get('ActiveState')!='active' or values.get('SubState')!='running':raise RuntimeError('Protected service is not running')
    return int(values['MainPID'])
def freeze_protected(values):
    result=[]
    for entry in values:
        if not isinstance(entry,dict) or set(entry)-{'pid','service'}:raise ValueError('Protected entries accept pid or service')
        pid=service_pid(entry['service']) if entry.get('service') else entry.get('pid')
        if type(pid) is not int or pid<=0:raise ValueError('Expected positive protected PID')
        if entry.get('pid',pid)!=pid:raise ValueError('Protected service/PID differs')
        exe=Path(f'/proc/{pid}/exe').resolve(strict=True)
        result.append(dict(pid=pid,birth=birth(pid),exe=str(exe),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),service=entry.get('service')))
    return result

def verify_protected(values):
    for entry in values:
        if entry.get('service') and service_pid(entry['service'])!=entry['pid']:raise RuntimeError('Protected service identity changed')
        if birth(entry['pid'])!=entry['birth'] or str(Path(f"/proc/{entry['pid']}/exe").resolve(strict=True))!=entry['exe']:raise RuntimeError('Protected process identity changed')
        # Initial binary hash is retained; identity is never followed to a replacement PID.

def gpu(uuid):
    value=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
    if canonical_uuid(value[0].strip(),require_prefix=True)!=uuid:raise RuntimeError('Physical GPU identity changed')
    apps=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=30).strip()
    return dict(used_bytes=int(float(value[1])*1024**2),pids={int(l.split(',')[0]) for l in apps.splitlines() if l})

def run(config):
    for field in ['gpu_uuid','snapshot','upstream','run_dir']:
        if not config.get(field):raise ValueError(field+' is required for native serving')
    uuid=canonical_uuid(config['gpu_uuid'],require_prefix=True)
    if len(config['leases'])!=3 or len(set(config['leases']))!=3:raise ValueError('Configure three distinct host/admission/device lease paths')
    root=Path(config['run_dir']);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();now=datetime.now(timezone.utc);deadline=now+timedelta(seconds=config['duration_seconds'])
    record=dict(status='admitting',started_UTC=now.isoformat(),deadline_UTC=deadline.isoformat(),guardian_pid=os.getpid(),policy=POLICY)
    atomic_json(root/'receipt.json',record);worker=None;baseline_oom=oom();scope=cgroup(os.getpid());protected=[]
    try:
        check_memory(available(),baseline_oom,oom(),admission=True)
        protected=freeze_protected(config['protected_processes']);protected_pids={v['pid'] for v in protected}
        with ExitStack() as stack:
            locks=[]
            for index,name in enumerate(config['leases']):
                path=Path(name);path.parent.mkdir(parents=True,exist_ok=True)
                handle=stack.enter_context(path.open('a+b'));fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB if index==0 else fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
            initial=gpu(uuid)
            if not initial['pids']<=protected_pids:raise RuntimeError('GPU has an unowned allocation; no worker started')
            plan=dict(config,attempt=str(root),deadline_UTC=deadline.isoformat(),preview=False)
            atomic_json(root/'worker-plan.json',plan)
            parent=os.getpid()
            def before_exec():
                libc=ctypes.CDLL(None,use_errno=True)
                if libc.prctl(1,signal.SIGKILL)!=0 or os.getppid()!=parent:os._exit(125)
                for handle in locks:os.set_inheritable(handle.fileno(),True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,GLIMMER_GUARDIAN=str(parent),GLIMMER_RAM_POLICY='telemetry_only',GLIMMER_LEASE_END_UTC=deadline.isoformat(),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/'worker.log').open('w') as log:
                worker=subprocess.Popen([sys.executable,'-m','glimmer_slider.studio','--config',str(root/'worker-plan.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(f.fileno() for f in locks),preexec_fn=before_exec)
                def finish():
                    if worker.poll() is None:
                        os.killpg(worker.pid,signal.SIGTERM)
                        try:worker.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=5)
                    for _ in range(5):
                        if gpu(uuid)['pids']<=protected_pids:return
                        time.sleep(1)
                    raise RuntimeError('Owned allocation release not confirmed before lease release')
                stack.callback(finish)
                def stop(signum,*_):raise TimeoutError('lease deadline' if signum==signal.SIGALRM else 'guardian stop signal')
                previous={s:signal.signal(s,stop) for s in [signal.SIGALRM,signal.SIGTERM,signal.SIGINT]}
                signal.setitimer(signal.ITIMER_REAL,max(.001,(deadline-datetime.now(timezone.utc)).total_seconds()))
                record.update(status='running',worker_pid=worker.pid,protected_processes=protected);atomic_json(root/'receipt.json',record)
                last_gpu=0;last_good=time.monotonic()
                try:
                    while worker.poll() is None:
                        check_memory(available(),baseline_oom,oom());verify_protected(protected)
                        if cgroup(os.getpid())!=scope:raise RuntimeError('Guardian cgroup identity changed')
                        owned=descendants(os.getpid());memory={}
                        for pid in owned:
                            try:memory[str(pid)]=process_memory(pid)
                            except (OSError,ValueError):pass
                        atomic_json(root/'live.json',dict(UTC=datetime.now(timezone.utc).isoformat(),available=available(),owned_rss=sum(v.get('VmRSS',0) for v in memory.values()),sampler_owned_memory_indicators=owned_indicators(owned),host_PSI=pressure('/proc/pressure/memory'),shared_cgroup=scope,shared_cgroup_is_sampler_exclusive=False,shared_PSI=pressure(Path('/sys/fs/cgroup')/scope.lstrip('/')/'memory.pressure'),policy=POLICY))
                        if time.monotonic()-last_gpu>=5:
                            try:current=gpu(uuid)
                            except subprocess.TimeoutExpired:
                                fallback=progress_fallback(root,worker.pid,uuid,birth(worker.pid),initial['used_bytes'],VRAM_CAP)
                                if time.monotonic()-last_good>90 and not fallback:raise RuntimeError('GPU telemetry unobservable beyond90seconds')
                            else:
                                last_good=time.monotonic()
                                if not current['pids']<=set(owned)|protected_pids or current['used_bytes']>VRAM_CAP:raise RuntimeError('GPU ownership or30GiB ceiling changed')
                            last_gpu=time.monotonic()
                        if sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink())>8*1024**3 or __import__('shutil').disk_usage(root).free<8*1024**3:raise RuntimeError('Owned output cap or8GiB disk reserve reached')
                        time.sleep(1)
                    if worker.returncode!=0:raise RuntimeError('Sampler worker exited with code '+str(worker.returncode))
                    record['status']='completed'
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                    for s,handler in previous.items():signal.signal(s,handler)
    except TimeoutError as error:record.update(status='stopped',reason=str(error))
    except Exception as error:record.update(status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        record.update(finished_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,3),worker_gone=worker is None or worker.poll() is not None,oom_unchanged=oom()==baseline_oom)
        try:
            verify_protected(protected);record['protected_processes_unchanged']=True
            after=gpu(uuid);record['gpu_allocation_released']=not after['pids']-set(v['pid'] for v in protected)
        except Exception as error:record['cleanup_verification_error']=str(error)
        atomic_json(root/'receipt.json',record)
    if record['status']=='failed' or not record.get('gpu_allocation_released'):raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);args=parser.parse_args();run(load_config(args.config))
if __name__=='__main__':main()
