"""Owned-process attribution plus an explicit available-RAM safety reserve.

Linux PSI in the shared app scope is not per-process PSI. Neither shared nor
host PSI alone decides a stop. Owned faults/swap are indicators, not proof of
memory pressure: loading a cold checkpoint can legitimately fault heavily.
"""
from pathlib import Path
from .admission import HOST_RESERVE

# Maximum observed owned load RSS across four retained sampler attempts:6.663GiB.
MEASURED_PEAK_OWNED_LOAD_BYTES = 7154352128
LOAD_HEADROOM = 7 * 1024**3
ADMISSION_AVAILABLE = HOST_RESERVE + LOAD_HEADROOM

POLICY = dict(version='owned-process-and-available-v1',host_reserve_bytes=HOST_RESERVE,
    admission_available_bytes=ADMISSION_AVAILABLE,load_headroom_bytes=LOAD_HEADROOM,
    measured_peak_owned_load_bytes=MEASURED_PEAK_OWNED_LOAD_BYTES,
    host_PSI='diagnostic_only',shared_cgroup_PSI='diagnostic_shared_not_exclusive',
    owned_attribution='guardian_and_descendants_only',owned_faults_and_swap='diagnostic_not_sole_stop',
    actual_OOM='stop_on_counter_increase',allocation_failure='propagate_and_cleanup')


def check_memory(available, baseline_oom, current_oom, *, admission=False):
    if current_oom != baseline_oom:
        raise RuntimeError('Actual host OOM observed; clean up owned sampler')
    if available is None:
        raise RuntimeError('MemAvailable unavailable; available-RAM protection cannot be verified')
    threshold=ADMISSION_AVAILABLE if admission else HOST_RESERVE
    if available < threshold:
        label='23GiB admission requirement (16GiB reserve +7GiB measured loading headroom)' if admission else '16GiB host reserve'
        raise RuntimeError('MemoryPressurePause: available RAM below '+label)


def pressure(path):
    try:
        result={}
        for line in Path(path).read_text().splitlines():
            name,*items=line.split();result[name]={key:float(value) if key!='total' else int(value)
                for key,value in (item.split('=',1) for item in items)}
        return result
    except (OSError,ValueError):return dict(unavailable=True)


def cgroup(pid):
    line=next(l for l in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if l.startswith('0::'))
    return line[3:]


def owned_indicators(pids):
    result={}
    for pid in pids:
        try:
            root=Path('/proc')/str(pid);stat=(root/'stat').read_text().rsplit(')',1)[1].split()
            status=dict(l.split(':',1) for l in (root/'status').read_text().splitlines() if ':' in l)
            result[str(pid)]=dict(major_faults=int(stat[9]),minor_faults=int(stat[7]),
                blkio_delay_ticks=int(stat[39]) if len(stat)>39 else None,
                VmSwap=int(status.get('VmSwap','0 kB').split()[0])*1024,state=stat[0])
        except (OSError,ValueError,IndexError):result[str(pid)]=dict(unavailable=True)
    return result
