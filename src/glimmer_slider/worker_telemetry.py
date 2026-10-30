"""Owned-worker CUDA allocator telemetry when external NVML queries stall."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


def progress_fallback(attempt, pid, uuid, process_birth, baseline_bytes, cap_bytes, *, now=None):
    """Unknown global NVML data require fresh owned stats AND actual progress."""
    now = time.time() if now is None else now
    root = Path(attempt)
    try:
        value = json.loads((root/'worker-cuda.json').read_text())
        if (value['pid'] != pid or value['birth'] != process_birth
                or value['gpu_uuid'] != uuid or not 0 <= now-value['timestamp'] <= 15):
            return None
        if value.get('error'):
            raise RuntimeError('Owned CUDA telemetry failed: '+value['error'])
        reserved = value['reserved_bytes']
        if type(reserved) is not int or reserved < 0:
            raise ValueError('Invalid owned CUDA allocation telemetry')
        # Keep context/non-PyTorch overhead margin; global usage remains unknown.
        upper = reserved + baseline_bytes + 1024**3
        if upper > cap_bytes:
            raise RuntimeError('Conservative owned CUDA usage exceeds VRAM ceiling')
        files = [root/'progress.json', root/'worker.log', *root.glob('incremental-*.jsonl')]
        fresh = any(f.is_file() and 0 <= now-f.stat().st_mtime <= 120 for f in files)
        if not fresh:
            return None
        return dict(status='unknown_NVML_owned_progress_verified', reserved_bytes=reserved,
                    conservative_owned_plus_baseline_bytes=upper, global_usage_verified=False)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


@contextmanager
def cuda_heartbeat(attempt, uuid):
    import torch
    from .admission import require_cuda_identity
    require_cuda_identity(uuid)
    root = Path(attempt)
    pid = os.getpid()
    birth = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            value = dict(pid=pid, birth=birth, gpu_uuid=uuid, timestamp=time.time())
            try:
                value.update(allocated_bytes=torch.cuda.memory_allocated(0),
                             reserved_bytes=torch.cuda.memory_reserved(0))
            except Exception as error:
                value['error'] = f'{type(error).__name__}: {error}'
            temporary = root/'worker-cuda.writing'
            temporary.write_text(json.dumps(value)+'\n')
            temporary.replace(root/'worker-cuda.json')
            stop.wait(2)
    thread = threading.Thread(target=monitor, name='owned-cuda-telemetry', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=3)
