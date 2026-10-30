"""Portable private-studio configuration; never downloads models or credentials."""
from pathlib import Path
from datetime import datetime,timezone
import ipaddress,json,os

def validate_address(host,port):
    address=ipaddress.ip_address(host)
    if not (address.is_loopback or address in ipaddress.ip_network('100.64.0.0/10')):
        raise ValueError('Bind only loopback or an operator-configured Tailscale address')
    if type(port) is not int or not 1024<=port<=65535:raise ValueError('Port must be1024..65535')

def load_config(path):
    path=Path(path).resolve();value=json.loads(path.read_text())
    allowed={'host','port','snapshot','upstream','particle_catalog','run_dir','gpu_uuid','duration_seconds','leases','protected_processes','particle_execution'}
    if not isinstance(value,dict) or set(value)-allowed:raise ValueError('Unknown studio configuration field')
    value.setdefault('host','127.0.0.1');value.setdefault('port',8789);value.setdefault('duration_seconds',7200);value.setdefault('particle_execution','scripted');value.setdefault('protected_processes',[])
    validate_address(value['host'],value['port'])
    if type(value['duration_seconds']) is not int or not 1<=value['duration_seconds']<=28800:raise ValueError('Studio lease must be1..28800seconds')
    if value['particle_execution'] not in ('eager','scripted'):raise ValueError('Choose eager or scripted execution explicitly')
    for name in ['particle_catalog','run_dir','snapshot','upstream']:
        if name in value:value[name]=str((path.parent/value[name]).resolve())
    if 'particle_catalog' not in value:raise ValueError('particle_catalog is required')
    value['leases']=[str((path.parent/x).resolve()) for x in value.get('leases',[])]
    return value

def validate_runtime(plan):
    validate_address(plan['host'],plan['port'])
    if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['deadline_UTC']):raise ValueError('Studio lease expired')
    if not plan.get('preview') and (os.environ.get('GLIMMER_GUARDIAN')!=str(os.getppid()) or os.environ.get('GLIMMER_RAM_POLICY')!='telemetry_only'):
        raise ValueError('Native studio requires the bounded studio_guard parent')
    if plan.get('preview') and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preview must explicitly hide CUDA')
