"""Private particle-only studio, with native execution admitted by studio_guard."""
from datetime import datetime,timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
import argparse,hashlib,json,os
import torch
from . import sampler_server as studio
from .sampler import ResidentSliders
from .demo_native import load_demo_local
from .upstream import load_upstream
from .worker_telemetry import cuda_heartbeat
from .studio_files import atomic_json
from .particle_catalog_runtime import replace_particles
from .sampler_stream import StreamingBackend,StreamingQueue,streaming_handler_for
from .restored_sampler import LabeledRequest,catalog_particles,prompt_tokens,ORIGINAL
from .studio_config import load_config,validate_runtime


def serve(plan,artifacts,model=None,tokenizer=None,sliders=None,ids=None,*,server):
    backend=StreamingBackend(model,tokenizer,sliders) if sliders is not None else None
    class Queue(StreamingQueue):
        catalog_sha=hashlib.sha256(Path(plan['particle_catalog']).read_bytes()).hexdigest()
        last_attempt_sha=catalog_sha;catalog_error=None
        def state(self):
            with self.lock:
                observed=hashlib.sha256(Path(plan['particle_catalog']).read_bytes()).hexdigest()
                idle=self.pending.empty() and not any(j['status']=='running' for j in self.jobs.values())
                if observed!=self.last_attempt_sha and idle:
                    self.last_attempt_sha=observed
                    try:
                        if sliders is None:raise ValueError('Restart preview to inspect a changed catalog')
                        candidates=catalog_particles(plan['particle_catalog'])
                        def probe():return model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.cpu()
                        receipt=replace_particles(sliders,candidates,probe=probe)
                        self.artifacts=tuple(candidates);self.catalog_sha,self.catalog_error=observed,None
                        atomic_json(Path(plan['attempt'])/'CATALOG-APPLIED.json',dict(receipt,UTC=datetime.now(timezone.utc).isoformat(),catalog_sha256=observed,model_loads=1))
                    except Exception as error:
                        self.catalog_error=f'{type(error).__name__}: {error}'
                value=super().state()
                value.update(ui_revision='reasoning-framing-p1-v2',feature_revision='particle-speed-scope-v1',default_particle_id=None,
                    sampling_format='particle_only_no_distilled_fallback',particle_execution=plan.get('particle_execution','scripted'),
                    generation_scopes=['thinking','response','both'],inference_system_conditions=['explicit_blank','original','omitted','custom'],
                    original_neutral_text=ORIGINAL,lease_end_UTC=plan['deadline_UTC'],particle_catalog_sha256=self.catalog_sha,
                    particle_catalog_error=self.catalog_error,automatic_particle_replacement=sliders is not None,preview=sliders is None)
                return value
    runtime=Queue(artifacts,backend,qualification_root=None)
    if backend is None:runtime.status='Preview — model not loaded; recorded runs only'
    parent=streaming_handler_for(runtime,plan['host'],plan['port'])
    class Handler(parent):
        def do_GET(self):
            if self.path.split('?',1)[0]=='/':
                authority=f"{plan['host']}:{plan['port']}"
                if self.headers.get('Host')!=authority or self.headers.get('Origin') not in (None,'http://'+authority):
                    return self.reply(403,dict(error='Expected private sampler origin'))
                return self.reply(200,Path(__file__).with_name('studio.html').read_bytes(),'text/html; charset=utf-8')
            return super().do_GET()
    ready=dict(runtime.state(),UTC=datetime.now(timezone.utc).isoformat(),url=f"http://{plan['host']}:{plan['port']}/",strength_zero_exact=sliders is not None,training_updates=0)
    server.RequestHandlerClass=Handler;server.daemon_threads=True;server.server_activate()
    atomic_json(Path(plan['attempt'])/'ready.json',ready)
    server.serve_forever()


def run(plan):
    validate_runtime(plan);attempt=Path(plan['attempt']);attempt.mkdir(parents=True,exist_ok=True)
    artifacts=catalog_particles(plan['particle_catalog'])
    if not artifacts:raise ValueError('Catalog requires at least one verified raw particle adapter')
    # Reserve the listener before any expensive model load; a busy address must
    # fail before CUDA allocation and never leave a false ready receipt.
    with ThreadingHTTPServer((plan['host'],plan['port']),None,bind_and_activate=False) as server:
        server.server_bind()
        if plan.get('preview'):return serve(plan,artifacts,server=server)
        load_upstream(plan['upstream']);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        with cuda_heartbeat(attempt,plan['gpu_uuid']):
            model,tokenizer,identity=load_demo_local(plan['snapshot'],expected_gpu_uuid=plan['gpu_uuid'],guardian_pid=os.getppid())
            request=LabeledRequest.parse(dict(system_condition='explicit_blank',prompt='Say hello.',strengths={},max_new_tokens=8),[a.id for a in artifacts])
            ids=prompt_tokens(tokenizer,request).to(next(model.parameters()).device)
            with torch.inference_mode():before=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.cpu()
            sliders=ResidentSliders(model,identity,artifacts,execution='nf4',particle_execution=plan.get('particle_execution','scripted'))
            with torch.inference_mode(),sliders.using({}):after=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.cpu()
            if not torch.equal(before,after):raise RuntimeError('Resident OFF changed base logits')
            serve(plan,artifacts,model,tokenizer,sliders,ids,server=server)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    if args.preview:
        from datetime import timedelta
        plan=load_config(args.config);plan.update(preview=True,deadline_UTC=(datetime.now(timezone.utc)+timedelta(seconds=plan['duration_seconds'])).isoformat(),attempt=plan.get('run_dir','outputs/studio-preview'))
        import signal
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Preview lease ended')));signal.alarm(plan['duration_seconds'])
    else:plan=json.loads(args.config.read_text())
    run(plan)
if __name__=='__main__':main()
