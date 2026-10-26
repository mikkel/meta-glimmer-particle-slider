"""Execute the pinned original implementations; never a fallback local loss."""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REVISION = "1f5b380d317b73779a8aeab63b6bec0093c2c65c"
FILES = {
    "conceptmod/textsliders/particle_bridge_gan.py": "738d50200ea5de9a9c4943b34474f862b06a9321cd78e6838196ef9b9d6993f0",
    "conceptmod/textsliders/lora.py": "2bcf1a1267ca0369e0e2d4e2584db6821b2ab518c9cd3c535292d20a5e25d85d",
    "conceptmod/textsliders/lm_adv.py": "8b42a4bad78e2a5314289ef23598908a5d967281aab91c88841ef344a9d33a68",
    "analysis/slider2d/adv.py": "1a939c65626574028a90f7a96c95b6363d59f4199833d74950ba625b3f9491fb",
    "analysis/slider2d/grad_regularizers.py": "c1372f537c8f53b6d3fdedc3dc9521533d3b5f954c8d1e012af02d5c87f864e5",
    "packages/particle-sliders-core/src/particle_sliders/distillation.py": "b277e561517a9cb40d1a399ad07c6350aa22a287232aea816b7f6e434e18a232",
}
CORE_FILES = dict(zip(
    ("__init__", "formulation", "reference", "grad_regularizers", "recipe", "endpoint_game", "formulation_game", "teachers"),
    ("5572dfa0f1e599f62f3383c7cdc4c0d4b0580e4373ac77497fa0962b6b8cef68",
     "7e82efbef5e67107c717d8249d47542dd55e07ce0d508154b9ddb06848e6ad0a",
     "d35963f072c4cd4acc92d255af5d3ea32535d2deb3d4a3c18625d85f30c42de3",
     "ada2b754b35f4970360939856f03b2ecc8407fb53e0bc8139e83d5ec59997af6",
     "63726226e681e517c49943f6150989a8277031762c6253a338892dac340b5600",
     "85ea55b67a4f9112e8d9dd1df1096d3488ca8f2c69c666174370b8f3a5c77655",
     "f7e78be912e309e01f305ca4ed09a5e25fe67fa720e736694fe343be89fc9f74",
     "9a35fcff05a15d00f6aa67b5cb9e53cf55526d7e06631ef8cfeb0169519818d2")))
_loaded = None


def load_upstream(checkout):
    global _loaded
    root = Path(checkout).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"particle-sliders revision differs: {actual}")
    for name, digest in FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Modified upstream source: {name}")
    core_root = root / "packages/particle-sliders-core/src"
    for name, digest in CORE_FILES.items():
        if hashlib.sha256((core_root / "particle_sliders" / (name + ".py")).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Modified upstream core: {name}")
    dependency = importlib.metadata.distribution("particlegan")
    provenance = json.loads(dependency.read_text("direct_url.json") or "{}")
    if (provenance.get("url") != "https://github.com/255BITS/ParticleGAN.git"
            or provenance.get("vcs_info", {}).get("commit_id") != "82b72661c03377f348ccb76de37cd4ca02d45a8d"):
        raise ValueError("Install the pinned ParticleGAN Git dependency with recorded provenance")
    if _loaded is not None:
        if _loaded.root != root:
            raise ValueError("A different upstream checkout is already loaded")
        return _loaded
    # Prevent writing __pycache__ into the dependency checkout.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(core_root))
    core = importlib.import_module("particle_sliders")
    if Path(core.__file__).resolve() != core_root / "particle_sliders/__init__.py":
        raise RuntimeError("Core import shadowed by another installation")
    stamp = core.winning_formulation()
    stamp.require(stamp.as_dict())
    game = importlib.import_module("conceptmod.textsliders.particle_bridge_gan")
    lora = importlib.import_module("conceptmod.textsliders.lora")
    for module, relative in ((game, "conceptmod/textsliders/particle_bridge_gan.py"),
                             (lora, "conceptmod/textsliders/lora.py")):
        if Path(module.__file__).resolve() != root / relative:
            raise RuntimeError("Upstream import shadowed by another installation")
    file = root / "packages/particle-sliders-core/src/particle_sliders/distillation.py"
    spec = importlib.util.spec_from_file_location("_glimmer_upstream_distillation", file)
    distillation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distillation)
    _loaded = SimpleNamespace(root=root, game=game, LoRAModule=lora.LoRAModule,
                              fit_routed_down=distillation.fit_routed_down, stamp=stamp)
    return _loaded


def api():
    if _loaded is None:
        raise RuntimeError("Call load_upstream(path_to_pinned_particle_sliders) first")
    return _loaded
