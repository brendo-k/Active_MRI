"""The C/C++ compilers torch.compile builds its kernels with, and how many compiled variants it may keep."""
import os
import shutil

RECOMPILE_LIMIT = 32


def use_system_compilers_if_missing() -> None:
    """Inductor builds its kernels with $CC / $CXX; a conda activation can point them at compilers the env lacks."""
    for variable, system_compiler in (('CC', 'gcc'), ('CXX', 'g++')):
        if os.environ.get(variable) and shutil.which(os.environ[variable]) is None:
            os.environ[variable] = system_compiler


def allow_recompiles(limit: int = RECOMPILE_LIMIT) -> None:
    """Let dynamo keep up to `limit` compiled variants of each function (its default is 8).

    Every grad mode, requires_grad pattern and batch shape compiles its own variant. A frozen copy of the VarNet running
    beside the trained one (sampler.input=recon) needs more than 8, and the fused coil kernels compile with
    fullgraph=True, so reaching the limit raises instead of falling back to eager. A run that stays under 8 variants
    compiles exactly as before.
    """
    import torch._dynamo
    config = torch._dynamo.config
    name = 'recompile_limit' if hasattr(config, 'recompile_limit') else 'cache_size_limit'
    setattr(config, name, max(getattr(config, name), limit))
