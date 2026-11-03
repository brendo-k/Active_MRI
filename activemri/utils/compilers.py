"""The C/C++ compilers torch.compile builds its kernels with."""
import os
import shutil


def use_system_compilers_if_missing() -> None:
    """Inductor builds its kernels with $CC / $CXX; a conda activation can point them at compilers the env lacks."""
    for variable, system_compiler in (('CC', 'gcc'), ('CXX', 'g++')):
        if os.environ.get(variable) and shutil.which(os.environ[variable]) is None:
            os.environ[variable] = system_compiler
