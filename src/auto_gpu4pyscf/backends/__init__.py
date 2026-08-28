"""The two ways to have gpu4pyscf: a docker image, or a local environment."""
from .docker import DockerBackend
from .env import EnvBackend

BACKENDS = {"docker": DockerBackend, "env": EnvBackend}


def get(settings):
    return BACKENDS[settings.backend](settings)
