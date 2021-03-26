from .ResNet_CSI_model import *
from .Prototypical_1DResNet import *
__factory = {
    'PrototypicalResNet':PrototypicalResNet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
