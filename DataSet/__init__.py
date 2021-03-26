from .ARIL import ARIL
from .Widar import Widar
from .CSI_301 import CSI_301
import os
from pathlib import Path

__factory = {
    'aril':ARIL,
    'widar':Widar,
    'csi_301':CSI_301,
}


def names():
    return sorted(__factory.keys())


def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__


def create(name, root, *args, **kwargs):
    if root is not None:
        root = root/get_full_name(name)
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)
