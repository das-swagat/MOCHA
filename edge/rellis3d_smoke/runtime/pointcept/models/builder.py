import copy
import torch.nn as nn

class Registry:
    def __init__(self):
        self._items = {}
    def register_module(self, name=None):
        def deco(cls):
            key = name or cls.__name__
            self._items[key] = cls
            return cls
        return deco
    def build(self, cfg):
        cfg = copy.deepcopy(dict(cfg))
        typ = cfg.pop("type")
        if typ not in self._items:
            raise KeyError(f"{typ} not registered; registered={list(self._items)}")
        return self._items[typ](**cfg)

MODELS = Registry()
def build_model(cfg):
    return MODELS.build(cfg)
