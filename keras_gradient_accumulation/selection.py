from .backend import TF_KERAS

__all__ = ['AdamAccumulated']


if TF_KERAS:
    from .optimizer_v2 import AdamAccumulated
else:
    from .optimizer_v1 import AdamAccumulated
