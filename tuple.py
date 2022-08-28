from collections import namedtuple
import functools
from typing import Any, Callable, Dict, Type, TypeVar

import torch
from torch.utils._pytree import PyTree, _namedtuple_flatten, _namedtuple_unflatten, _register_pytree_node, tree_map

T = TypeVar('T')

def map_only(ty: Type[T]) -> Callable[[Callable[[T], Any]], Callable[[Any], Any]]:
    """
    Suppose you are writing a tree_map over tensors, leaving everything
    else unchanged.  Ordinarily you would have to write:
        def go(t):
            if isinstance(t, Tensor):
                return ...
            else:
                return t
    With this function, you only need to write:
        @map_only(Tensor)
        def go(t):
            return ...
    You can also directly use 'tree_map_only'
    """
    def deco(f: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(f)
        def inner(x: T) -> Any:
            if isinstance(x, ty):
                return f(x)
            else:
                return x
        return inner
    return deco

def tree_map_only(ty: Type[T], fn: Callable[[T], Any], pytree: PyTree) -> PyTree:
    return tree_map(map_only(ty)(fn), pytree)

def register_tuple(typ: Any):
    _register_pytree_node(typ, _namedtuple_flatten, _namedtuple_unflatten)

class TupleBase:

    def float(self):
        return tree_map_only(torch.Tensor, lambda x: x.float(), self)

    def to(self, device):
        return tree_map_only(torch.Tensor, lambda x: x.to("cuda").requires_grad_(x.requires_grad), self)
