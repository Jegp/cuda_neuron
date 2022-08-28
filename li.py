from collections import namedtuple
from email.policy import default
from typing import NamedTuple
import torch

import li_cpp
import tuple


class LIParameters(
    tuple.TupleBase,
    namedtuple(
        "LIParameters",
        field_names=["leak", "tau_inv"],
        defaults=[torch.as_tensor(0.0, dtype=torch.float32), torch.as_tensor(900, dtype=torch.float32)],
    ),
):
    pass

tuple.register_tuple(LIParameters)

class LIState(tuple.TupleBase, namedtuple("LIState", ["v"], defaults=[torch.as_tensor(0.0, dtype=torch.float32)])):
    pass

tuple.register_tuple(LIState)

def li_step_python(i, state: LIState, p: LIParameters, dt: float = 0.001):
    v = dt * p.tau_inv * (p.leak - state.v + i)
    return v, LIState(v)

def li_integral_python(xs, state: LIState, p: LIParameters, dt: float = 0.001):
    out = []
    for x in xs:
        v, state = li_step_python(x, state, p, dt)
        out.append(v)
    return torch.stack(out), state


def li_step_cuda(i, state: LIState, p: LIParameters, dt: float = 0.001):
    v = li_cpp.step_forward(i, state.v, p.leak, p.tau_inv, dt)
    return v, LIState(v)

def li_integral_cuda(i, state: LIState, p: LIParameters, dt: float = 0.001):
    return li_cpp.integral_forward(i, state.v, p.leak, p.tau_inv, dt)


# class LLTMFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, v):
#         outputs = li_cuda.forward(input, v)
#         z, v = outputs
#         ctx.save_for_backward(outputs)

#         return z, v

#     @staticmethod
#     def backward(ctx, grad_h, grad_cell):
#         outputs = lltm_cpp.backward(
#             grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
#         d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
#         return d_input, d_weights, d_bias, d_old_h, d_old_cell