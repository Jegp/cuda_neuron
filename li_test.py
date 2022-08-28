import time
import timeit

import torch

import li

def benchmark(f, n: int = 7):
    b = timeit.timeit(f)
    return b

def data(*shape):
    return f"""
p = li.LIParameters().to("cuda")
i = torch.zeros(*{shape}, device="cuda")
s = li.LIState(torch.zeros(*{shape[1:]}, device="cuda"))
"""

print("Python:", benchmark(f"import torch, li; {data(1, 10)}li.li_manual(i, s, p)"))
print("CUDA:", benchmark(f"import torch, li; {data(1, 10)}li.li_manual(i, s, p)"))