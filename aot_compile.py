# AOT Autograd has a suite of already integrated backends. Lets import the NNC compiler backend - ts_compile
from functorch.compile import ts_compile

# Lets compile the forward and backward through ts_compile.
aot_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile)

# Correctness checking. Lets clone the input so that we can check grads.
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()
assert torch.allclose(ref, res)
assert torch.allclose(a.grad, cloned_a.grad)
assert torch.allclose(b.grad, cloned_b.grad)
assert torch.allclose(c.grad, cloned_c.grad)
assert torch.allclose(d.grad, cloned_d.grad)

# Lets write a function to benchmark the forward and backward pass
import time
import statistics

def bench(fn, args, prefix):
    warmup = 10
    iterations = 100

    for _ in range(warmup):
        ref = fn(*args)
        ref.sum().backward()
    
    fw_latencies = []
    bw_latencies = []
    for _ in range(iterations):
        for arg in args:
            arg.grad = None

        fw_begin = time.perf_counter()
        ref = fn(*args)
        fw_end = time.perf_counter()

        loss = ref.sum() 

        bw_begin = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()

        fw_latencies.append(fw_end - fw_begin)
        bw_latencies.append(bw_end - bw_begin)
    
    avg_fw_latency = statistics.mean(fw_latencies) * 10**6
    avg_bw_latency = statistics.mean(bw_latencies) * 10**6
    print(prefix, "Fwd = " + str(avg_fw_latency) + " us", "Bwd = " + str(avg_bw_latency) + " us", sep=', ')