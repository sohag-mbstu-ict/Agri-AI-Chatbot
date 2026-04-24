
import torch
import pynvml

def print_gpu_usage():
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU Compute Usage: {util.gpu}% | GPU Memory Usage: {util.memory}%")