import torch
import math


class Quant:
    def linear(input, bits):
        assert bits >= 1, bits
        if bits == 1:
            return torch.sign(input) - 1
        sf = torch.ceil(torch.log2(torch.max(torch.abs(input))))
        delta = math.pow(2.0, -sf)
        bound = math.pow(2.0, bits - 1)
        min_val = - bound
        max_val = bound - 1
        rounded = torch.floor(input / delta)

        clipped_value = torch.clamp(rounded, min_val, max_val) * delta
        return clipped_value


input = torch.Tensor([x - 5 for x in range(10)])
print(input)
print(Quant.linear(input, 8))
