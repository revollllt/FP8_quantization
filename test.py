import torch

from quantization.quantizers.fp8_quantizer import *


n_bits = 8
expo_bits = 3
mant_bits = n_bits - expo_bits - 1
bias = 2**(expo_bits-1)

all_values = gen(n_bits, expo_bits, bias)
maxval = get_max_value(expo_bits, bias)
print(maxval)
print(all_values)
print(len(all_values))


input_tensor = torch.randn(1, 5, 5)
tensor_maxval = torch.Tensor([maxval])
tensor_mantissa_bits = torch.Tensor([float(mant_bits)])

result = quantize_to_fp8_ste_MM(input_tensor, n_bits, tensor_maxval, tensor_mantissa_bits, 1)
print(input_tensor)
print(result)