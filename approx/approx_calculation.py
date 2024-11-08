import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import cupy as cp
from quantization.autoquant_utils import QuantConv, QuantLinear, BNQConv

from quantization.base_quantized_classes import QuantizedActivation, QuantizedModule
from quantization.hijacker import QuantizationHijacker, activations_set
from quantization.quantization_manager import QuantizationManager
from quantization.quantized_folded_bn import BNFusedHijacker

from quantization.quantizers.fp8_quantizer import quantize_to_fp8_ste_MM

# from approx.approx_matmul_whole_v6 import *
# from approx.approx_matmul_whole_v8 import *
from approx.approx_matmul_whole_v9 import *

import time

import pandas as pd

class CustomConv2dNumPy(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomConv2dNumPy, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        self.use_bias = bias is not None

    def forward(self, input):
        # 确保输入是连续的
        input = input.contiguous()
        
        # 将输入转换为NumPy数组
        input_np = input.detach().cpu().numpy()
        weight_np = self.weight.detach().cpu().numpy()
        
        batch_size, in_channels, in_height, in_width = input_np.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_np.shape
        
        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # 初始化输出
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        
        # 对输入进行填充
        if self.padding[0] > 0 or self.padding[1] > 0:
            input_np = np.pad(input_np, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        
        # 执行分组卷积
        for b in range(batch_size):
            for g in range(self.groups):
                for c_out in range(out_channels // self.groups):
                    c_out_idx = g * (out_channels // self.groups) + c_out
                    for h_out in range(out_height):
                        for w_out in range(out_width):
                            h_start = h_out * self.stride[0]
                            w_start = w_out * self.stride[1]
                            h_end = h_start + kernel_height * self.dilation[0]
                            w_end = w_start + kernel_width * self.dilation[1]
                            
                            # 提取当前位置的输入块
                            input_slice = input_np[b, 
                                                   g * in_channels_per_group:(g + 1) * in_channels_per_group, 
                                                   h_start:h_end:self.dilation[0], 
                                                   w_start:w_end:self.dilation[1]]
                            
                            # 执行卷积操作
                            output[b, c_out_idx, h_out, w_out] = np.sum(input_slice * weight_np[c_out_idx])
        
        # 添加偏置（如果有）
        if self.use_bias:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np.reshape(1, -1, 1, 1)
        
        # 将结果转回PyTorch张量
        return torch.from_numpy(output).to(input.device).type(input.dtype)
    

class CustomLinearNumPy(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinearNumPy, self).__init__(in_features, out_features, bias)
        self.use_bias = bias is not None

    def forward(self, input):
        input_np = input.detach().cpu().numpy()
        weight_np = self.weight.detach().cpu().numpy()
        
        batch_size, in_features = input_np.shape
        out_features = self.weight.shape[0]
        
        output = np.zeros((batch_size, out_features))
        
        for b in range(batch_size):
            for c_out in range(out_features):
                output[b, c_out] = np.dot(input_np[b], weight_np[c_out])  # replace this dot operation with approximate dot operation after

        if self.use_bias:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np
        
        return torch.from_numpy(output).to(input.device).type(input.dtype)
        

'''
CustomConv2dNumPy with Quantization
'''
class QCustomConv2dNumPy(QuantizationHijacker, nn.Conv2d):
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        # 实现im2col函数
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出大小
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        
        # 初始化输出数组
        col = np.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width))
        
        # 填充col数组
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑col数组
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        return col
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return np.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # Convert input to NumPy arrays
        input_np = x.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        
        batch_size, in_channels, in_height, in_width = input_np.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_np.shape
        
        # Compute output dimensions
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # Reshape input using im2col
        input_col = self.im2col(input_np, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col shape: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # Handle grouped convolution
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # Reshape weights
        weight_col = weight_np.reshape(out_channels, -1)  # Shape: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # Initialize output
        output = np.zeros((batch_size * out_height * out_width, out_channels))
        
        for g in range(self.groups):
            # Indices for input
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # Indices for output channels
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # Shape: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            
            # Perform matrix multiplication
            # output_group = self.multiply(input_group[:, :, np.newaxis], weight_group.T[np.newaxis, :, :]).sum(axis=1)
            output_group = self.multiply(input_group, weight_group.T)

            # Store the output
            output[:, start_out:end_out] = output_group
        
        # Reshape output
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.transpose(0, 3, 1, 2)  # Shape: (batch_size, out_channels, out_height, out_width)
        
        # Add bias if needed
        if bias is not None:
            bias_np = bias.detach().cpu().numpy()
            output += bias_np.reshape(1, -1, 1, 1)
        
        # Convert back to torch tensor
        return torch.from_numpy(output).to(x.device).type(x.dtype)


class QCustomBNConv2dNumPy(BNFusedHijacker, nn.Conv2d):
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        # 实现im2col函数
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出大小
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        
        # 初始化输出数组
        col = np.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width))
        
        # 填充col数组
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑col数组
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        return col
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return np.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # Convert input to NumPy arrays
        input_np = x.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        
        batch_size, in_channels, in_height, in_width = input_np.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_np.shape
        
        # Compute output dimensions
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # Reshape input using im2col
        input_col = self.im2col(input_np, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col shape: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # Handle grouped convolution
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # Reshape weights
        weight_col = weight_np.reshape(out_channels, -1)  # Shape: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # Initialize output
        output = np.zeros((batch_size * out_height * out_width, out_channels))
        
        for g in range(self.groups):
            # Indices for input
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # Indices for output channels
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # Shape: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            
            # Perform matrix multiplication
            # output_group = self.multiply(input_group[:, :, np.newaxis], weight_group.T[np.newaxis, :, :]).sum(axis=1)
            output_group = self.multiply(input_group, weight_group.T)

            # Store the output
            output[:, start_out:end_out] = output_group
        
        # Reshape output
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.transpose(0, 3, 1, 2)  # Shape: (batch_size, out_channels, out_height, out_width)
        
        # Add bias if needed
        if bias is not None:
            bias_np = bias.detach().cpu().numpy()
            output += bias_np.reshape(1, -1, 1, 1)
        
        # Convert back to torch tensor
        return torch.from_numpy(output).to(x.device).type(x.dtype)


class QCustomLinearNumPy(QuantizationHijacker, nn.Linear):
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return np.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        
        input_np = x.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()

        # batch_size, in_features = input_np.shape
        # out_features = self.weight.shape[0]

        # output = np.zeros((batch_size, out_features))
        
        # for b in range(batch_size):
        #     for c_out in range(out_features):
        #         output[b, c_out] = np.dot(input_np[b], weight_np[c_out])  # replace this dot operation with approximate dot operation after

        # output = np.matmul(input_np, weight_np.T)
        output = self.multiply(input_np, weight_np.T)   

        if bias is not None:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np
        
        return torch.from_numpy(output).to(x.device).type(x.dtype)


'''
CustomConv2dCuPy with Quantization
'''
class QCustomConv2dCuPy(QuantizationHijacker, nn.Conv2d):    
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = cp.pad(input_data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        
        # 初始化输出数组
        col = cp.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width))
        
        # 填充输出数组
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑数组以匹配矩阵乘法的要求
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        
        return col
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return cp.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # Convert input to NumPy arrays
        input_cp = cp.asarray(x.detach())   
        weight_cp = cp.asarray(weight.detach())
        # input_cp = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        # weight_cp = cp.ascontiguousarray(cp.from_dlpack(weight.detach()))
        
        batch_size, in_channels, in_height, in_width = input_cp.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_cp.shape
        
        # Compute output dimensions
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # Reshape input using im2col
        input_col = self.im2col(input_cp, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col shape: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # Handle grouped convolution
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # Reshape weights
        weight_col = weight_cp.reshape(out_channels, -1)  # Shape: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # Initialize output
        output = cp.zeros((batch_size * out_height * out_width, out_channels))
        
        for g in range(self.groups):
            # Indices for input
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # Indices for output channels
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # Shape: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            
            # Perform matrix multiplication
            # output_group = self.multiply(input_group[:, :, np.newaxis], weight_group.T[np.newaxis, :, :]).sum(axis=1)
            output_group = self.multiply(input_group, weight_group.T)
            
            # Store the output
            output[:, start_out:end_out] = output_group
        
        # Reshape output
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.transpose(0, 3, 1, 2)  # Shape: (batch_size, out_channels, out_height, out_width)
        
        # Add bias if needed
        if bias is not None:
            bias_cp = cp.asarray(bias)
            output += bias_cp.reshape(1, -1, 1, 1)
        
        # Convert back to torch tensor
        return torch.as_tensor(output, device=x.device, dtype=x.dtype)
        # return torch.from_dlpack(output)

        
class QCustomBNConv2dCuPy(BNFusedHijacker, nn.Conv2d):
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = cp.pad(input_data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        
        # 初始化输出数组
        col = cp.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width))
        
        # 填充输出数组
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑数组以匹配矩阵乘法的要求
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        
        return col
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return cp.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # Convert input to NumPy arrays
        input_cp = cp.asarray(x.detach())   
        weight_cp = cp.asarray(weight.detach())
        # input_cp = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        # weight_cp = cp.ascontiguousarray(cp.from_dlpack(weight.detach()))
        
        batch_size, in_channels, in_height, in_width = input_cp.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_cp.shape
        
        # Compute output dimensions
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # Reshape input using im2col
        input_col = self.im2col(input_cp, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col shape: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # Handle grouped convolution
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # Reshape weights
        weight_col = weight_cp.reshape(out_channels, -1)  # Shape: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # Initialize output
        output = cp.zeros((batch_size * out_height * out_width, out_channels))
        
        for g in range(self.groups):
            # Indices for input
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # Indices for output channels
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # Shape: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            
            # Perform matrix multiplication
            # output_group = self.multiply(input_group[:, :, np.newaxis], weight_group.T[np.newaxis, :, :]).sum(axis=1)
            output_group = self.multiply(input_group, weight_group.T)
            
            # Store the output
            output[:, start_out:end_out] = output_group
        
        # Reshape output
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.transpose(0, 3, 1, 2)  # Shape: (batch_size, out_channels, out_height, out_width)
        
        # Add bias if needed
        if bias is not None:
            bias_cp = cp.asarray(bias)
            output += bias_cp.reshape(1, -1, 1, 1)
        
        # Convert back to torch tensor
        # print(torch.as_tensor(output, device=x.device, dtype=x.dtype))
        return torch.as_tensor(output, device=x.device, dtype=x.dtype)
        # return torch.from_dlpack(output)
    
class QCustomLinearCuPy(QuantizationHijacker, nn.Linear):
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return cp.matmul(x, y)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        input_cp = cp.asarray(x.detach())   
        weight_cp = cp.asarray(weight.detach())
        # input_cp = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        # weight_cp = cp.ascontiguousarray(cp.from_dlpack(weight.detach()))

        output = self.multiply(input_cp, weight_cp.T)   

        if bias is not None:
            bias_cp = cp.asarray(bias)
            output += bias_cp
        
        return torch.as_tensor(output, device=x.device, dtype=x.dtype)
        # return torch.from_dlpack(output)


'''
CustomConv2dTorch with Quantization
'''
# class QCustomTorchApprox():
#     def __init__(self):
#         super().__init__()
        
#     def get_approx_params(self):
#         self.approx_params = {
#             'expo_width'          : 3    ,    # The width of Exponent
#             'mant_width'          : 4    ,    # The width of Mantissa
#             'dnsmp_factor'        : 3    ,    # [3, 4, 5] Down-Sample-Compensation factor
#             'withComp'            : True ,    # [False, True] (def. False) With Compensation opened or not
#             'with_approx'         : True ,
#             'with_s2nn2s_opt'     : True ,
#             'sim_hw_add_OFUF'     : False,    # [False, True] (def. False) 
#             'with_OF_opt'         : False,    # [False, True] (def. False) 
#             'with_UF_opt'         : False,    # [False, True] (def. False) 
#             'golden_clip_OF'      : False,    # [False, True] (def. False)  
#             'quant_btw_mult_accu' : True ,    # [False, True] (def. True ) Quant after both Mult & Accu
#             'debug_mode'          : False,    # [False, True] (def. False) Print some Tensors for debug
#             'self_check_mode'     : False     # [False, True] (def. False) Open this together with dnsmp_factor >= mant_width
#         }

#         return self.approx_params


class QCustomConv2dTorch(QuantizationHijacker, nn.Conv2d):
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = F.pad(input_data, (padding[1], padding[1], padding[0], padding[0]), mode='constant')
        
        # 初始化输出张量
        col = torch.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width), device=input_data.device)
        
        # 填充输出张量
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑张量以匹配矩阵乘法的要求
        col = col.permute(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        
        return col
    # def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
    #     # 使用 F.unfold 实现 im2col
    #     # 输入形状: (batch_size, channels, height, width)
    #     # unfold 后的形状: (batch_size, channels * kernel_height * kernel_width, L)
    #     # 其中 L = 输出的高度 * 输出的宽度
    #     input_unfold = F.unfold(input_data, 
    #                             kernel_size=(kernel_height, kernel_width),
    #                             dilation=dilation,
    #                             padding=padding,
    #                             stride=stride)
    #     return input_unfold  # 形状: (batch_size, channels * kernel_height * kernel_width, L)
    
    def approx_multiply(self, x, y, x_bias, y_bias, res_bias):
        # self.approx_params = self.get_approx_params()
        # print(f"self.approx_params {self.approx_params}")
        expo_width = self.custom_approx_params['expo_width']
        mant_width = self.custom_approx_params['mant_width']
        dnsmp_factor = self.custom_approx_params['dnsmp_factor']
        with_approx = self.custom_approx_params['with_approx']
        with_s2nn2s_opt = self.custom_approx_params['with_s2nn2s_opt']
        sim_hw_add_OFUF = self.custom_approx_params['sim_hw_add_OFUF']
        with_OF_opt = self.custom_approx_params['with_OF_opt']
        with_UF_opt = self.custom_approx_params['with_UF_opt']
        golden_clip_OF = self.custom_approx_params['golden_clip_OF']
        quant_btw_mult_accu = self.custom_approx_params['quant_btw_mult_accu']
        debug_mode = self.custom_approx_params['debug_mode']
        self_check_mode = self.custom_approx_params['self_check_mode']
        withComp = self.custom_approx_params['withComp']

        x_bias = torch.tensor(2**(expo_width-1)) if x_bias is None else x_bias
        res_bias = torch.tensor(2**(expo_width-1)) if res_bias is None else res_bias
        x_bias = x_bias.to(torch.int32)
        y_bias = y_bias.to(torch.int32)
        res_bias = res_bias.to(torch.int32)
        
        comp_table_NN = get_error_table_NN(expo_width, mant_width, withComp=withComp, dnsmp_factor=dnsmp_factor)
        
        if y.shape[1] != 1:
            results = []
            for i in range(y.shape[1]):  # y.shape[1] = out_channels
                # print(f"x.shape: {x.shape}, y.shape: {y[:, i].unsqueeze(1).shape}")
                if self.approx_flag:
                    result = custom_matmul_vectorize(x, y[:, i].unsqueeze(1), expo_width, mant_width,
                                                    x_bias.item(), y_bias[i].item(), res_bias.item(), 
                                                    comp_table_NN,
                                                    with_approx=with_approx,
                                                    with_s2nn2s_opt=with_s2nn2s_opt,
                                                    sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                                    quant_btw_mult_accu=quant_btw_mult_accu,
                                                    debug_mode=debug_mode, self_check_mode=self_check_mode)
                elif self.quantize_after_mult_and_add:
                    result3d = x.unsqueeze(2) * y[:, i].unsqueeze(1).unsqueeze(0)
                    # result3d_quantized = self.res_quantizer(result3d)
                    result3d_quantized, _ = quantize_to_fp8_ste_MM(result3d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result3d_quantized = result3d
                    result2d = result3d_quantized.sum(dim=1)
                    # result = self.res_quantizer(result2d)
                    result, _ = quantize_to_fp8_ste_MM(result2d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result = result2d
                else:
                    result = x @ y[:, i].unsqueeze(1)
                results.append(result)
            output = torch.cat(results, dim=1)
        else:
            if self.approx_flag:
                output = custom_matmul_vectorize(x, y, expo_width, mant_width,
                                x_bias, y_bias, res_bias,
                                comp_table_NN,
                                with_approx=with_approx,
                                with_s2nn2s_opt=with_s2nn2s_opt,
                                sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                quant_btw_mult_accu=quant_btw_mult_accu,
                                debug_mode=debug_mode, self_check_mode=self_check_mode)
            else:
                output = x @ y
        
        torch.cuda.empty_cache()
        return output
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return torch.matmul(x, y)

    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()  # 确保输入张量是连续的
        weight = weight.contiguous()
        # 保持输入为PyTorch张量，x.detach()可以防止梯度回传
        input_torch = x.detach()   
        weight_torch = weight.detach()
        weight_fp_bias = self.get_weights_fp_bias() 
        act_fp_bias = self.get_acts_fp_bias()
        res_fp_bias = self.get_res_fp_bias()

        batch_size, in_channels, in_height, in_width = input_torch.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_torch.shape
        
        # 计算输出的尺寸
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # 使用 im2col 重塑输入
        input_col = self.im2col(input_torch, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col 形状: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # 处理分组卷积
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # 重塑权重
        weight_col = weight_torch.reshape(out_channels, -1)  # 形状: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # 初始化输出张量
        output = torch.zeros((batch_size * out_height * out_width, out_channels), device=x.device, dtype=x.dtype)
        
        for g in range(self.groups):
            # 输入的索引
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # 输出通道的索引
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # 形状: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            weight_group_fp_bias = weight_fp_bias[start_out:end_out].squeeze()
            
            # 执行矩阵乘法
            output_group = self.multiply(input_group, weight_group.T)
            # output_group = self.approx_multiply(input_group, weight_group.T, act_fp_bias, weight_group_fp_bias, res_fp_bias)
            
            # 存储输出
            output[:, start_out:end_out] = output_group
        
        # 重塑输出
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.permute(0, 3, 1, 2)  # 形状: (batch_size, out_channels, out_height, out_width)
        
        # 如果有 bias，添加 bias
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        
        # 返回 PyTorch 张量
        return output



class QCustomBNConv2dTorch(BNFusedHijacker, nn.Conv2d):    
    def im2col(self, input_data, kernel_height, kernel_width, stride, padding, dilation):
        batch_size, channels, height, width = input_data.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
        
        # 填充输入
        padded_input = F.pad(input_data, (padding[1], padding[1], padding[0], padding[0]), mode='constant')
        
        # 初始化输出张量
        col = torch.zeros((batch_size, channels, kernel_height, kernel_width, out_height, out_width), device=input_data.device)
        
        # 填充输出张量
        for y in range(kernel_height):
            y_max = y * dilation[0] + out_height * stride[0]
            for x in range(kernel_width):
                x_max = x * dilation[1] + out_width * stride[1]
                col[:, :, y, x, :, :] = padded_input[:, :, y*dilation[0]:y_max:stride[0], x*dilation[1]:x_max:stride[1]]
        
        # 重塑张量以匹配矩阵乘法的要求
        col = col.permute(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
        
        return col
    
    def approx_multiply(self, x, y, x_bias, y_bias, res_bias):
        # self.approx_params = self.get_approx_params()
        # print(f"self.approx_params {self.approx_params}")
        expo_width = self.custom_approx_params['expo_width']
        mant_width = self.custom_approx_params['mant_width']
        dnsmp_factor = self.custom_approx_params['dnsmp_factor']
        with_approx = self.custom_approx_params['with_approx']
        with_s2nn2s_opt = self.custom_approx_params['with_s2nn2s_opt']
        sim_hw_add_OFUF = self.custom_approx_params['sim_hw_add_OFUF']
        with_OF_opt = self.custom_approx_params['with_OF_opt']
        with_UF_opt = self.custom_approx_params['with_UF_opt']
        golden_clip_OF = self.custom_approx_params['golden_clip_OF']
        quant_btw_mult_accu = self.custom_approx_params['quant_btw_mult_accu']
        debug_mode = self.custom_approx_params['debug_mode']
        self_check_mode = self.custom_approx_params['self_check_mode']
        withComp = self.custom_approx_params['withComp']

        x_bias = torch.tensor(2**(expo_width-1)) if x_bias is None else x_bias
        res_bias = torch.tensor(2**(expo_width-1)) if res_bias is None else res_bias
        x_bias = x_bias.to(torch.int32)
        y_bias = y_bias.to(torch.int32)
        res_bias = res_bias.to(torch.int32)
        
        comp_table_NN = get_error_table_NN(expo_width, mant_width, withComp=withComp, dnsmp_factor=dnsmp_factor)
        
        if y.shape[1] != 1:
            results = []
            for i in range(y.shape[1]):  # y.shape[1] = out_channels
                # print(f"x.shape: {x.shape}, y.shape: {y[:, i].unsqueeze(1).shape}")
                if self.approx_flag:
                    result = custom_matmul_vectorize(x, y[:, i].unsqueeze(1), expo_width, mant_width,
                                                    x_bias.item(), y_bias[i].item(), res_bias.item(), 
                                                    comp_table_NN,
                                                    with_approx=with_approx,
                                                    with_s2nn2s_opt=with_s2nn2s_opt,
                                                    sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                                    quant_btw_mult_accu=quant_btw_mult_accu,
                                                    debug_mode=debug_mode, self_check_mode=self_check_mode)
                elif self.quantize_after_mult_and_add:
                    result3d = x.unsqueeze(2) * y[:, i].unsqueeze(1).unsqueeze(0)
                    # result3d_quantized = self.res_quantizer(result3d)
                    result3d_quantized, _ = quantize_to_fp8_ste_MM(result3d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result3d_quantized = result3d
                    result2d = result3d_quantized.sum(dim=1)
                    # result = self.res_quantizer(result2d)
                    result, _ = quantize_to_fp8_ste_MM(result2d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result = result2d
                else:
                    result = x @ y[:, i].unsqueeze(1)
                results.append(result)
            output = torch.cat(results, dim=1)
        else:
            if self.approx_flag:
                output = custom_matmul_vectorize(x, y, expo_width, mant_width,
                                x_bias, y_bias, res_bias,
                                comp_table_NN,
                                with_approx=with_approx,
                                with_s2nn2s_opt=with_s2nn2s_opt,
                                sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                quant_btw_mult_accu=quant_btw_mult_accu,
                                debug_mode=debug_mode, self_check_mode=self_check_mode)
            else:
                output = x @ y
        
        torch.cuda.empty_cache()
        return output
    
    def multiply(self, x, y):
        output = torch.matmul(x, y)

        torch.cuda.empty_cache()
        return output
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()  # 确保输入张量是连续的
        weight = weight.contiguous() #
        # print(f"weight: {weight}, weight.shape: {weight.shape}") 
        # if self.groups == 1:
        weight_fp_bias = self.get_weights_fp_bias() 
        act_fp_bias = self.get_acts_fp_bias()
        res_fp_bias = self.get_res_fp_bias()
        # print(f"weight_fp_bias: {weight_fp_bias}, weight_fp_bias.shape: {weight_fp_bias.shape}")
        # print(f"act_fp_bias: {act_fp_bias}, act_fp_bias.shape: {act_fp_bias.shape if act_fp_bias is not None else 'None'}")
        # print(f"res_fp_bias: {res_fp_bias}, res_fp_bias.shape: {res_fp_bias.shape if res_fp_bias is not None else 'None'}")
        # print(f"weigt.shape: {weight.shape}, act.shape: {x.shape}")
        # 保持输入为PyTorch张量，x.detach()可以防止梯度回传
        input_torch = x.detach()   
        weight_torch = weight.detach()

        batch_size, in_channels, in_height, in_width = input_torch.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_torch.shape
        
        # 计算输出的尺寸
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
        # 使用 im2col 重塑输入
        input_col = self.im2col(input_torch, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # input_col 形状: (batch_size * out_height * out_width, in_channels * kernel_height * kernel_width)
        
        # 处理分组卷积
        group_in_channels = in_channels // self.groups
        group_out_channels = out_channels // self.groups
        
        # 重塑权重
        weight_col = weight_torch.reshape(out_channels, -1)  # 形状: (out_channels, in_channels_per_group * kernel_height * kernel_width)
        
        # 初始化输出张量
        output = torch.zeros((batch_size * out_height * out_width, out_channels), device=x.device, dtype=x.dtype)
        for g in range(self.groups):
            # 输入的索引
            start_in = g * group_in_channels * kernel_height * kernel_width
            end_in = (g + 1) * group_in_channels * kernel_height * kernel_width
            input_group = input_col[:, start_in:end_in]
            
            # 输出通道的索引
            start_out = g * group_out_channels
            end_out = (g + 1) * group_out_channels
            weight_group = weight_col[start_out:end_out, :]  # 形状: (group_out_channels, in_channels_per_group * kernel_height * kernel_width)
            weight_group_fp_bias = weight_fp_bias[start_out:end_out].squeeze()
            
            
            # 执行矩阵乘法
            # output_group = self.multiply(input_group, weight_group.T)
            output_group = self.approx_multiply(input_group, weight_group.T, act_fp_bias, weight_group_fp_bias, res_fp_bias)
            # if self.groups == 1:
            # print(f"weight_fp_bias: {weight_fp_bias}, weight_fp_bias.shape: {weight_fp_bias.shape}")
            # print(f"act_fp_bias: {act_fp_bias}, act_fp_bias.shape: {act_fp_bias.shape if act_fp_bias is not None else 'None'}")
            # print(f"input_col.shape: {input_col.shape}")
            # print(f"weight_col.shape: {weight_col.shape}")
            # print(f"self.groups = {self.groups}, group = {g}")
            # print(f"input_group.shape: {input_group.shape}, weight_group.shape: {weight_group.shape} \n")
            # if self.groups == 1 and act_fp_bias is not None:  # input_col is the same with input_group when groups == 1, so as weight_col and weight_group
            #     print(f"input_group: {input_group}, input_group.shape: {input_group.shape}")
            #     print(f"weight_group: {weight_group}, weight_group.shape: {weight_group.shape}")
            #     print(f"output_group: {output_group}, output_group.shape: {output_group.shape}")
                
            #     if input_group.shape[0] + input_group.shape[1] < 1000:
            #         weight_np = weight_group.T.cpu().numpy()
            #         weight_bias_np = weight_fp_bias.squeeze().cpu().numpy()
            #         act_np = input_group.cpu().numpy()
            #         act_bias_np = act_fp_bias.cpu().numpy()
                    
            #         weight_df = pd.DataFrame(weight_np)
            #         weight_bias_df = pd.DataFrame(weight_bias_np)
            #         act_df = pd.DataFrame(act_np)
            #         act_bias_df = pd.DataFrame(act_bias_np)
                    
            #         weight_df.to_csv('/home/zou/codes/FP8-quantization/debug_params/weight.csv', index=False, header=False)
            #         weight_bias_df.to_csv('/home/zou/codes/FP8-quantization/debug_params/weight_bias.csv', index=False, header=False)
            #         act_df.to_csv('/home/zou/codes/FP8-quantization/debug_params/act.csv', index=False, header=False)
            #         act_bias_df.to_csv('/home/zou/codes/FP8-quantization/debug_params/act_bias.csv', index=False, header=False)
                    
            #         raise ValueError("Stop here")
                
            
            # 存储输出
            output[:, start_out:end_out] = output_group
        
        # 重塑输出
        output = output.reshape(batch_size, out_height, out_width, out_channels)
        output = output.permute(0, 3, 1, 2)  # 形状: (batch_size, out_channels, out_height, out_width)
        
        # 如果有 bias，添加 bias
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        
        # 返回 PyTorch 张量
        return output
    
    
class QCustomLinearTorch(QuantizationHijacker, nn.Linear):      
    def approx_multiply(self, x, y, x_bias, y_bias, res_bias):
        # self.approx_params = self.get_approx_params()
        # print(f"self.approx_params {self.approx_params}")
        expo_width = self.custom_approx_params['expo_width']
        mant_width = self.custom_approx_params['mant_width']
        dnsmp_factor = self.custom_approx_params['dnsmp_factor']
        with_approx = self.custom_approx_params['with_approx']
        with_s2nn2s_opt = self.custom_approx_params['with_s2nn2s_opt']
        sim_hw_add_OFUF = self.custom_approx_params['sim_hw_add_OFUF']
        with_OF_opt = self.custom_approx_params['with_OF_opt']
        with_UF_opt = self.custom_approx_params['with_UF_opt']
        golden_clip_OF = self.custom_approx_params['golden_clip_OF']
        quant_btw_mult_accu = self.custom_approx_params['quant_btw_mult_accu']
        debug_mode = self.custom_approx_params['debug_mode']
        self_check_mode = self.custom_approx_params['self_check_mode']
        withComp = self.custom_approx_params['withComp']

        x_bias = torch.tensor(2**(expo_width-1)) if x_bias is None else x_bias
        res_bias = torch.tensor(2**(expo_width-1)) if res_bias is None else res_bias
        x_bias = x_bias.to(torch.int32)
        y_bias = y_bias.to(torch.int32)
        res_bias = res_bias.to(torch.int32)
        
        comp_table_NN = get_error_table_NN(expo_width, mant_width, withComp=withComp, dnsmp_factor=dnsmp_factor)
        
        if y.shape[1] != 1:
            results = []
            for i in range(y.shape[1]):  # y.shape[1] = out_channels
                # print(f"x.shape: {x.shape}, y.shape: {y[:, i].unsqueeze(1).shape}")
                if self.approx_flag:
                    result = custom_matmul_vectorize(x, y[:, i].unsqueeze(1), expo_width, mant_width,
                                                    x_bias.item(), y_bias[i].item(), res_bias.item(), 
                                                    comp_table_NN,
                                                    with_approx=with_approx,
                                                    with_s2nn2s_opt=with_s2nn2s_opt,
                                                    sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                                    quant_btw_mult_accu=quant_btw_mult_accu,
                                                    debug_mode=debug_mode, self_check_mode=self_check_mode)
                elif self.quantize_after_mult_and_add:
                    result3d = x.unsqueeze(2) * y[:, i].unsqueeze(1).unsqueeze(0)
                    # result3d_quantized = self.res_quantizer(result3d)
                    result3d_quantized, _ = quantize_to_fp8_ste_MM(result3d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result3d_quantized = result3d
                    result2d = result3d_quantized.sum(dim=1)
                    # result = self.res_quantizer(result2d)
                    result, _ = quantize_to_fp8_ste_MM(result2d, self.res_quantizer.quantizer.n_bits, self.res_quantizer.quantizer.maxval, self.res_quantizer.quantizer.mantissa_bits, self.res_quantizer.quantizer.sign_bits)
                    # result = result2d
                else:
                    result = x @ y[:, i].unsqueeze(1)
                results.append(result)
            output = torch.cat(results, dim=1)
            if self.approx_flag:
                print(f"approx output: {output}\napprox output.shape: {output.shape}")
                print(f"bias: {res_bias}\nbias.shape: {res_bias.shape}")
                print(f"expo_width: {expo_width}, mant_width: {mant_width}, dnsmp_factor: {dnsmp_factor}, withComp: {withComp}\n"+
                  f"with_approx: {with_approx}, with_s2nn2s_opt: {with_s2nn2s_opt}, sim_hw_add_OFUF: {sim_hw_add_OFUF}\n"+
                  f"with_OF_opt: {with_OF_opt}, with_UF_opt: {with_UF_opt}, golden_clip_OF: {golden_clip_OF}\n"+
                  f"quant_btw_mult_accu: {quant_btw_mult_accu}, debug_mode: {debug_mode}, self_check_mode: {self_check_mode}")
            elif self.quantize_after_mult_and_add:
                print(f"qamaa output: {output}\nqamaa output.shape: {output.shape}")
                print(f"qamaa bias: {res_bias}\nqamaa bias.shape: {res_bias.shape}")
            else:
                print(f"output: {output}\noutput.shape: {output.shape}")
                print(f"bias: {res_bias}\nbias.shape: {res_bias.shape}")
        else:
            if self.approx_flag:
                output = custom_matmul_vectorize(x, y, expo_width, mant_width,
                                x_bias, y_bias, res_bias,
                                comp_table_NN,
                                with_approx=with_approx,
                                with_s2nn2s_opt=with_s2nn2s_opt,
                                sim_hw_add_OFUF=sim_hw_add_OFUF, with_OF_opt=with_OF_opt, with_UF_opt=with_UF_opt, golden_clip_OF=golden_clip_OF,
                                quant_btw_mult_accu=quant_btw_mult_accu,
                                debug_mode=debug_mode, self_check_mode=self_check_mode)
            else:
                output = x @ y
        
        torch.cuda.empty_cache()
        return output
                
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        return torch.matmul(x, y)

    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        weight = weight.contiguous()
        weight_fp_bias = self.get_weights_fp_bias() 
        act_fp_bias = self.get_acts_fp_bias()
        res_fp_bias = self.get_res_fp_bias()
        # print(f"weight_fp_bias: {weight_fp_bias}, weight_fp_bias.shape: {weight_fp_bias.shape}")
        # print(f"act_fp_bias: {act_fp_bias}, act_fp_bias.shape: {act_fp_bias.shape if act_fp_bias is not None else 'None'}")
        # print(f"weigt.shape: {weight.shape}, act.shape: {x.shape}")
        # print(f"x.shape: {x.shape}, weight.shape: {weight.shape}\n")
        # output = self.multiply(x, weight.t())   
        output = self.approx_multiply(x, weight.t(), act_fp_bias, weight_fp_bias, res_fp_bias)

        if bias is not None:
            output += bias
        
        return output
    
