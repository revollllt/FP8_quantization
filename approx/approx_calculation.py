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

from approx.approx_matmul_whole_v4 import *

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
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        # return torch.matmul(x, y)
        expo_width = 3
        mant_width = 4
        dnsmp_factor = 3
        comp_table_NN = get_comp_table_NN(expo_width, mant_width, withComp=True, dnsmp_factor=dnsmp_factor, device=x.device)
        sim_hw_add_OFUF = False
        with_OF_opt = False
        with_UF_opt = False
        debug_mode = False

        output = custom_matmul_vectorize(x, y, 
                                       expo_width, 
                                       mant_width, 
                                       comp_table_NN   = comp_table_NN, 
                                       sim_hw_add_OFUF = sim_hw_add_OFUF, 
                                       with_OF_opt     = with_OF_opt, 
                                       with_UF_opt     = with_UF_opt, 
                                       golden_clip_OF  = False,
                                       debug_mode      = debug_mode,
                                       self_check_mode = False)
    
        return output
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()  # 确保输入张量是连续的
        weight = weight.contiguous()
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
            
            # 执行矩阵乘法
            output_group = self.multiply(input_group, weight_group.T)
            
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
    # def run_forward(self, x, weight, bias, offsets=None):
    #     batch_size, in_channels, in_height, in_width = x.shape
    #     out_channels, _, kernel_height, kernel_width = weight.shape
        
    #     # 计算输出尺寸
    #     out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
    #     out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
    #     # 展开输入张量
    #     input_unfold = self.im2col(x, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
    #     # 形状: (batch_size, in_channels * kernel_height * kernel_width, out_height * out_width)
        
    #     # 处理分组卷积
    #     groups = self.groups
    #     group_in_channels = in_channels // groups
    #     group_out_channels = out_channels // groups
        
    #     # 重塑输入张量以适应分组
    #     input_unfold = input_unfold.view(batch_size, groups, group_in_channels * kernel_height * kernel_width, -1)
    #     # 形状: (batch_size, groups, group_in_channels * kernel_height * kernel_width, out_height * out_width)
        
    #     # 重塑权重张量
    #     weight = weight.view(groups, group_out_channels, group_in_channels * kernel_height * kernel_width)
    #     # 形状: (groups, group_out_channels, group_in_channels * kernel_height * kernel_width)
        
    #     # 执行批量矩阵乘法
    #     # 首先交换 input_unfold 的最后两个维度以匹配矩阵乘法要求
    #     input_unfold = input_unfold.permute(0, 1, 3, 2)
    #     # 形状: (batch_size, groups, out_height * out_width, group_in_channels * kernel_height * kernel_width)
        
    #     # 进行矩阵乘法
    #     # output = torch.matmul(input_unfold, weight.transpose(2, 1))
    #     output = self.multiply(input_unfold, weight.transpose(2, 1))
    #     # 形状: (batch_size, groups, out_height * out_width, group_out_channels)
        
    #     # 调整输出形状
    #     output = output.permute(0, 1, 3, 2).contiguous()
    #     # 形状: (batch_size, groups, group_out_channels, out_height * out_width)
        
    #     output = output.view(batch_size, out_channels, out_height, out_width)
    #     # 形状: (batch_size, out_channels, out_height, out_width)
        
    #     # 添加偏置（如果有）
    #     if bias is not None:
    #         output += bias.view(1, -1, 1, 1)
        
    #     return output


class QCustomBNConv2dTorch(BNFusedHijacker, nn.Conv2d):
    # def __init__(self, *args, **kwargs):
    #     super(QCustomBNConv2dTorch, self).__init__(*args, **kwargs)
    #     self.mult_times = 0
    # def __init__(self, *args, **kwargs):
    #     super(QCustomBNConv2dTorch, self).__init__(*args, **kwargs)
    #     self.approx_calculation = ApproxCalculation()
    
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
    
    def multiply(self, x, y):
        # return np.multiply(x, y)
        # return torch.matmul(x, y)
        # matmul_start_time = time.time()
        # print(f"x.dtype: {x.dtype}")
        # print(f"x.shape: {x.shape}")
        # print(f"y.shape: {y.shape}")
        # output1 = torch.matmul(x, y)
        # matmul_end_time = time.time()
        # matmul_time = matmul_end_time - matmul_start_time
        # print(f"matmul_time: {matmul_time}")
        expo_width = 3
        mant_width = 4
        dnsmp_factor = 3
        # comp_start_time = time.time()
        comp_table_NN = get_comp_table_NN(expo_width, mant_width, withComp=True, dnsmp_factor=dnsmp_factor, device=x.device)
        # comp_end_time = time.time()
        # comp_time = comp_end_time - comp_start_time
        sim_hw_add_OFUF = False
        with_OF_opt = False
        with_UF_opt = False
        debug_mode = False

        # custom_start_time = time.time()
        # print(f"y: {y}")
        output = custom_matmul_vectorize(x, y, 
                                       expo_width, 
                                       mant_width, 
                                       comp_table_NN   = comp_table_NN, 
                                       sim_hw_add_OFUF = sim_hw_add_OFUF, 
                                       with_OF_opt     = with_OF_opt, 
                                       with_UF_opt     = with_UF_opt, 
                                       golden_clip_OF  = False,
                                       debug_mode      = debug_mode,
                                       self_check_mode = False)
        # custom_end_time = time.time()
        # custom_time = custom_end_time - custom_start_time
        # print(f"comp_time: {comp_time}")
        # print(f"custom_time: {custom_time}")
        # print(f"output1: {output1}")
        # print(f"output: {output}")
        torch.cuda.empty_cache()
        return output
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()  # 确保输入张量是连续的
        weight = weight.contiguous() #
        # print(f"weight: {weight}, weight.shape: {weight.shape}") 
        # if self.groups == 1:
        # weight_fp_bias = self.get_weights_fp_bias() 
        # act_fp_bias = self.get_acts_fp_bias()
        # print(f"weight_fp_bias: {weight_fp_bias}, weight_fp_bias.shape: {weight_fp_bias.shape}")
        # print(f"act_fp_bias: {act_fp_bias}, act_fp_bias.shape: {act_fp_bias.shape if act_fp_bias is not None else 'None'}")
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
            
            # 执行矩阵乘法
            output_group = self.multiply(input_group, weight_group.T)
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
    def multiply(self, x, y):
        # return np.multiply(x, y)
        # return torch.matmul(x, y)
        expo_width = 3
        mant_width = 4
        dnsmp_factor = 3
        
        comp_table_NN = get_comp_table_NN(expo_width, mant_width, withComp=True, dnsmp_factor=dnsmp_factor, device=x.device)
        sim_hw_add_OFUF = False
        with_OF_opt = False
        with_UF_opt = False
        debug_mode = False

        return custom_matmul_vectorize(x, y, 
                                       expo_width, 
                                       mant_width, 
                                       comp_table_NN   = comp_table_NN, 
                                       sim_hw_add_OFUF = sim_hw_add_OFUF, 
                                       with_OF_opt     = with_OF_opt, 
                                       with_UF_opt     = with_UF_opt, 
                                       golden_clip_OF  = False,
                                       debug_mode      = debug_mode,
                                       self_check_mode = False)
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        weight = weight.contiguous()
        # weight_fp_bias = self.get_weights_fp_bias() 
        # act_fp_bias = self.get_acts_fp_bias()
        # print(f"weight_fp_bias: {weight_fp_bias}, weight_fp_bias.shape: {weight_fp_bias.shape}")
        # print(f"act_fp_bias: {act_fp_bias}, act_fp_bias.shape: {act_fp_bias.shape if act_fp_bias is not None else 'None'}")
        # print(f"weigt.shape: {weight.shape}, act.shape: {x.shape}")
        output = self.multiply(x, weight.t())   

        if bias is not None:
            output += bias
        
        return output
    
