import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

from quantization.autoquant_utils import QuantConv, QuantLinear, BNQConv

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
        

class QCustomConv2dNumPy(QuantConv):
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
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # 将输入转换为NumPy数组
        input_np = x.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        
        batch_size, in_channels, in_height, in_width = input_np.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_np.shape
        
        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
    #     # 初始化输出
    #     output = np.zeros((batch_size, out_channels, out_height, out_width))
        
    #     # 对输入进行填充
    #     if self.padding[0] > 0 or self.padding[1] > 0:
    #         input_np = np.pad(input_np, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        
    #    # 执行分组卷积
    #     for b in range(batch_size):
    #         for g in range(self.groups):
    #             for c_out in range(out_channels // self.groups):
    #                 c_out_idx = g * (out_channels // self.groups) + c_out
    #                 for h_out in range(out_height):
    #                     for w_out in range(out_width):
    #                         h_start = h_out * self.stride[0]
    #                         w_start = w_out * self.stride[1]
    #                         h_end = h_start + kernel_height * self.dilation[0]
    #                         w_end = w_start + kernel_width * self.dilation[1]
                            
    #                         # 提取当前位置的输入块
    #                         input_slice = input_np[b, 
    #                                                g * in_channels_per_group:(g + 1) * in_channels_per_group, 
    #                                                h_start:h_end:self.dilation[0], 
    #                                                w_start:w_end:self.dilation[1]]
                            
    #                         # 执行卷积操作
    #                         output[b, c_out_idx, h_out, w_out] = np.sum(input_slice * weight_np[c_out_idx])
        # 使用im2col重塑输入
        input_col = self.im2col(input_np, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # 处理分组卷积
        group_size = in_channels // self.groups
        input_col = input_col.reshape(batch_size, out_height, out_width, self.groups, group_size, kernel_height * kernel_width)
        input_col = input_col.transpose(3, 0, 4, 5, 1, 2).reshape(self.groups, -1, out_height * out_width)
        # 重塑权重
        weight_col = weight_np.reshape(out_channels, -1)
        # 执行分组矩阵乘法
        output = np.zeros((self.groups, out_channels // self.groups, batch_size * out_height * out_width))
        for g in range(self.groups):
            output[g] = np.dot(weight_col[g], input_col[g])
        # 重塑输出
        output = output.reshape(out_channels, batch_size, out_height, out_width)
        output = output.transpose(1, 0, 2, 3)
        
        # 添加偏置（如果有）
        if bias is not None:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np.reshape(1, -1, 1, 1)
            
        
        # 将结果转回PyTorch张量
        return torch.from_numpy(output).to(x.device).type(x.dtype)


class QCustomBNConv2dNumPy(BNQConv):
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
    
    def run_forward(self, x, weight, bias, offsets=None):
        x = x.contiguous()
        # 将输入转换为NumPy数组
        input_np = x.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        
        batch_size, in_channels, in_height, in_width = input_np.shape
        out_channels, in_channels_per_group, kernel_height, kernel_width = weight_np.shape
        
        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (kernel_height - 1) - 1) // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (kernel_width - 1) - 1) // self.stride[1] + 1
        
    #     # 初始化输出
    #     output = np.zeros((batch_size, out_channels, out_height, out_width))
        
    #     # 对输入进行填充
    #     if self.padding[0] > 0 or self.padding[1] > 0:
    #         input_np = np.pad(input_np, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        
    #    # 执行分组卷积
    #     for b in range(batch_size):
    #         for g in range(self.groups):
    #             for c_out in range(out_channels // self.groups):
    #                 c_out_idx = g * (out_channels // self.groups) + c_out
    #                 for h_out in range(out_height):
    #                     for w_out in range(out_width):
    #                         h_start = h_out * self.stride[0]
    #                         w_start = w_out * self.stride[1]
    #                         h_end = h_start + kernel_height * self.dilation[0]
    #                         w_end = w_start + kernel_width * self.dilation[1]
                            
    #                         # 提取当前位置的输入块
    #                         input_slice = input_np[b, 
    #                                                g * in_channels_per_group:(g + 1) * in_channels_per_group, 
    #                                                h_start:h_end:self.dilation[0], 
    #                                                w_start:w_end:self.dilation[1]]
                            
    #                         # 执行卷积操作
    #                         output[b, c_out_idx, h_out, w_out] = np.sum(input_slice * weight_np[c_out_idx])
        # 使用im2col重塑输入
        input_col = self.im2col(input_np, kernel_height, kernel_width, self.stride, self.padding, self.dilation)
        # 处理分组卷积
        group_size = in_channels // self.groups
        input_col = input_col.reshape(batch_size, out_height, out_width, self.groups, group_size, kernel_height * kernel_width)
        input_col = input_col.transpose(3, 0, 4, 5, 1, 2).reshape(self.groups, -1, out_height * out_width)
        # 重塑权重
        weight_col = weight_np.reshape(out_channels, -1)
        # 执行分组矩阵乘法
        output = np.zeros((self.groups, out_channels // self.groups, batch_size * out_height * out_width))
        for g in range(self.groups):
            output[g] = np.dot(weight_col[g], input_col[g])
        # 重塑输出
        output = output.reshape(out_channels, batch_size, out_height, out_width)
        output = output.transpose(1, 0, 2, 3)
        
        # 添加偏置（如果有）
        if bias is not None:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np.reshape(1, -1, 1, 1)
            
        
        # 将结果转回PyTorch张量
        return torch.from_numpy(output).to(x.device).type(x.dtype)


class QCustomLinearNumPy(QuantLinear):
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

        output = np.matmul(input_np, weight_np.T)

        if bias is not None:
            bias_np = self.bias.detach().cpu().numpy()
            output += bias_np
        
        return torch.from_numpy(output).to(x.device).type(x.dtype)



