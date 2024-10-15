import torch.nn as nn
from approx.approx_calculation import *
from models.mobilenet_v2 import InvertedResidual, MobileNetV2
from models.mobilenet_v2_quantized import QuantizedInvertedResidual, BNQConv
from quantization.base_quantized_classes import QuantizedActivation

from quantization.autoquant_utils import QuantConv, QuantLinear

def replace_conv2d_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, CustomConv2dNumPy(child.in_channels, child.out_channels, 
                                                    child.kernel_size, stride=child.stride,
                                                    padding=child.padding, dilation=child.dilation,
                                                    groups=child.groups, bias=child.bias is not None))
        elif isinstance(child, nn.Sequential) or isinstance(child, InvertedResidual):
            replace_conv2d_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_numpy(submodule)
                
def replace_linear_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, CustomLinearNumPy(child.in_features, child.out_features, bias=child.bias is not None))
        elif isinstance(child, nn.Sequential) or isinstance(child, InvertedResidual):
            replace_linear_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_numpy(submodule)

def replace_operations_in_mobilenet_v2(model):
    replace_conv2d_with_numpy(model)
    replace_linear_with_numpy(model)


'''
replace the conv2d and linear operations in the mobilenet_v2_quantized model with numpy operations
'''
def replace_conv2d_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantConv):
            new_conv = QCustomConv2dNumPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            setattr(module, name, new_conv)
        elif isinstance(child, BNQConv):
            new_conv = QCustomBNConv2dNumPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            setattr(module, name, new_conv)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_conv2d_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_numpy(submodule)

def replace_linear_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantLinear):
            new_linear = QCustomLinearNumPy(child.in_features, child.out_features, bias=child.bias is not None)
            # 复制权重和偏置
            new_linear.weight.data = child.weight.data
            if child.bias is not None:
                new_linear.bias.data = child.bias.data
            setattr(module, name, new_linear)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_linear_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_numpy(submodule)

def replace_conv2d_with_cupy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantConv):
            new_conv = QCustomConv2dCuPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            new_comv = new_conv.to(child.weight.device)
            setattr(module, name, new_conv)
        elif isinstance(child, BNQConv):
            new_conv = QCustomConv2dCuPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            new_comv = new_conv.to(child.weight.device)
            setattr(module, name, new_conv)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_conv2d_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_numpy(submodule)

def replace_linear_with_cupy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantLinear):
            new_linear = QCustomLinearCuPy(child.in_features, child.out_features, bias=child.bias is not None)
            # 复制权重和偏置
            new_linear.weight.data = child.weight.data
            if child.bias is not None:
                new_linear.bias.data = child.bias.data
            new_linear = new_linear.to(child.weight.device)
            setattr(module, name, new_linear)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_linear_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_numpy(submodule)

def replace_operations_in_mobilenet_v2_quantized(model):
    # replace_conv2d_with_numpy(model)
    # replace_linear_with_numpy(model)
    replace_conv2d_with_cupy(model)
    replace_linear_with_cupy(model)

# Example usage
if __name__ == "__main__":
    from models.mobilenet_v2_quantized import mobilenetv2_quantized  # Add this import
    model = mobilenetv2_quantized(pretrained=True, model_dir='/home/zou/codes/FP8-quantization/model_dir/mobilenet_v2.pth.tar')
    replace_operations_in_mobilenet_v2_quantized(model)

    print(model)

