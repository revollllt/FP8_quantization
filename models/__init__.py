#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from models.mobilenet_v2_quantized import mobilenetv2_quantized
from models.resnet_quantized import resnet18_quantized, resnet50_quantized
from models.vit_quantized import vit_quantized
from models.demo_quantized import demo_quantized
from utils import ClassEnumOptions, MethodMap

from models.mobilenet_v2_quantized_approx import mobilenetv2_quantized_approx

class QuantArchitectures(ClassEnumOptions):
    mobilenet_v2_quantized = MethodMap(mobilenetv2_quantized)
    resnet18_quantized = MethodMap(resnet18_quantized)
    resnet50_quantized = MethodMap(resnet50_quantized)
    vit_quantized = MethodMap(vit_quantized)
    demo_quantized = MethodMap(demo_quantized)
    
    '''
    approx architecture
    '''
    mobilenet_v2_quantized_approx = MethodMap(mobilenetv2_quantized_approx)
