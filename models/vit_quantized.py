import os
import re
import torch
from collections import OrderedDict
from quantization.autoquant_utils import quantize_sequential, Flattener, quantize_model, BNQConv
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def vit_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    fp_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # if pretrained and load_type == "fp32":
    return fp_model