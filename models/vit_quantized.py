import os
import re
import torch
from collections import OrderedDict
from quantization.autoquant_utils import quantize_sequential, Flattener, quantize_model, BNQConv
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel

from transformers.models.vit.modeling_vit import *
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

from typing import Dict, List, Optional, Set, Tuple, Union


class VisionTransformerForImageClassification(ViTForImageClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        return logits

class QuantizedVitEmbeddings(QuantizedModel):
    def __init__(self, vit_emb_orig, **quant_params):
        super().__init__(**quant_params)
        specials = {}

class QuantizedViTModel(QuantizedModel):
    def __init__(self, vit_mod_orig, **quant_params):
        super().__init__(**quant_params)
        specials = {ViTEmbeddings, ViTEncoder, ViTPooler}

class QuantizedVisionTransformerForImageClassification(QuantizedModel):
    def __init__(self, model_fp, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)
        specials = {ViTModel: QuantizedViTModel}
        # quantize and copy parts from original model
        quantize_input = quant_setup and quant_setup == "LSQ_paper"
        self.vit = quantize_sequential(
            model_fp.vit, 
            tie_activation_quantizers=not quantize_input,
            specials=specials, 
            **quant_params,
            )
        
        self.classifier = quantize_model(model_fp.classifier, **quant_params)
        
        
    def forward(self, x):
        outputs = self.vit(x)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        
        return logits


def vit_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    fp_model = VisionTransformerForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # if pretrained and load_type == "fp32":
    return fp_model