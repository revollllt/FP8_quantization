#!/bin/bash
device=0

seed=10

image_dir="/home/zou/data/ImageNet"
model_dir="/home/zou/codes/FP8-quantization/model_dir/mobilenet_v2.pth.tar"

architecture="mobilenet_v2_quantized_approx"

batch_size=16

n_bits=8

expo_width=3
mant_width=4
dnsmp_factor=3


CUDA_VISIBLE_DEVICES=$device python image_net.py validate-quantized \
    --images-dir ${image_dir} \
    --architecture ${architecture} \
    --batch-size ${batch_size} \
    --seed ${seed} \
    --model-dir ${model_dir} \
    --n-bits ${n_bits}  \
    --cuda \
    --load-type fp32 \
    --quant-setup all \
    --qmethod fp_quantizer \
    --per-channel \
    --fp8-mantissa-bits=$mant_width \
    --fp8-set-maxval \
    --no-fp8-mse-include-mantissa-bits \
    --weight-quant-method=current_minmax \
    --act-quant-method=allminmax \
    --num-est-batches=1 \
    --quantize-input \
    --expo-width ${expo_width} \
    --mant-width ${mant_width} \
    --dnsmp-factor ${dnsmp_factor}