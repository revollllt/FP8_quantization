# FP8 Quantization: The Power of the Exponent
This repository contains the implementation and experiments for the paper presented in

**Andrey Kuzmin<sup>\*1</sup>, Mart van Baalen<sup>\*1</sup>,  Yuwei Ren<sup>1</sup>, 
Markus Nagel<sup>1</sup>, Jorn Peters<sup>1</sup>, Tijmen Blankevoort<sup>1</sup> "FP8 Quantization: The Power of the Exponent", NeurIPS 
2022.** [[ArXiv]](https://arxiv.org/abs/2208.09225)

fork from https://github.com/Qualcomm-AI-research/FP8-quantization



## How to install
Make sure to have Python ≥3.8 (tested with Python 3.8.10) and 
ensure the latest version of `pip` (tested with 21.3.1):
```bash
conda create -n fp8_quantization python=3.9
conda activate fp8_quantization
```

Next, install PyTorch 2.4.0 with the appropriate CUDA version (tested with CUDA 11.8):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```


## Running experiments
```bash
sh scripts/image_net.sh
 ```


## 选择approx算子所需要的操作
1. 在scripts/image_net.sh中修改architecture，目前支持的architecture可在/home/xinyuc/jiaxiang/FP8_quantization/models/__init__.py中查看

2. 在/home/xinyuc/jiaxiang/FP8_quantization/quantization/quantized_folded_bn.py和它的父类/home/xinyuc/jiaxiang/FP8_quantization/quantization/hijacker.py中修改BNFusedHijacker和QuantizationHijacker
这两个class中的forward方法。即需要approx则解注释self.approx_flag = True和其底下的self.run_forward，需要quantize_after_mult_and_add则解注释self.quantize_after_mult_and_add = True。这里需要注意的是，如果要
使QCustomLinearTorch等approx_calculation.py中算子中的self.get_res_fp_bias()能够正常工作，BNFusedHijacker和QuantizationHijacker中forward中的res = self.res_quantizer(res)是需要保留的

3. 可以快速调整量化框架的mant_width和expo_width，即在scripts/image_net.sh中修改mant_width。但是目前仍然不能在脚本中快速调整approx算子或者说其中的approx_v8所用的mant_width以及dnsmp_factor等参数，这些参数需要在调整
scripts/image_net.sh中mant_width后在/home/xinyuc/jiaxiang/FP8_quantization/approx/approx_calculation.py中大概526行的QCustomTorchApprox类中手动修改。

## 选择空闲的device
首先在bash界面输入
```bash
nvitop
```
根据空闲或显存使用较少的GPU修改scripts/image_net.sh中的device

## 目前本地进行的工作
主要是优化脚本的自动化操作，比如上述内容3中的approx算子如何放到scripts/image_net.sh脚本中一起修改，然后通过一个新的脚本去进行批量测试。
