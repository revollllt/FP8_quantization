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

1. 在 `scripts/image_net.sh` 中修改 architecture，目前支持的 architecture 可在 `/home/xinyuc/jiaxiang/FP8_quantization/models/__init__.py` 中查看。

2. 在 `/home/xinyuc/jiaxiang/FP8_quantization/quantization/quantized_folded_bn.py` 和它的父类 `/home/xinyuc/jiaxiang/FP8_quantization/quantization/hijacker.py` 中修改 `BNFusedHijacker` 和 `QuantizationHijacker` 这两个 class 中的 `forward` 方法。  
   即需要 approx 则解注释 `self.approx_flag = True` 和其底下的 `self.run_forward`，需要 `quantize_after_mult_and_add` 则解注释 `self.quantize_after_mult_and_add = True`。  
   这里需要注意的是，如果要使 `QCustomLinearTorch` 等 `approx_calculation.py` 中算子中的 `self.get_res_fp_bias()` 能够正常工作，`BNFusedHijacker` 和 `QuantizationHijacker` 中 `forward` 中的 `res = self.res_quantizer(res)` 是需要保留的。

3. 可以快速调整量化框架的 `mant_width` 和 `expo_width`，即在 `scripts/image_net.sh` 中修改 `mant_width`。  
   但是目前仍然不能在脚本中快速调整 approx 算子或者说其中的 `approx_v8` 所用的 `mant_width` 以及 `dnsmp_factor` 等参数，这些参数需要在调整 `scripts/image_net.sh` 中 `mant_width` 后在 `/home/xinyuc/jiaxiang/FP8_quantization/approx/approx_calculation.py` 中大概526行的 `QCustomTorchApprox` 类中手动修改。

## 选择空闲的 device

首先在 bash 界面输入：
```bash
nvitop
```
根据空闲或显存使用较少的 GPU 修改 `scripts/image_net.sh` 中的 device。

## 目前本地进行的工作

主要是优化脚本的自动化操作，比如上述内容 3 中的 approx 算子如何放到 `scripts/image_net.sh` 脚本中一起修改，然后通过一个新的脚本去进行批量测试。

## 新增 image_net.sh 参数

- `--approx_flag` / `--no-approx_flag`  
- `--quantize-after-mult-and-add` / `--no-quantize-after-mult-and-add`  
- `--res-quantizer-flag` / `--no-res-quantizer-flag`  

- `--expo-width ${expo_width}`  
- `--mant-width ${mant_width}`  
- `--dnsmp-factor ${dnsmp_factor}`  
- `--withComp` / `--no-withComp`  
- `--with_approx` / `--no-with_approx`  
- `--with_s2nn2s_opt` / `--no-with_s2nn2s_opt`  
- `--sim_hw_add_OFUF` / `--no-sim_hw_add_OFUF`  
- `--with_OF_opt` / `--no-with_OF_opt`  
- `--with_UF_opt` / `--no-with_UF_opt`  
- `--golden-clip-OF` / `--no-golden-clip-OF`  
- `--quant_btw_mult_accu` / `--no-quant_btw_mult_accu`  
- `--debug-mode` / `--no-debug-mode`  
- `--self-check-mode` / `--no-self-check-mode`  
