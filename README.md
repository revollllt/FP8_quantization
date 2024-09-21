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
conda create -n fp8_python=3.9
conda activate fp8_python
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


