### GPT-OSS-20B on Windows, MXFP4, fits <16 GB VRAM
- Running GPT-OSS-20B on Windows, with an RTX 4090, requiring ~ 14 GB VRAM
- Weights are FP4, but rapidly de-quantized on-the-fly for computations in BF16
- How fast? 5-10 tokens/second, probably. Faster than you read, most likely.

## Usage
- Unformatted, raw, simple; streaming tokens:
```
python simple-inference-stream.py
```
- Clean formatting, reasoning + response, streaming tokens:
```
python clean-formatted-inference-stream.py
```
- Stuff some text into the model, get a short response and logprobs:
```
python test-get-logprobs.py
```
------
## SETUP
- Requires PyTorch >=2.4, install from [here](https://pytorch.org/get-started/locally/), check with:
```
pip show torch
```
What works for me / what I use: `torch 2.7.0+cu128`

- Install Triton for Windows (props to [woct0rdho](https://github.com/woct0rdho/triton-windows)):
```
pip install -U "triton-windows<3.5"
```
- Install Triton Kernels:
```
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"
```
- Uninstall huggingface transformers:
```
pip uninstall transformers
```
- Install [this fork](https://github.com/Tsumugii24/transformers):
```
pip install git+https://github.com/Tsumugii24/transformers
```
- Check accelerate is >=0.33:
```
pip show accelerate
```
- What I use / what works for me:
```
pip install accelerate==1.2.1
```
------
- Source: [github.com/huggingface/transformers/issues/39985](https://github.com/huggingface/transformers/issues/39985)
------

Example inference on RTX 4090, real time video (clean-formatted-inference-stream.py):


https://github.com/user-attachments/assets/8aa949ef-0402-4c6e-8af5-c8ee8cf1d1a0

