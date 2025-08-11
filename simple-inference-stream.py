import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TextStreamer
import os

model_name = "openai/gpt-oss-20b"
# Or "path/to/downloaded/model"

# This stops triton compiler spam, stuff like:
# Creating library <path>\__triton_launcher.cp310-win_amd64.lib and object <path>
os.environ["CL"] = "/nologo"

tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_config = Mxfp4Config(dequantize=False)  # or True to dequantize -> BF16
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quant_config)

messages = [
    {
        "role": "system",
        "content": (
            "You are GPT-OSS, an open-weights large language model by OpenAI, running local."
            "You are a helpful assistant. Provide nerdy responses."
            "You should not use lists in chit chat, in casual conversations, or in empathetic or advice-driven conversations."
        )
    },
    {"role": "user", "content": "Who are you? And how do you work? What is your purpose?"}
]


inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Streaming tokens
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
 
_ = model.generate(
    **inputs,
    max_new_tokens=2000,
    temperature=0.7,
    do_sample=True,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,
)