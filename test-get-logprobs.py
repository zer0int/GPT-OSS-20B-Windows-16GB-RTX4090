import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from transformers import LogitsProcessorList

model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_config = Mxfp4Config(dequantize=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quant_config
)

# What happens if we just stuff something into the model? :P
prompt = "Who are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

gen_out = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    top_p=0.9,
    no_repeat_ngram_size=6,
    repetition_penalty=1.05,
    do_sample=True,
    return_dict_in_generate=True,
    output_scores=True,
)

sequences = gen_out.sequences[0]
scores = gen_out.scores
new_tokens = sequences[len(inputs["input_ids"][0]):]

chosen_tokens_str = []

def _print_step(step_idx, tok_id, step_scores, k=5):
    probs = torch.softmax(step_scores[0], dim=-1)
    logprobs = torch.log(probs + 1e-45)
    entropy = -(probs * logprobs).sum().item()

    topk = torch.topk(probs, k)
    alts = [(int(t), float(p), float(logprobs[t])) for p, t in zip(topk.values, topk.indices)]
    chosen_lp = float(logprobs[tok_id])
    
    chosen_tokens_str.append(tokenizer.decode([tok_id]))

    tok_str = tokenizer.decode([tok_id])
    print(f"Step {step_idx+1:02d} {tok_str!r:14} logprob={chosen_lp:.4f}  H={entropy:.3f}")
    for t_id, p, lp in alts:
        s = tokenizer.decode([t_id]).replace("\n","\\n")
        print(f"   ↳ {s!r:14}  p={p:.3f}  lp={lp:.3f}")

for i, (tok_id, step_scores) in enumerate(zip(new_tokens, scores)):
    _print_step(i, tok_id, step_scores, k=5)

joined_str = "".join(chosen_tokens_str).replace("'", "").replace('"', "")
print("\n----- Full sequence below: -----\n\n", joined_str)

