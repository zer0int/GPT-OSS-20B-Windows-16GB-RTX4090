import os, sys, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, TextIteratorStreamer

# -----------------------------
# Model setup
# -----------------------------
model_name = "openai/gpt-oss-20b"
# Or "path/to/downloaded/model"
os.environ["CL"] = "/nologo"  # quiet MSVC banners (compiler verbose spam) on Windows

tokenizer = AutoTokenizer.from_pretrained(model_name)
quant_config = Mxfp4Config(dequantize=False) # or True to dequantize -> BF16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    quantization_config=quant_config
).eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are GPT-OSS, an open-weights large language model by OpenAI, running local. "
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

# -----------------------------
# Streaming with robust tag handling
# -----------------------------
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=False,  # preserve raw tags for detection
)

gen_kwargs = dict(
    **inputs,
    max_new_tokens=2000,
    temperature=0.7,
    do_sample=True,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,
)

thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
thread.start()

# Tags we care about (as *strings* emitted by the tokenizer)
ANALYSIS_TAG  = "<|channel|>analysis"
FINAL_TAG     = "<|channel|>final"
DROP_TAGS     = ("<|message|>", "<|start|>", "<|end|>", "<|return|>")
ANALYSIS_HDR  = "\n----- REASONING CHAIN-OF-THOUGHT -----\n\n"
FINAL_HDR     = "\n\n----- MODEL RESPONSE -----\n\n"

tags_all = [ANALYSIS_TAG, FINAL_TAG, *DROP_TAGS]
max_tag_len = max(len(t) for t in tags_all)

hold = ""            # unprocessed tail
analysis_seen = False
final_seen = False

# capture clean text for files
mode = None         # None | "analysis" | "final"
cap_analysis = []
cap_final = []

def longest_suffix_that_is_tag_prefix(s: str) -> int:
    """Return length L of the longest suffix of s that is a prefix of any tag."""
    maxL = 0
    for tag in tags_all:
        # check up to len(tag)-1 so we can hold potential partial matches
        for L in range(1, min(len(s), len(tag)) + 1):
            if s[-L:] == tag[:L]:
                if L > maxL:
                    maxL = L
    return maxL

def emit_transformed(text: str):
    """Emit text while replacing/stripping tags. Update captures by current mode."""
    global analysis_seen, final_seen, mode
    i = 0
    N = len(text)
    while i < N:
        # Find nearest tag occurrence
        next_pos = N
        next_tag = None
        for tag in tags_all:
            j = text.find(tag, i)
            if j != -1 and j < next_pos:
                next_pos, next_tag = j, tag

        if next_tag is None:
            chunk = text[i:]
            if chunk:
                sys.stdout.write(chunk)
                if mode == "analysis":
                    cap_analysis.append(chunk)
                elif mode == "final":
                    cap_final.append(chunk)
            break

        # Emit plain text up to the tag (with special handling for 'assistant' right before FINAL_TAG)
        plain = text[i:next_pos]
        if next_tag == FINAL_TAG:
            # trim a trailing 'assistant' just before the FINAL tag (common in templates)
            if plain.endswith("assistant"):
                trimmed = plain[:-len("assistant")]
                if trimmed:
                    sys.stdout.write(trimmed)
                    if mode == "analysis":
                        cap_analysis.append(trimmed)
                    elif mode == "final":
                        cap_final.append(trimmed)
            else:
                if plain:
                    sys.stdout.write(plain)
                    if mode == "analysis":
                        cap_analysis.append(plain)
                    elif mode == "final":
                        cap_final.append(plain)
        else:
            if plain:
                sys.stdout.write(plain)
                if mode == "analysis":
                    cap_analysis.append(plain)
                elif mode == "final":
                    cap_final.append(plain)

        # Handle the tag itself
        if next_tag == ANALYSIS_TAG and not analysis_seen:
            sys.stdout.write(ANALYSIS_HDR)
            analysis_seen = True
            mode = "analysis"
        elif next_tag == FINAL_TAG and not final_seen:
            sys.stdout.write(FINAL_HDR)
            final_seen = True
            mode = "final"
        # DROP_TAGS are ignored entirely

        i = next_pos + len(next_tag)

    sys.stdout.flush()

# Consume stream live
for chunk in streamer:
    hold += chunk

    # Compute safe prefix that cannot contain a partial tag tail
    L = longest_suffix_that_is_tag_prefix(hold)
    safe_len = len(hold) - L
    if safe_len > 0:
        emit_transformed(hold[:safe_len])
        hold = hold[safe_len:]

# Flush any remaining buffered text at end
thread.join()
if hold:
    emit_transformed(hold)

print("\n\n----- END OF MODEL RESPONSE STREAM -----")

# -----------------------------
# Save clean captures
# -----------------------------
with open("analysis.txt", "w", encoding="utf-8") as fa:
    fa.write("".join(cap_analysis))
with open("final.txt", "w", encoding="utf-8") as ff:
    ff.write("".join(cap_final))

print("[Saved analysis -> analysis.txt]")
print("[Saved final    -> final.txt]")
