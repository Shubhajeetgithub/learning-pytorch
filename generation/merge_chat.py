"""
merge_chat.py

Merges the Stage 2 LoRA adapter into the base model weights (so you get one
standalone model, no PEFT needed at inference time), saves it, and opens a
quick interactive chat loop to sanity-check the style.

USAGE
-----
python merge_chat.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --adapter checkpoints/stage2_chat/final_adapter \
    --merged-out merged_model
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", type=str, required=True, help="Path to final LoRA adapter directory")
    ap.add_argument("--merged-out", type=str, default="merged_model")
    ap.add_argument("--skip-merge", action="store_true", help="Just load adapter & chat, don't save merged model")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else "cpu",
        trust_remote_code=True,
    )

    if Path(args.adapter).exists():
        print(f"Loading adapter weights from: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model = model.merge_and_unload()
    else:
        print(f"[warning] Adapter path '{args.adapter}' not found. Using base model directly.")
        model = base_model

    if not args.skip_merge:
        print(f"Saving merged model to {args.merged_out}...")
        model.save_pretrained(args.merged_out)
        tokenizer.save_pretrained(args.merged_out)
        print(f"Merged model saved to {args.merged_out}")

    model.eval()
    history = []
    print("\n==============================================")
    print("Chat session initialized. Type 'quit' or 'exit' to end.")
    print("==============================================\n")

    while True:
        try:
            user_input = input("friend> ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in {"quit", "exit"}:
            break
        if not user_input.strip():
            continue

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        print(f"you> {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()