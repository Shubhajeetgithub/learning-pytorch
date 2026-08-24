"""
train_stage2.py

Stage 2: instruction fine-tuning (SFT) on your chat Q&A pairs, continuing
from the Stage 1 diary adapter (or directly from the base model). This teaches
the model the conversational Q&A form (friend asks -> you answer) in your style.

Uses QLoRA + gradient checkpointing + paged optimizers for single 8GB GPU execution.

CHECKPOINTING
-------------
- Periodic step checkpoints in --output-dir, resume with --resume.

USAGE
-----
python train_stage2.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --stage1-adapter checkpoints/stage1_diary/final_adapter \
    --train-file data/chat_train.jsonl \
    --val-file data/chat_val.jsonl \
    --output-dir checkpoints/stage2_chat \
    --epochs 4
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


def get_latest_checkpoint(output_dir: str):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = list(p.glob("checkpoint-*"))
    if not ckpts:
        return None
    try:
        ckpts.sort(key=lambda x: int(x.name.split("-")[-1]))
        return str(ckpts[-1])
    except Exception:
        return None


class PrintProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "eval_loss" in logs:
                print(f"step {state.global_step} | eval_loss {logs['eval_loss']:.4f}")
            elif "loss" in logs:
                print(f"step {state.global_step} | train_loss {logs['loss']:.4f}")


def load_replay_data(tokenizer, frac: float, n_target: int):
    """Pull a small slice of a generic public instruction dataset to regularize."""
    if frac <= 0:
        return None
    n_replay = max(1, int(n_target * frac))
    try:
        generic = load_dataset("HuggingFaceH4/no_robots", split="train")
        generic = generic.shuffle(seed=42).select(range(min(n_replay, len(generic))))
        return generic.map(
            lambda ex: {"messages": ex["messages"]},
            remove_columns=[c for c in generic.column_names if c != "messages"],
        )
    except Exception as e:
        print(f"[warning] Failed to load replay dataset: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--stage1-adapter",
        type=str,
        default=None,
        help="Path to the Stage 1 LoRA adapter to continue from. Omit to start SFT directly from base model.",
    )
    ap.add_argument("--train-file", type=str, default="data/chat_train.jsonl")
    ap.add_argument("--val-file", type=str, default="data/chat_val.jsonl")
    ap.add_argument("--output-dir", type=str, default="checkpoints/stage2_chat")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=75)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--eval-steps", type=int, default=75)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument(
        "--replay-frac",
        type=float,
        default=0.0,
        help="Fraction of generic instruction data to mix in. Set 0 to disable.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading base model for SFT: {args.model_name} (compute_dtype: {compute_dtype})")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4-bit base model ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if args.stage1_adapter and Path(args.stage1_adapter).exists():
        print(f"Loading Stage 1 adapter from {args.stage1_adapter}")
        model = PeftModel.from_pretrained(model, args.stage1_adapter, is_trainable=True)
    else:
        if args.stage1_adapter:
            print(f"[warning] Stage 1 adapter path '{args.stage1_adapter}' not found. Initializing new LoRA adapter.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ---- Dataset ----
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    replay_ds = load_replay_data(tokenizer, args.replay_frac, len(train_ds))
    if replay_ds is not None:
        train_ds = concatenate_datasets([train_ds, replay_ds]).shuffle(seed=args.seed)
        print(f"Mixed in {len(replay_ds)} generic replay examples -> {len(train_ds)} total train examples")

    def formatting_fn(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        max_length=args.max_seq_len,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_fn,
        callbacks=[PrintProgressCallback()],
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = get_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"Resuming SFT from checkpoint: {resume_ckpt}")
        else:
            print("No existing checkpoint found. Starting fresh SFT training.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = out_path / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Stage 2 SFT completed successfully! Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
