"""
train_stage1.py

Stage 1: continued (causal LM) pretraining on your diary text.
Goal: get the base model comfortable with your vocabulary, code-switching
patterns (English/Bengali/Hindi in Latin script), and personal tone,
before we shape it into a Q&A assistant in Stage 2.

Uses QLoRA (4-bit base + LoRA adapters) with gradient checkpointing and paged
optimizers so this runs smoothly on an 8GB VRAM GPU.

CHECKPOINTING
--------------
- Checkpoints (LoRA adapter weights + optimizer + scheduler state) are
  saved every --save-steps steps to --output-dir, keeping the latest N (--save-total-limit).
- To resume after an interruption, run the command with --resume.
  If a checkpoint exists in --output-dir, HF Trainer will pick it up automatically.

USAGE
-----
python train_stage1.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --train-file data/diary_blocks.jsonl \
    --output-dir checkpoints/stage1_diary \
    --epochs 3
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
    """Lightweight progress logger so you can see loss without a full TB setup."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step {state.global_step} | loss {logs['loss']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train-file", type=str, default="data/diary_blocks.jsonl")
    ap.add_argument("--output-dir", type=str, default="checkpoints/stage1_diary")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in output-dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading base model: {args.model_name} (compute_dtype: {compute_dtype})")

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
    raw_ds = load_dataset("json", data_files=args.train_file, split="train")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
        )

    tokenized_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
        callbacks=[PrintProgressCallback()],
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = get_latest_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")
        else:
            print("No existing checkpoint found. Starting fresh training.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final adapter
    final_dir = out_path / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Stage 1 completed successfully! Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()