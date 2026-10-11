from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_text = """summarize: If you wished to know a system, would you study its chaos, or consider an averaged version? The answer often feels contradicting to me. Physicists would ignore the chaos, but in their description, my life would be simplified to only the polished checkpoints. Yet real life is full of stress, uncertainty, and conflict, not just the curated highlights. I am attending classes, managing deadlines, dealing with competition, and trying to keep up with difficult coursework while also handling emotional exhaustion and self-doubt."""

prompt = "summarize: " + input_text.strip()
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=120,
    min_length=30,
    num_beams=4,
    length_penalty=1.5,
    no_repeat_ngram_size=3,
    early_stopping=True,
)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
