"""
prepare_data.py

Converts your raw diary + chat data (WhatsApp text exports or JSONL) into two JSONL datasets:

  1. diary_blocks.jsonl   -> plain text chunks, used for Stage 1 (continued pretraining)
  2. chat_train.jsonl / chat_val.jsonl -> chat-formatted Q&A pairs, used for Stage 2 (SFT)

USAGE
-----
python prepare_data.py \
    --diary data/diary.txt \
    --chats data/bk_chat.txt data/kp_chat.txt data/sd_chat.txt \
    --user-name "Shubhajeet Das" \
    --out-dir data \
    --context-turns 6 \
    --val-frac 0.1
"""

import argparse
import json
import random
import re
from pathlib import Path

# Match typical WhatsApp export lines: "DD/MM/YY, HH:MM - Speaker Name: Message"
# e.g., "28/04/25, 19:57 - Messages and calls are..." or "01/11/25, 19:13 - Shubhajeet Das: Hey..."
WA_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}(?:\s*(?:[aApP][mM]))?\s+-\s+([^:]+):\s+(.*)$"
)

# Strings to ignore (system notices & omitted media)
IGNORE_SUBSTRINGS = [
    "Messages and calls are end-to-end encrypted",
    "<Media omitted>",
    "<View once voice message omitted>",
    "This message was deleted",
    "You deleted this message",
]


def chunk_diary(raw_text: str, min_words: int = 120, max_words: int = 300):
    """Split diary text into training-sized chunks.

    Entries are first split on blank lines (assumed entry boundaries),
    then merged/split so each chunk is roughly min_words-max_words long.
    """
    entries = [e.strip() for e in re.split(r"\n\s*\n", raw_text) if e.strip()]

    chunks = []
    buf_words = []
    for entry in entries:
        words = entry.split()
        buf_words.extend(words)
        if len(buf_words) >= min_words:
            while len(buf_words) >= max_words:
                chunks.append(" ".join(buf_words[:max_words]))
                buf_words = buf_words[max_words:]
            if len(buf_words) >= min_words:
                chunks.append(" ".join(buf_words))
                buf_words = []
    if buf_words:
        if chunks and len(buf_words) < min_words // 2:
            chunks[-1] = chunks[-1] + " " + " ".join(buf_words)
        else:
            chunks.append(" ".join(buf_words))

    return [c for c in chunks if c.strip()]


def parse_whatsapp_file(file_path: Path, user_name: str = "Shubhajeet Das"):
    """Parses a WhatsApp export text file into a list of message dicts:
    [{"speaker": "me" | "friend", "text": "..."}, ...]
    Multi-line messages and rapid consecutive turns from the same speaker are grouped.
    """
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    raw_parsed = []
    current_msg = None

    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue

        match = WA_HEADER_RE.match(line_str)
        if match:
            speaker_raw, text = match.group(1).strip(), match.group(2).strip()
            if any(ign in text for ign in IGNORE_SUBSTRINGS) or any(ign in speaker_raw for ign in IGNORE_SUBSTRINGS):
                current_msg = None
                continue

            speaker = "me" if (user_name.lower() in speaker_raw.lower()) else "friend"
            current_msg = {"speaker": speaker, "text": text}
            raw_parsed.append(current_msg)
        else:
            # Continuation line of previous message
            if current_msg and not any(ign in line_str for ign in IGNORE_SUBSTRINGS):
                current_msg["text"] += "\n" + line_str

    # Group consecutive messages from the same speaker
    grouped = []
    for msg in raw_parsed:
        if not msg["text"].strip():
            continue
        if grouped and grouped[-1]["speaker"] == msg["speaker"]:
            grouped[-1]["text"] += "\n" + msg["text"]
        else:
            grouped.append(msg)

    return grouped


def load_chat_data(chat_paths, user_name: str):
    """Loads chat messages from WhatsApp text files or JSONL files."""
    messages = []
    for path_str in chat_paths:
        p = Path(path_str)
        if not p.exists():
            print(f"[warning] Chat file not found: {p}")
            continue

        if p.is_dir():
            files = list(p.glob("*.txt")) + list(p.glob("*.jsonl"))
        else:
            files = [p]

        for f in files:
            if f.suffix.lower() == ".txt":
                parsed = parse_whatsapp_file(f, user_name=user_name)
                print(f"[chat parser] Parsed {len(parsed)} message turns from WhatsApp file: {f.name}")
                messages.extend(parsed)
            elif f.suffix.lower() == ".jsonl":
                count = 0
                with open(f, "r", encoding="utf-8") as json_f:
                    for line in json_f:
                        line = line.strip()
                        if line:
                            messages.append(json.loads(line))
                            count += 1
                print(f"[chat parser] Loaded {count} messages from JSONL file: {f.name}")
    return messages


def build_chat_examples(messages, context_turns: int):
    """Turns a flat chronological message stream into multi-turn training examples.
    Every 'me' message becomes an assistant target, paired with preceding history.
    """
    examples = []
    history = []

    for msg in messages:
        role = "assistant" if msg["speaker"] == "me" else "user"
        if role == "assistant" and history:
            trimmed_history = history[-context_turns:]
            if trimmed_history and trimmed_history[0]["role"] == "assistant":
                idx = next((i for i, m in enumerate(trimmed_history) if m["role"] == "user"), None)
                trimmed_history = trimmed_history[idx:] if idx is not None else []
            if trimmed_history:
                examples.append(
                    {
                        "messages": trimmed_history + [{"role": "assistant", "content": msg["text"]}]
                    }
                )
        history.append({"role": role, "content": msg["text"]})

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", type=str, required=True, help="Path to diary text file")
    ap.add_argument("--chats", type=str, nargs="+", required=True, help="Path(s) to chat export .txt or .jsonl file(s)")
    ap.add_argument("--user-name", type=str, default="Shubhajeet Das", help="Name of assistant speaker in WhatsApp exports")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--context-turns", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Diary -> pretraining blocks ----
    raw_diary = Path(args.diary).read_text(encoding="utf-8")
    diary_chunks = chunk_diary(raw_diary)
    with open(out_dir / "diary_blocks.jsonl", "w", encoding="utf-8") as f:
        for chunk in diary_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
    print(f"[diary] wrote {len(diary_chunks)} blocks -> {out_dir / 'diary_blocks.jsonl'}")

    # ---- Chats -> SFT pairs ----
    messages = load_chat_data(args.chats, args.user_name)
    chat_examples = build_chat_examples(messages, args.context_turns)
    random.shuffle(chat_examples)

    if not chat_examples:
        print("[error] No chat examples generated! Please check chat files and user-name argument.")
        return

    n_val = max(1, int(len(chat_examples) * args.val_frac))
    val_examples = chat_examples[:n_val]
    train_examples = chat_examples[n_val:]

    with open(out_dir / "chat_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_dir / "chat_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[chat] {len(train_examples)} train / {len(val_examples)} val examples")
    print(f"[chat] wrote -> {out_dir / 'chat_train.jsonl'}, {out_dir / 'chat_val.jsonl'}")


if __name__ == "__main__":
    main()