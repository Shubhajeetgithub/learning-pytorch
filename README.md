## BERT family
These are used for classification tasks: **GLUE** (General Language Understanding Evaluation).

-  **SST-2**: Binary sentiment analysis (positive/negative) on movie reviews.
-   **CoLA**: Grammatical acceptability (linguistic acceptability).
-   **MRPC**: Paraphrase detection (determining if two sentences are semantically equivalent).
-   **STS-B**: Semantic textual similarity (predicting a similarity score from 1–5).
-   **QQP**: Quora Question Pairs (identifying duplicate questions).
-   **MNLI**: Multi-genre natural language inference (entailment, contradiction, or neutral).
-   **QNLI**: Question-answering/NLI (does the sentence contain the answer to the question?).
-   **RTE**: Recognizing textual entailment.
-   **WNLI**: Winograd schema challenge (pronoun resolution).

#### Pre-training
- Dataset: Next sentence prediction and Masked lm (80-10-10)
- Model architecture: Tokenizer(wordpiece) -> **Embedding -> [MHA + LN + FF + LN] * N** ->  (classifier)
```bash

 Inputs
 ┌─────────────────────┐
 │ input_ids           │  (B, S)
 │ segment_ids         │  (B, S)
 │ input_mask          │  (B, S)
 │ masked_pos          │
 └──────────┬──────────┘
            │
            ▼
 ┌───────────────────────────────────────────────┐
 │                  EMBEDDINGS                   │
 │                                               │
 │  Token Embedding ─────┐                       │
 │                       ├──► + ──► + ──► LN ─► Dropout
 │  Position Embedding ──┤      │               │
 │                       │      │               │
 │  Segment Embedding ───┘      │               │
 │                              │               │
 └──────────────────────────────┴───────────────┘
            │
            │  (B, S, 768)
            ▼
 ╔═══════════════════════════════════════════════╗
 ║              TRANSFORMER BLOCK × 12           ║
 ║                                               ║
 ║   ┌───────────────────────────────────────┐   ║
 ║   │        Multi-Head Self-Attention      │   ║
 ║   │                                       │   ║
 ║   │  X ──┬──► Linear ──► Q ──┐            │   ║
 ║   │      ├──► Linear ──► K ──┤            │   ║
 ║   │      └──► Linear ──► V ──┤            │   ║
 ║   │                          │            │   ║
 ║   │          ┌───────────────┘            │   ║
 ║   │          ▼                            │   ║
 ║   │   QKᵀ / √(768/12)                     │   ║
 ║   │          │                            │   ║
 ║   │      + Attention Mask                 │   ║
 ║   │          │                            │   ║
 ║   │       Softmax                         │   ║
 ║   │          │                            │   ║
 ║   │          ▼                            │   ║
 ║   │       Attention                       │   ║
 ║   │          │                            │   ║
 ║   │       Attention × V                   │   ║
 ║   │          │                            │   ║
 ║   │     concat 12 heads                   │   ║
 ║   │          │                            │   ║
 ║   │       Linear(768→768)                 │   ║
 ║   └──────────┬────────────────────────────┘   ║
 ║              │                                ║
 ║              ▼                                ║
 ║       Dropout + Residual X                    ║
 ║              │                                ║
 ║             LN                                ║
 ║              │                                ║
 ║              ▼                                ║
 ║   ┌───────────────────────────────────────┐   ║
 ║   │        Position-Wise FFN              │   ║
 ║   │                                       │   ║
 ║   │       Linear 768 → 3072               │   ║
 ║   │                │                      │   ║
 ║   │               GELU                    │   ║
 ║   │                │                      │   ║
 ║   │       Linear 3072 → 768               │   ║
 ║   └──────────┬────────────────────────────┘   ║
 ║              │                                ║
 ║              ▼                                ║
 ║       Dropout + Residual                      ║
 ║              │                                ║
 ║             LN                                ║
 ║              │                                ║
 ╚══════════════╪════════════════════════════════╝
                │
                │ Repeat × 12
                ▼
        H = (B, S, 768)
                │
        ┌───────┴───────────────────┐
        │                           │
        │                           │
        ▼                           ▼
 ┌──────────────────┐       ┌────────────────────────┐
 │ Sentence Pair    │       │ Masked Language Model  │
 │ Classification   │       │                        │
 └────────┬─────────┘       └───────────┬────────────┘
          │                             │
          ▼                             ▼
      H[:, 0]                      masked_pos
          │                             │
          ▼                             ▼
   Linear 768→768                  Gather H
          │                             │
        Tanh                            │
          │                             ▼
          ▼                       Linear 768→768
   Linear 768→2                         │
          │                           GELU
          │                             │
          ▼                             ▼
   NSP logits                    LayerNorm
   (B, 2)                              │
                                      ▼
                               Linear 768→Vocab
                                      +
                               vocab bias
                                      │
                                      ▼
                                MLM logits
```
- Loss function: loss_mlm + loss_nsp

### Other members
- **RoBERTa**: BPE instead of wordpiece and loss = loss_mlm (delete the NSP head and the sentence-pair sampling logic).
- **ALBERT**
- **XLNet**: Permutation LM.