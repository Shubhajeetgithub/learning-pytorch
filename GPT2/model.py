import torch
import torch.nn as nn
class GPT2(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, ):
        pass
    def generate(
            self,
            input_ids,  max_new_tokens, 
            num_beams, # 1 means no beam search.
            do_sample, # Whether or not to use sampling; use greedy decoding otherwise.
            temperature=1.0, 
            top_k=50, # The number of highest probability vocabulary tokens to keep for top-k-filtering
            top_p=1.0, # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
            min_p=0.01, # Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0 and 1.
            repetition_penalty=1.0, # 1.0 means no penalty
            no_repeat_ngram_size=0.0, # If set to > 0, all ngrams of that size can only occur once.
            watermark_config = None, # TODO: SynthID
        ):
        pass