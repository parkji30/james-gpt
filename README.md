# March 29, 2026

This repository is me creating a GPT model for learning my fundamentals. It's mainly for my purposes but maybe it can be used as part of my portfolio. I will be updating it and also utilizing the model i train here for testing and implementing new techniques that come out in the literature.

# March 31, 2026
I kind of want to investigate the new Muon optimizer (2x speedup claim for LLMs). I will run an experiment with 20,000 steps using AdamW and then one with using Muon to see where the train loss converges to.


# April 1, 2026
- Fix CausalSelfAttention to be MultiHeadCausalSelfAttention.
- Final linear project output after MHA attention added (to combine the features learned from each head).
- I also am well aware this is not the most optimized throughput training. I will attempt that after I finalize the end to end training pipeline with a better tokenizer (BPE most likely) and an actual dataset to train a language model.
- Muon Optimizer is significantly better than pure AdamW. I got train loss of ~1.0 on my math train set, with only 100 steps, but AdamW takes roughly 2000 steps to get ~1.0 loss.
- I set Muon LR to 1e-3. It is proving to be significantly better than using raw adamw!.
- I also set the dtype to be bfloat16. As expected, roughly ~2x speedup (~1.98 it/s).
- Added torch compile, seems (~3.20 it/s)


# April 2, 2026
- Okay, let's optimize this training process a bit.
- The one major thing I need to add is `scaled_dot_product_attention`
- Other than that, I will use huggingface datasets (DCLM / DataComp-LM) with the GPT2 BPE tokenizer as my pretraining. This dataset is known to have 4 Trillion tokens.
- We are training on 327,680,000 tokens. 
- we are using the  `mlfoundations/dclm-baseline-1.0` as our pretraining dataset! Quite exciting!. I will look for a SFT set after this.
- Time to add `scaled-dot-product-attention` to this to make it faster.
- Okay before i move on, i need to look into SDPA, how to write my own inference function (taking logits, decoding, etc.) and BPE encoding.
- make sure i learn about temperature, top_p, top_k, repitition penalty etc.
- Learn about weight decay, gradient clipping, learning rate scheduling. 
- Really exciting though!![`real_pretraining_loss.png`](training_loss.png)



# Next Steps
- Once that is done, I will SFT it after on another Education Math Dataset.
- Finally, the last part of this repo, I will investigate the chinchilla scaling laws, code up my own BPE and scaled_dot product attention (probably these will go into scribbles).
- In another repo, I will take a already trained baseline model (Qwen 27B) and RL Tune it to my environment!