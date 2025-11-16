# app/grpo/sampling.py
import torch, random
from transformers import TextStreamer

def generate_k(model, tok, prompt, k=4, max_new=256, temperature=0.9, top_p=0.95,
               repetition_penalty=1.1, stop_after=None, seed_base=1234, device="cuda"):
    """Return list of (text, input_ids, output_ids, logprobs_per_token)."""
    results = []
    enc = tok(prompt, return_tensors="pt").to(device)
    for i in range(k):
        s = seed_base + i
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        out = model.generate(
            **enc,
            do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new, repetition_penalty=repetition_penalty,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            output_scores=True, return_dict_in_generate=True
        )
        seq = out.sequences[0]
        # decode
        text = tok.decode(seq, skip_special_tokens=True)
        # optional stop-after clip (e.g., cut any junk after "Final Answer:")
        if stop_after and stop_after in text:
            text = text.split(stop_after)[0] + stop_after + text.split(stop_after,1)[1].splitlines()[0][:64]
        # collect per-token logprobs (for KL/advantages if needed)
        scores = out.scores  # list of [t, vocab] logits for generated tokens
        gen_len = len(scores)
        # compute logprob of chosen token at each step
        logps = []
        for t in range(gen_len):
            logits = scores[t][0]                  # [vocab]
            token_id = seq[len(enc.input_ids[0])+t].item()
            logp = torch.log_softmax(logits, dim=-1)[token_id].item()
            logps.append(logp)
        gen_ids = seq[len(enc.input_ids[0]):]
        results.append(dict(text=text, input_ids=enc.input_ids[0], output_ids=gen_ids, logps=logps))
    return results
