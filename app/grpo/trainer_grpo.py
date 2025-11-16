# app/grpo/trainer_grpo.py
import os, json, math, torch, random
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from .sampling import generate_k
from .rewards import total_reward


@dataclass
class Args:
    # use your DeepSeek distill as base
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    out_dir: str = "/workspace/ckpt/base-grpo"

    # tiny toy dataset
    train_path: str = "/workspace/data/train.jsonl"
    val_path: str = "/workspace/data/val.jsonl"

    # GRPO config
    k: int = 4                   # samples per prompt
    max_new: int = 256
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 1          # micro-batch
    grad_accum: int = 16
    alpha: float = 0.5           # ORM weight
    beta: float = 0.5            # PRM weight
    kl_coef: float = 0.01        # KL penalty to reference model

    # LoRA / precision
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_bf16: bool = False       # can keep False on 16GB GPU


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def kl_divergence(pol_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """
    Simple L2 distance between policy and reference logprobs.
    pol_logps, ref_logps: shape [T]
    """
    T = min(pol_logps.shape[0], ref_logps.shape[0])
    diffs = pol_logps[:T] - ref_logps[:T]
    return torch.mean(diffs * diffs)


def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ---------- tokenizer ----------
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- reference model (frozen) ----------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    ref.eval()

    # ---------- policy model (LoRA on 4-bit base) ----------
    pol = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    pol = prepare_model_for_kbit_training(pol)
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    pol = get_peft_model(pol, lcfg)
    pol.train()

    optim = torch.optim.AdamW(pol.parameters(), lr=args.lr)

    # ---------- data ----------
    train = load_dataset("json", data_files=args.train_path, split="train")

    step = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_bf16)

    device = next(pol.parameters()).device

    for epoch in range(args.epochs):
        for ex in tqdm(train, desc=f"epoch{epoch}"):
            prompt = ex["prompt"]
            gold   = ex.get("gold", None)

            # 1) sample K candidates from current policy (no grad)
            cands = generate_k(
                pol, tok, prompt,
                k=args.k,
                max_new=args.max_new,
                device=device,
            )

            # 2) compute rewards (ORM + PRM)
            rewards = []
            reward_parts = []
            for c in cands:
                R, parts = total_reward(
                    c["text"],
                    gold,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                rewards.append(R)
                reward_parts.append(parts)

            # convert to tensor for advantages
            R = torch.tensor(rewards, dtype=torch.float32, device=device)
            A = R - R.mean()   # group-relative baseline

            # 3) compute reference logprobs (teacher forcing)
            ref_logps_all = []
            with torch.no_grad():
                enc = tok(prompt, return_tensors="pt").to(device)
                for c in cands:
                    # concat prompt + generated ids
                    input_ids = torch.cat(
                        [enc.input_ids, c["output_ids"].unsqueeze(0)], dim=1
                    )
                    attn_mask = torch.ones_like(input_ids)
                    out_ref = ref(input_ids=input_ids, attention_mask=attn_mask)
                    logits_ref = out_ref.logits[0, -len(c["output_ids"]):]  # [T, V]

                    logps_ref = []
                    for t, wid in enumerate(c["output_ids"]):
                        logp = torch.log_softmax(logits_ref[t], dim=-1)[wid]
                        logps_ref.append(logp.detach())
                    ref_logps_all.append(torch.stack(logps_ref))   # [T]

            # 4) compute policy logprobs again (this time with grad)
            pol_logps_all = []
            enc = tok(prompt, return_tensors="pt").to(device)
            for c in cands:
                input_ids = torch.cat(
                    [enc.input_ids, c["output_ids"].unsqueeze(0)], dim=1
                )
                attn_mask = torch.ones_like(input_ids)
                out_pol = pol(input_ids=input_ids, attention_mask=attn_mask)
                logits_pol = out_pol.logits[0, -len(c["output_ids"]):]  # [T, V]

                logps_pol = []
                for t, wid in enumerate(c["output_ids"]):
                    logp = torch.log_softmax(logits_pol[t], dim=-1)[wid]
                    logps_pol.append(logp)
                pol_logps_all.append(torch.stack(logps_pol))  # [T]

            # 5) build GRPO-style loss: REINFORCE + KL
            pol_loss = torch.zeros((), device=device)
            kl_loss  = torch.zeros((), device=device)

            for i in range(len(cands)):
                adv = A[i]
                seq_logp = pol_logps_all[i].sum()          # log p_\theta(y | x)
                pol_loss = pol_loss + (-adv * seq_logp)    # maximize adv * logp

                # KL between policy and reference logprobs
                kl_loss = kl_loss + kl_divergence(
                    pol_logps_all[i].detach(), ref_logps_all[i]
                )

            pol_loss = pol_loss / len(cands)
            kl_loss  = kl_loss  / len(cands)

            loss = pol_loss + args.kl_coef * kl_loss

            # 6) optimize LoRA params
            loss = loss / args.grad_accum
            loss.backward()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        # save adapter after each epoch
        epoch_dir = os.path.join(args.out_dir, f"epoch{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        pol.save_pretrained(epoch_dir)

    # final save
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    pol.save_pretrained(final_dir)
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump({"args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
