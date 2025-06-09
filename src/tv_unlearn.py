#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(name: str, local_path: str = None):
    """
    Read configs/model/{name}.yaml, override pretrained path if local_path set,
    and return a AutoModelForCausalLM in bfloat16 on CPU.
    """
    cfg = load_yaml(os.path.join("configs", "model", f"{name}.yaml"))
    margs = cfg["model_args"].copy()
    if local_path:
        margs["pretrained_model_name_or_path"] = local_path

    torch_dtype = getattr(torch, margs.pop("torch_dtype"))
    config = AutoConfig.from_pretrained(**margs)
    return AutoModelForCausalLM.from_pretrained(
        **margs, config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )

def main():
    p = argparse.ArgumentParser(
        description="Blend four models with unlearning formula"
    )
    # required model names
    for tag in ("clean", "corrupted", "forget", "retain"):
        p.add_argument(f"--{tag}_model_name", required=True,
                       help=f"config key for the {tag} model")
        p.add_argument(f"--local_{tag}_model_path", default=None,
                       help=f"if set, load {tag} weights from this dir")
    p.add_argument("--alpha", type=float, required=True,
                   help="α ≥ 0 (scaling for forget-clean)")
    p.add_argument("--beta",  type=float, required=True,
                   help="β ≥ 0 (scaling for retain-clean)")
    p.add_argument("--save_path", type=str, required=True,
                   help="where to write the new unlearned model")
    args = p.parse_args()

    # validate
    if args.alpha < 0 or args.beta < 0:
        raise ValueError(f"Require α ≥ 0 and β ≥ 0 but got α={args.alpha}, β={args.beta}")

    # 1) load all four models on CPU in bfloat16
    clean     = load_model(args.clean_model_name,     args.local_clean_model_path)
    corrupted = load_model(args.corrupted_model_name, args.local_corrupted_model_path)
    forget    = load_model(args.forget_model_name,    args.local_forget_model_path)
    retain    = load_model(args.retain_model_name,    args.local_retain_model_path)

    # bring corrupted onto CPU (it is already), we'll modify it in-place
    target = corrupted

    # grab state dicts
    base_sd     = clean.state_dict()
    forget_sd   = forget.state_dict()
    retain_sd   = retain.state_dict()
    target_sd   = target.state_dict()

    # 2) task vector arithmetic
    avg_norm = 0.0
    with torch.no_grad():
        for name, param in target.named_parameters():
            if name not in base_sd or name not in forget_sd or name not in retain_sd:
                raise KeyError(f"Parameter '{name}' missing in one of the models")
            d_f = forget_sd[name]   - base_sd[name]
            d_r = retain_sd[name]   - base_sd[name]
            delta = - args.alpha * d_f + args.beta * d_r
            param.add_(delta)
            avg_norm += delta.norm().item()

    avg_norm /= len(target_sd)
    print(f"✔️  Average norm of update: {avg_norm:.4f}")

    # 3) save
    os.makedirs(args.save_path, exist_ok=True)
    target.save_pretrained(args.save_path)
    print(f"✔️  Unlearned model saved to {args.save_path}")

    # 4) copy tokenizer from the *clean* model config
    clean_cfg = load_yaml(os.path.join("configs", "model",
                                       f"{args.clean_model_name}.yaml"))
    tok = AutoTokenizer.from_pretrained(**clean_cfg["tokenizer_args"], use_fast=True)
    tok.save_pretrained(args.save_path)
    print("✔️  Tokenizer saved")

if __name__ == "__main__":
    main()
