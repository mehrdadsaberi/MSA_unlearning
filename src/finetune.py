import argparse
import os
import yaml
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from data.datasets import QADataset
from data.utils import load_hf_dataset, add_dataset_index
from data.collator import DataCollatorForSupervisedDataset

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True,
                        help="name of model config file (no .yaml)")
    parser.add_argument("--local_model_path", default=None,
                        help="if set, load weights from this dir instead of HF")
    parser.add_argument("--dataset", required=True,
                        help="name of data config file (no .yaml)")
    parser.add_argument("--output_dir", required=True,
                        help="where to save the fine-tuned model")
    parser.add_argument("--trainer_config", required=True,
                        help="path to trainer_config.yaml")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    args = parser.parse_args()

    # 1) load model config
    mcfg = load_yaml(os.path.join("configs", "model", f"{args.model_name}.yaml"))
    model_args = mcfg["model_args"].copy()
    if args.local_model_path:
        model_args["pretrained_model_name_or_path"] = args.local_model_path
    tokenizer_args = mcfg["tokenizer_args"]
    template_args = mcfg["template_args"]

    # 2) load model & tokenizer
    torch_dtype = getattr(torch, model_args.pop("torch_dtype"))
    config = AutoConfig.from_pretrained(**model_args)
    model = AutoModelForCausalLM.from_pretrained(**model_args, config=config, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 3) load data config
    dcfg = load_yaml(os.path.join("configs", "data", f"{args.dataset}.yaml"))
    ds_args = dcfg["args"]
    # if you want to support local-data override, you could check for args.local_data_path here
    train_ds = QADataset(
        hf_args=ds_args["hf_args"],
        template_args=template_args,
        tokenizer=tokenizer,
        question_key=ds_args.get("question_key", "question"),
        answer_key=ds_args.get("answer_key", "answer"),
        max_length=ds_args.get("max_length", 512),
        predict_with_generate=ds_args.get("predict_with_generate", False),
        sample_fraction=ds_args.get("sample_fraction", None),
    )
    train_ds.data = add_dataset_index(train_ds.data)

    # 4) load trainer config
    tcfg = load_yaml(args.trainer_config)
    # compute warmup_steps from warmup_epochs
    steps_per_epoch = len(train_ds) // args.per_device_train_batch_size
    warmup_steps = int(tcfg["warmup_epochs"] * steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_steps=warmup_steps,
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_strategy="no",
        logging_steps=steps_per_epoch // 10 or 1,
        fp16=False,
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        padding_side="right"
    )

    # 5) trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
