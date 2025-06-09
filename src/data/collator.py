## File: data/collator.py
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning, padding input_ids and labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        return [instance[key] for instance in instances]

    def _pad_tokens(self, sequences: Sequence[torch.Tensor], padding_value: int):
        # pad on right or left
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=padding_value
            )
        else:
            flipped = [seq.flip(dims=[0]) for seq in sequences]
            padded = torch.nn.utils.rnn.pad_sequence(
                flipped, batch_first=True, padding_value=padding_value
            )
            return padded.flip(dims=[1])

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # if nested dict-of-lists, recurse
        if not isinstance(instances[0], dict):
            raise ValueError("Each instance must be a dict of tensors.")
        if "input_ids" not in instances[0]:
            # loop over keys
            batched = {}
            for key in instances[0].keys():
                batched[key] = self(self.get_instances_from_key(instances, key))
            return batched

        # pad input_ids
        input_ids = self._pad_tokens(
            [inst["input_ids"] for inst in instances],
            self.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # pad labels if present
        if "labels" in instances[0]:
            labels = self._pad_tokens(
                [inst["labels"] for inst in instances],
                IGNORE_INDEX,
            )
            batch["labels"] = labels

        # collect index if requested
        if self.index:
            if self.index in instances[0]:
                batch[self.index] = torch.tensor(
                    [inst[self.index] for inst in instances], dtype=torch.long
                )
            else:
                raise KeyError(f"Index field '{self.index}' not found in instances")

        return batch
