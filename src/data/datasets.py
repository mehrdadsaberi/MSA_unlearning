from torch.utils.data import Dataset
from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index

class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        sample_fraction=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

        # load full dataset
        ds = load_hf_dataset(**hf_args)

        # optionally shuffle & sub-sample
        if sample_fraction is not None:
            assert 0 < sample_fraction <= 1, "sample_fraction must be in (0,1]"
            ds = ds.shuffle(seed=42)
            keep_n = int(len(ds) * sample_fraction)
            ds = ds.select(range(keep_n))

        # index for tracking
        self.data = add_dataset_index(ds)

        # few-shot data if provided
        self.fs_data = None
        if few_shot_dataset_hf_args:
            raw_fs = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {
                question_key: raw_fs[question_key],
                answer_key:   raw_fs[answer_key],
            }


    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index):
        if self.fs_data:
            prompts   = self.fs_data[self.question_key] + [question]
            responses = self.fs_data[self.answer_key] + [answer]
        else:
            prompts, responses = [question], [answer]

        tok = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompts,
            responses,
            self.max_length,
            self.predict_with_generate,
        )
        tok["index"] = index
        return tok

    def __getitem__(self, idx):
        row = self.data[idx]
        q, a, i = row[self.question_key], row[self.answer_key], row["index"]
        if isinstance(a, str):
            return self._process_sample(q, a, i)
        elif isinstance(a, list):
            return {j: self._process_sample(q, ans, i) for j, ans in enumerate(a)}
        else:
            raise ValueError("Unsupported answer format")
