import torch

class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_ids = [self.tokenizer.cls_token_id] + ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        res = {
            "ids": input_ids,
            "mask": attention_mask,
            "targets": input_labels,
        }
        return res


class Collate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]

        batch_max_len = max([len(ids) for ids in output["ids"]])
        if batch_max_len > self.max_len:
            batch_max_len = self.max_len

        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max_len - len(s))*[self.tokenizer.pad_token_id]
                            for s in output["ids"]]
            output["mask"] = [s + (batch_max_len - len(s))*[self.tokenizer.pad_token_id]
                            for s in output["ids"]]
        else:
            output["ids"] = [(batch_max_len - len(s)) * [self.tokenizer.pad_token_id] + s 
                             for s in output["ids"]]
            output["mask"] = [(batch_max_len - len(s)) * [0] + s 
                              for s in output["mask"]]

        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.float)
        return output