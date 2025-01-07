import config
import pandas as pd
import numpy as np
import torch
import engine
from model import FeedbackModel
from early_stopping import EarlyStopping
from torch.utils.data import RandomSampler
from tqdm import tqdm
from joblib import Parallel, delayed
from dataset import Collate, FeedbackDataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import subprocess

# def check_gpu_status():
#     result = subprocess.run(['nvidia-smi'], capture_output = True, text = True)
#     print(result.stdout)


def _prepare_data_helper(tokenizer, df, text_ids):
    samples = []
    lbls = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    for idx in tqdm(text_ids):
        full_text = df[df.text_id == idx].reset_index(drop=True).full_text.values[0]
        encoded_text = tokenizer.encode_plus(
            full_text,
            None,
            add_special_tokens=False,
        )
        input_ids = encoded_text["input_ids"]
        sample = {
            "input_ids": input_ids,
            "text_id": idx,
            "full_text": full_text,
            "attention_mask": encoded_text["attention_mask"],
            "input_labels": df[df.text_id == idx].reset_index(drop=True)[lbls].values[0, :].tolist(),
        }
        samples.append(sample)

    return samples
    

def prepare_data(df, tokenizer, num_jobs):
    samples = []
    text_ids = df["text_id"].unique()
    text_ids_splits = np.array_split(text_ids, num_jobs)
    results = Parallel(n_jobs = num_jobs, backend="multiprocessing")(
        delayed(_prepare_data_helper)(tokenizer, df, idx)
        for idx in text_ids_splits
    )
    for result in results:
        samples.extend(result)
    return samples

def run(fold):
    NUM_JOBS = 12
    df = pd.read_csv(config.TRAINING_FILE)
    target_columns = ["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
    train_dataset = df[df["FOLD"] != fold].reset_index(drop = True)
    valid_dataset = df[df["FOLD"] == fold].reset_index(drop = True)
    training_samples = prepare_data(train_dataset, config.tokenizer, num_jobs = NUM_JOBS)
    valid_samples = prepare_data(valid_dataset, config.tokenizer, num_jobs = NUM_JOBS)
    print(len(valid_samples))
    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE *  config.EPOCHS)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        num_train_steps /= n_gpu

    train_dataset = FeedbackDataset(training_samples, config.MAX_LEN, config.tokenizer)
    valid_dataset = FeedbackDataset(valid_samples, config.MAX_LEN, config.tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=config.TRAIN_BATCH_SIZE,
                                                   collate_fn=Collate(config.tokenizer, config.MAX_LEN)
                                                  )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=config.VALID_BATCH_SIZE,
                                                   collate_fn=Collate(config.tokenizer, config.MAX_LEN)
                                                  )
    print("Training Length: ", len(train_dataloader))
    print("Validation Length: ", len(valid_dataloader))

    # model =  torch.nn.parallel.DistributedDataParallel(FeedbackModel(num_labels=len(target_columns)))
    model =  FeedbackModel(num_labels=len(target_columns))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    # check_gpu_status()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_parameters, lr = 3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    early_stopping = EarlyStopping()
    for epoch in range(config.EPOCHS):
        # check_gpu_status()
        train_loss = engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        val_loss = engine.eval_fn(valid_dataloader, model, device)
        print(f"Train loss = {train_loss} Valid loss = {val_loss}")

        early_stopping(val_loss, model)

        # Check if we should stop training
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break