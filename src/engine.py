from sklearn.metrics import mean_squared_error
from torch import nn
from tqdm import tqdm
import torch
import numpy as np
import torch

def check_gpu_status():
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        device = torch.device(f"cuda:{gpu_id}")
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**2
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**2
        free_memory = total_memory - reserved_memory
        
        print(f"GPU {gpu_id}:")
        print(f"  Total Memory:    {total_memory:.2f} MB")
        print(f"  Allocated Memory:{allocated_memory:.2f} MB")
        print(f"  Cached Memory:   {reserved_memory:.2f} MB")
        print(f"  Free Memory:     {free_memory:.2f} MB")
        print("-" * 30)

def loss_fn(outputs, targets):
    loss_fct = nn.MSELoss()
    loss = loss_fct(outputs, targets)
    return loss

def monitor_metrics(outputs, targets):
    device = targets.get_device()
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    num_labels = 6
    mcrmse = []
    for i in range(num_labels):
        mcrmse.append(
            mean_squared_error(
                targets[:, i],
                outputs[:, i],
                squared = False
            ),
        )
    mcrmse = np.mean(mcrmse)
    return {"mcrmse": torch.tensor(mcrmse, device = device)}

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    print("In train_fn: ", len(data_loader))
    for data in tqdm(data_loader, total = len(data_loader)):
        
        # torch.cuda.empty_cache()
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        # check_gpu_status()
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    with torch.no_grad():  
        for data in tqdm(data_loader, total = len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            _, loss, _ = model(**data)
            final_loss += loss.item()
    return final_loss / len(data_loader)

