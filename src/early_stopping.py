
import config
import torch

class EarlyStopping:
    def __init__(self, patience = 5, delta = 0.001, mode = "min"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.save_path = config.MODEL_PATH

    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == "min" else val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"Model save with {val_metric: .4f} performance")