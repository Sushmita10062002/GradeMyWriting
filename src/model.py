import config
import engine
from torch import nn
from transformers import AutoConfig, AutoModel

class FeedbackModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        hidden_dropout_prob: float = 0.0
        model_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        model_config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": hidden_dropout_prob,
            "add_pooling_layer": False,
            "num_labels": self.num_labels
        })
        self.transformer = AutoModel.from_pretrained(config.MODEL_NAME,
                                                    config = model_config)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(model_config.hidden_size, self.num_labels)

    def forward(self, ids, mask, targets = None):
        transformer_out = self.transformer(input_ids = ids, 
                                          attention_mask = mask)
        sequence_output = transformer_out.last_hidden_state[:, 0, :]
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        loss = 0

        if targets is not None:
            loss1 = engine.loss_fn(logits1, targets)
            loss2 = engine.loss_fn(logits2, targets)
            loss3 = engine.loss_fn(logits3, targets)
            loss4 = engine.loss_fn(logits4, targets)
            loss5 = engine.loss_fn(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            mcrmse = engine.monitor_metrics(logits, targets)
            return logits, loss, mcrmse
        return logits, loss, {}
            
        
        