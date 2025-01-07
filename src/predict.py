import config
from config import tokenizer
from model import FeedbackModel
import torch

def prepare_input_data(paragraph):
  encoded_text = tokenizer.encode_plus(
      paragraph,
      None,
      add_special_tokens=True,
      max_length=config.MAX_LEN,
      pad_to_max_length=True,
      return_attention_mask=True
  )
  sample = {
      "input_ids": encoded_text["input_ids"],
      "attention_mask": encoded_text["attention_mask"]
  }
  return sample

def predict_scores(model, input_data, device):
  model.eval()
  input_ids = torch.tensor(input_data["input_ids"]).unsqueeze(0).to(device)
  attention_mask = torch.tensor(input_data["attention_mask"]).unsqueeze(0).to(device)

  with torch.no_grad():
    logits, _, _ = model(input_ids, attention_mask)
    scores = logits.squeeze().cpu().numpy()
  return scores

metric_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
model = FeedbackModel(num_labels=len(metric_names))
device = "cuda"
model.load_state_dict(torch.load(config.MODEL_PATH, map_location = device))
model.to(device)


def predict(paragraph):
  input_data = prepare_input_data(paragraph)
  scores = predict_scores(model, input_data, device)
  results = {metric_names[i]: score for i, score in enumerate(scores)}

  print("Prediction scores for the given paragraph:")
  for metric, score in results.items():
      print(f"{metric}: {score:.4f}")
