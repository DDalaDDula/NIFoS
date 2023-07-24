import torch
from transformers import ElectraForSequenceClassification

def load_model(num_labels):
    return ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3", num_labels=num_labels)
