from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import torch
import os

from src.data.vqa_dataset import VQADataset

dataset_type = os.getenv('VQA_DATASET_TYPE', "mini")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    return img

def display_example(dataset, example):
  question_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
  image = unnormalize(example["image"])
  answer_label = dataset.answer_classes[example["label"]]

  print(f"Question: {question_text}")
  print(f"Answer label: {answer_label}")

  plt.imshow(image.permute(1, 2, 0))
  plt.show()

dataset = VQADataset(dataset_type)
example = dataset[0]
display_example(dataset, example)