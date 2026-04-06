import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from dataset import create_aspect_dataset, prepare_training_data
from arch import model_Bert
from train import train_model
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
train_sentences = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\After_process\train_with_aspect_predictions.csv")
test_sentences = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\After_process\test_with_aspect_predictions.csv")
val_sentences = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\After_process\validation_with_aspect_predictions.csv")

train_reviews = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\reviews\beer-com-reviews_train.csv")
test_reviews = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\reviews\beer-com-reviews_test.csv")
val_reviews = pd.read_csv(r"C:\work\Research_Same_User_Diff_Opinion\Data\reviews\beer-com-reviews_validation.csv")


train_aspect_reviews = create_aspect_dataset(train_sentences, contain_aspect_sentences=True)
test_aspect_reviews = create_aspect_dataset(test_sentences, contain_aspect_sentences=True)
val_aspect_reviews = create_aspect_dataset(val_sentences, contain_aspect_sentences=True)

train_df = prepare_training_data(train_reviews, train_aspect_reviews)
val_df = prepare_training_data(val_reviews, val_aspect_reviews)

model = model_Bert(freeze_layers=8)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

best_model = train_model(model, train_df, val_df, tokenizer, optimizer, num_epochs=10, batch_size=16)
