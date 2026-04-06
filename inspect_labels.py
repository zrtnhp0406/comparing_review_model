import pandas as pd
from dataset import create_aspect_dataset, prepare_training_data

train_sentences = pd.read_csv('C:/work/Research_Same_User_Diff_Opinion/Data/After_process/train_with_aspect_predictions.csv')
train_reviews = pd.read_csv('C:/work/Research_Same_User_Diff_Opinion/Data/reviews/beer-com-reviews_train.csv')
train_aspect_reviews = create_aspect_dataset(train_sentences, contain_aspect_sentences=True)
train_df = prepare_training_data(train_reviews, train_aspect_reviews)
print('aspect_pred unique:', sorted(train_df['aspect_pred'].unique().tolist()))
print('min/max', train_df['aspect_pred'].min(), train_df['aspect_pred'].max())
print('shape', train_df.shape)
print(train_df.head(5).to_string())
