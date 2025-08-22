import pandas as pd
import numpy as np 
import kagglehub
import os
import re
import string

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split


def download_data():

	path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
	print("Path to dataset files: ", path)

	real_df = pd.read_csv(os.path.join(path, "True.csv"))
	fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))

	# Give labels
	real_df["label"] = 1
	fake_df["label"] = 0

	df = pd.concat([real_df, fake_df], axis=0).reset_index(drop=True)

	return df 


def preprocess_subject(df, include_title=True):

	print("Custom subject preprocessing")

	df.loc[df["subject"] == 'politicsNews', "subject"] = 'politics'
	df.loc[df["subject"] == 'worldnews', "subject"] = 'world'
	df.loc[df["subject"] == 'Government News', "subject"] = 'government'
	df.loc[df["subject"] == 'US_News', "subject"] = 'usa'
	df.loc[df["subject"] == 'left-news', "subject"] = 'left'
	df.loc[df["subject"] == 'Middle-east', "subject"] = 'middle-east'
	df.loc[df["subject"] == 'News', "subject"] = 'news'
	
	print('unique subjects: ', df['subject'].unique())
	print('df dtypes: ', df.dtypes)

	df.dropna(subset=['title', 'text'], inplace=True)

    # Decide what to clean
    if include_title:
    	df["clean_text"] = (df["title"] + " " + df["text"]).apply(clean_text)
    else: 
    	df["clean_text"] = df["text"].apply(clean_text)

	# Remove rows where clean_text is empty
	df = df[df['clean_text'].str.strip() != ""]

	return df 


def clean_text(text):

    text = text.lower() # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # no urls
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces

    return text


# Split into training, validation and test sets
def split_data(texts, labels, extras):
	# 70, 15, 15 split
	train_texts, temp_texts, train_labels, temp_labels, train_extras, temp_extras = train_test_split(texts, labels, extras, test_size=0.3, random_state=42)
	val_texts, test_texts, val_labels, test_labels, val_extras, test_extras = train_test_split(temp_texts, temp_labels, temp_extras, test_size=0.5, random_state=42)

	return {'x_train': train_texts, 'y_train': train_labels, 'z_train': train_extras,
			'x_test': test_texts, 'y_test': test_labels, 'z_test': test_extras,
			'x_val': val_texts, 'y_val': val_labels, 'z_val': val_extras}


# Create a PyTorch Dataset
class NewsDataset(Dataset):

	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		item = {key: val[idx] for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item
