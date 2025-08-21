import pandas as pd
import numpy as np 
import kagglehub
import os

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

def prepocess_subject(df):

	print("Custom subject preprocessing")

	df.loc[df["subject"] == 'politicsNews', "subject"] = 'politics'
	df.loc[df["subject"] == 'worldnews', "subject"] = 'world'
	df.loc[df["subject"] == 'Government News', "subject"] = 'government'
	df.loc[df["subject"] == 'US_News', "subject"] = 'usa'
	df.loc[df["subject"] == 'left-news', "subject"] = 'left'
	df.loc[df["subject"] == 'Middle-east', "subject"] = 'middle-east'
	df.loc[df["subject"] == 'News', "subject"] = 'news'

	df["full_text"] = (df["title"] + " " + df["text"]).apply(clean_text)
	
	print('unique subjects: ', df['subject'].unique())
	print('df dtypes: ', df.dtypes)

	return df 

def clean_text(text):

    # lowercase
    text = text.lower()

    # no urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text