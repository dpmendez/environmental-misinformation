from utils import * 

def main():

	# dowload kaggle data
	df = download_data()

	# remove empties and edit subjects
	df["clean_text"] = df["text"].apply(clen_text)

	# split into train and test parts
	data_split = split_data(df['clean_text'], df['label'])
	print('\n\nsplit done')

	# convert to dataframes
	x_train = pd.DataFrame(data_split['x_train'])
	x_test = pd.DataFrame(data_split['x_test'])
	x_val = pd.DataFrame(data_split['x_val'])
	y_train = pd.Series(data_split['y_train'], name='label')
	y_test = pd.Series(data_split['y_test'], name='label')
	y_val = pd.Series(data_split['y_val'], name='label')
	print('df conversion done')

	# concatenate features with labels and domains
	train_df = pd.concat([x_train, y_train], axis=1)
	test_df = pd.concat([x_test, y_test], axis=1)
	val_df = pd.concat([x_val, y_val], axis=1)
	print('concatenation done')

	# save to csv
	train_output = "../data/train_data.csv"
	test_output = "../data/test_data.csv"
	val_output = "../data/val_data.csv"

	train_df.to_csv(train_output, index=False)
	test_df.to_csv(test_output, index=False)
	val_df.to_csv(val_output, index=False)
	print('dataframes saved in ', train_output, ', ', test_output, ' and ', val_output)

if __name__ == "__main__":
	main()