import dataio
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Split data into train-test folds.')
parser.add_argument('--dataset_name', type=str, nargs='?')
parser.add_argument('--generalization', type=str, nargs='?')
parser.add_argument('--n_folds', type=int, nargs='?', default=5)
parser.add_argument('--perc_init', type=float, nargs='?', default=.2)

options = parser.parse_args()

if __name__ == "__main__":
	df = pd.read_csv("data/"+options.dataset_name+"/preprocessed_data.csv")
	if options.generalization == "strongest":
		dataio.save_strongest_folds(df, options.dataset_name, options.n_folds)
	elif options.generalization == "pseudostrong":
		dataio.save_pseudostrong_folds(df, options.dataset_name, options.perc_init, options.n_folds)
	else:
		print("Unknown generalization scheme.")