import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Prepare datasets.')
parser.add_argument('--dataset', type=str, nargs='?', default='assistments12')
parser.add_argument('--min_interactions', type=int, nargs='?', default=10)
parser.add_argument('--remove_nan_skills', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

def prepare_assistments12(min_interactions_per_user, remove_nan_skills):
	"""Preprocess ASSISTments 2012-2013 dataset.

	Arguments:
	min_interactions_per_user -- minimum number of interactions per student
	remove_nan_skills -- if True, remove interactions with no skill tag

	Outputs:
	df -- preprocessed ASSISTments dataset (pandas DataFrame)
	Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
	"""
	df = pd.read_csv("data/assistments12/data.csv")
	
	df["timestamp"] = df["start_time"]
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds() / (3600*24))
	df.sort_values(by="timestamp", inplace=True)
	df.reset_index(inplace=True, drop=True)
	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

	if remove_nan_skills:
		df = df[~df["skill_id"].isnull()]
	else:
		df.ix[df["skill_id"].isnull(), "skill_id"] = -1

	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
	df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
	df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]
	
	df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
	df["inter_id"] = df.index

	# Build Q-matrix
	Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
	item_skill = np.array(df[["item_id", "skill_id"]])
	for i in range(len(item_skill)):
		Q_mat[item_skill[i,0],item_skill[i,1]] = 1

	df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id"]]
	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes

	# Save data
	sparse.save_npz("data/assistments12/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv("data/assistments12/preprocessed_data.csv", sep="\t", index=False)

	return df, Q_mat

def prepare_kddcup10(data_name, min_interactions_per_user, kc_col_name,
					 remove_nan_skills, drop_duplicates=True):
	"""Preprocess KDD Cup 2010 datasets.

	Arguments:
	data_name -- "bridge_algebra06" or "algebra05"
	min_interactions_per_user -- minimum number of interactions per student
	kc_col_name -- Skills id column
	remove_nan_skills -- if True, remove interactions with no skill tag
	drop_duplicates -- if True, drop duplicates from dataset

	Outputs:
	df -- preprocessed ASSISTments dataset (pandas DataFrame)
	Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
	"""
	folder_path = os.path.join("data", data_name)
	df = pd.read_csv(folder_path + "/data.txt", delimiter='\t').rename(columns={
		'Anon Student Id': 'user_id',
		'Problem Name': 'pb_id',
		'Step Name': 'step_id',
		kc_col_name: 'kc_id',
		'First Transaction Time': 'timestamp',
		'Correct First Attempt': 'correct'
	})[['user_id', 'pb_id', 'step_id' ,'correct', 'timestamp', 'kc_id']]
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds() / (3600*24))
	df.sort_values(by="timestamp",inplace=True)
	df.reset_index(inplace=True,drop=True)
	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

	# Create variables
	df["item_id"] = df["pb_id"]+":"+df["step_id"]
	df = df[['user_id', 'item_id', 'kc_id', 'correct', 'timestamp']]

	if drop_duplicates:
		df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)
	
	if remove_nan_skills:
		df = df[~df["kc_id"].isnull()]
	else:
		df.ix[df["kc_id"].isnull(), "kc_id"] = 'NaN'

	# Create list of KCs
	listOfKC = []
	for kc_raw in df["kc_id"].unique():
		for elt in kc_raw.split('~~'):
			listOfKC.append(elt)
	listOfKC = np.unique(listOfKC)

	dict1_kc = {}
	dict2_kc = {}
	for k, v in enumerate(listOfKC):
		dict1_kc[v] = k
		dict2_kc[k] = v

	# Transform ids into numeric
	df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]

	df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
	df["inter_id"] = df.index

	# Build Q-matrix
	Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
	item_skill = np.array(df[["item_id","kc_id"]])
	for i in range(len(item_skill)):
		splitted_kc = item_skill[i,1].split('~~')
		for kc in splitted_kc:
			Q_mat[item_skill[i,0],dict1_kc[kc]] = 1

	df = df[['user_id', 'item_id', 'timestamp', 'correct', 'inter_id']]
	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
	
	# Save data
	sparse.save_npz(folder_path + "/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv(folder_path + "/preprocessed_data.csv", sep="\t", index=False)

	return df, Q_mat

if __name__ == "__main__":
	if options.dataset == "assistments12":
		df, Q_mat = prepare_assistments12(min_interactions_per_user=options.min_interactions,
										  remove_nan_skills=options.remove_nan_skills)
	elif options.dataset == "bridge_algebra06":
		df, Q_mat = prepare_kddcup10(data_name="bridge_algebra06",
									 min_interactions_per_user=options.min_interactions,
									 kc_col_name="KC(SubSkills)",
									 remove_nan_skills=options.remove_nan_skills)
	elif options.dataset == "algebra05":
		df, Q_mat = prepare_kddcup10(data_name="algebra05",
									 min_interactions_per_user=options.min_interactions,
									 kc_col_name="KC(Default)",
									 remove_nan_skills=options.remove_nan_skills)
