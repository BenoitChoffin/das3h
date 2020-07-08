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
parser.add_argument('--verbose', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()


def prepare_assistments12(min_interactions_per_user, remove_nan_skills, verbose):
	"""Preprocess ASSISTments 2012-2013 dataset.

	Arguments:
	min_interactions_per_user -- minimum number of interactions per student
	remove_nan_skills -- if True, remove interactions with no skill tag

	Outputs:
	df -- preprocessed ASSISTments dataset (pandas DataFrame)
	Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
	"""
	df = pd.read_csv("data/assistments12/data.csv")
	if verbose:
		initial_shape = df.shape[0]
		print("Opened ASSISTments 2012 data. Output: {} samples.".format(initial_shape))
	
	df["timestamp"] = df["start_time"]
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
	#df.sort_values(by="timestamp", inplace=True)
	#df.reset_index(inplace=True, drop=True)
	if remove_nan_skills:
		df = df[~df["skill_id"].isnull()]
		if verbose:
			print("Removed {} samples with NaN skills.".format(df.shape[0]-initial_shape))
			initial_shape = df.shape[0]
	else:
		df.loc[df["skill_id"].isnull(), "skill_id"] = -1

	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
	if verbose:
		print("Removed {} samples with non-binary outcomes.".format(df.shape[0]-initial_shape))
		initial_shape = df.shape[0]
	df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
	if verbose:
		print('Removed {} samples (users with less than {} interactions).'.format((df.shape[0]-initial_shape,
																				   min_interactions_per_user)))
		initial_shape = df.shape[0]

	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
	df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
	df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]
	
	#df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
	#df["inter_id"] = df.index

	# Build Q-matrix
	Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
	item_skill = np.array(df[["item_id", "skill_id"]])
	for i in range(len(item_skill)):
		Q_mat[item_skill[i,0],item_skill[i,1]] = 1
	if verbose:
		print("Computed q-matrix. Shape: {}.".format(Q_mat.shape))

	#df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id"]]
	df = df[['user_id', 'item_id', 'timestamp', 'correct']]
	# Remove potential duplicates
	df.drop_duplicates(inplace=True)
	if verbose:
		print("Removed {} duplicated samples.".format(df.shape[0] - initial_shape))
		initial_shape = df.shape[0]

	df.sort_values(by="timestamp", inplace=True)
	df.reset_index(inplace=True, drop=True)
	print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))
	# Save data
	sparse.save_npz("data/assistments12/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv("data/assistments12/preprocessed_data.csv", index=False)

	return df, Q_mat

def prepare_assistments09(min_interactions_per_user, remove_nan_skills, verbose):
	"""Preprocess ASSISTments 2009-2010 dataset.
	Requires the collapsed version: skill_builder_data_corrected_collapsed.csv
	Download it on: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010 (the last link)

	Actually thanks to the ASSISTments team, we had access to another file,
	timestamp_data.csv, that contains the timestamps.
	This extra file does not seem openly available yet.

	Arguments:
	min_interactions_per_user -- minimum number of interactions per student
	remove_nan_skills -- if True, remove interactions with no skill tag

	Outputs:
	df -- preprocessed ASSISTments dataset (pandas DataFrame)
	Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
	"""

	df = pd.read_csv("data/assistments09/skill_builder_data_corrected_collapsed.csv",
		encoding = "latin1", index_col=False)
	df.drop(['Unnamed: 0'], axis=1, inplace=True)
	if verbose:
		initial_shape = df.shape[0]
		print("Opened ASSISTments 2009 data. Output: {} samples.".format(initial_shape))
	timestamps = pd.read_csv("data/assistments09/timestamp_data.csv")

	df = df.merge(timestamps, left_on="order_id", right_on="problem_log_id", how="inner")
	df["timestamp"] = df["start_time"]
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
	#df.sort_values(by="timestamp", inplace=True)
	#df.reset_index(inplace=True, drop=True)

	# Remove NaN skills
	if remove_nan_skills:
		initial_shape = df.shape[0] # in case the merge above removed some samples
		df = df[~df["skill_id"].isnull()]
		if verbose:
			print("Removed {} samples with NaN skills.".format(df.shape[0]-initial_shape))
			initial_shape = df.shape[0]
	else:
		df.loc[df["skill_id"].isnull(), "skill_id"] = -1

	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
	if verbose:
		print("Removed {} samples with non-binary outcomes.".format(df.shape[0]-initial_shape))
		initial_shape = df.shape[0]
	df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
	if verbose:
		print('Removed {} samples (users with less than {} interactions).'.format((df.shape[0]-initial_shape,
																				   min_interactions_per_user)))
		initial_shape = df.shape[0]

	df["item_id"] = np.unique(df["problem_id"], return_inverse=True)[1]
	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]

	# Build q-matrix
	listOfKC = []
	for kc_raw in df["skill_id"].unique():
		for elt in str(kc_raw).split('_'):
			listOfKC.append(str(int(float(elt))))
	listOfKC = np.unique(listOfKC)

	dict1_kc = {} ; dict2_kc = {}
	for k, v in enumerate(listOfKC):
		dict1_kc[v] = k
		dict2_kc[k] = v

	# Build Q-matrix
	Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
	item_skill = np.array(df[["item_id","skill_id"]])
	for i in range(len(item_skill)):
		splitted_kc = str(item_skill[i,1]).split('_')
		for kc in splitted_kc:
			Q_mat[item_skill[i,0],dict1_kc[str(int(float(kc)))]] = 1
	if verbose:
		print("Computed q-matrix. Shape: {}.".format(Q_mat.shape))

	df = df[['user_id', 'item_id', 'timestamp', 'correct']]
	# Remove potential duplicates
	df.drop_duplicates(inplace=True)
	if verbose:
		print("Removed {} duplicated samples.".format(df.shape[0] - initial_shape))
		initial_shape = df.shape[0]

	df.sort_values(by="timestamp", inplace=True)
	df.reset_index(inplace=True, drop=True)
	print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))

	# Save data
	sparse.save_npz("data/assistments09/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv("data/assistments09/preprocessed_data.csv", index=False)

	return df, Q_mat

def prepare_kddcup10(data_name, min_interactions_per_user, kc_col_name,
					 remove_nan_skills, verbose, drop_duplicates=True):
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
	if verbose:
		initial_shape = df.shape[0]
		print("Opened KDD Cup 2010 data. Output: {} samples.".format(initial_shape))

	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
	#df.sort_values(by="timestamp",inplace=True)
	#df.reset_index(inplace=True,drop=True)

	if remove_nan_skills:
		df = df[~df["kc_id"].isnull()]
		if verbose:
			print("Removed {} samples with NaN skills.".format(df.shape[0]-initial_shape))
			initial_shape = df.shape[0]
	else:
		df.loc[df["kc_id"].isnull(), "kc_id"] = 'NaN'

	df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
	if verbose:
		print("Removed {} samples with non-binary outcomes.".format(df.shape[0]-initial_shape))
		initial_shape = df.shape[0]
	df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

	df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
	if verbose:
		print('Removed {} samples (users with less than {} interactions).'.format((df.shape[0]-initial_shape,
																				   min_interactions_per_user)))
		initial_shape = df.shape[0]

	# Create variables
	df["item_id"] = df["pb_id"]+":"+df["step_id"]
	df = df[['user_id', 'item_id', 'kc_id', 'correct', 'timestamp']]

	# Transform ids into numeric
	df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
	df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]

	#if drop_duplicates:
	#	df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

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

	#df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
	#df["inter_id"] = df.index

	# Build Q-matrix
	Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
	item_skill = np.array(df[["item_id","kc_id"]])
	for i in range(len(item_skill)):
		splitted_kc = item_skill[i,1].split('~~')
		for kc in splitted_kc:
			Q_mat[item_skill[i,0],dict1_kc[kc]] = 1
	if verbose:
		print("Computed q-matrix. Shape: {}.".format(Q_mat.shape))

	#df = df[['user_id', 'item_id', 'timestamp', 'correct', 'inter_id']]
	df = df[['user_id', 'item_id', 'timestamp', 'correct']]
	# Remove potential duplicates
	df.drop_duplicates(inplace=True)
	if verbose:
		print("Removed {} duplicated samples.".format(df.shape[0] - initial_shape))
		initial_shape = df.shape[0]
	
	df.sort_values(by="timestamp", inplace=True)
	df.reset_index(inplace=True, drop=True)
	print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))
	
	# Save data
	sparse.save_npz(folder_path + "/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv(folder_path + "/preprocessed_data.csv", index=False)

	return df, Q_mat

def prepare_robomission(min_interactions_per_user, verbose):
	"""Preprocess Robomission dataset.
	Retrieved from https://github.com/adaptive-learning/adaptive-learning-research/tree/master/data/robomission-2019-12

	Arguments:
	min_interactions_per_user -- minimum number of interactions per student

	Outputs:
	df -- preprocessed Robomission dataset (pandas DataFrame)
	Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
	"""

	df = pd.read_csv("data/robomission/attempts.csv") # from robomission-2019-12-10
	if verbose:
		initial_shape = df.shape[0]
		print("Opened Robomission data. Output: {} samples.".format(initial_shape))

	df["correct"] = df["solved"].astype(np.int32)
	df["timestamp"] = df["start"]
	df["timestamp"] = pd.to_datetime(df["timestamp"])
	df["timestamp"] = df["timestamp"] - df["timestamp"].min()
	df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
	#df.sort_values(by="timestamp",inplace=True)
	#df.reset_index(inplace=True,drop=True)
	df = df.groupby("student").filter(lambda x: len(x) >= options.min_interactions)
	if verbose:
		print('Removed {} samples (users with less than {} interactions).'.format((df.shape[0]-initial_shape,
																				   min_interactions_per_user)))
		initial_shape = df.shape[0]

	# Change user/item identifiers
	df["user_id"] = np.unique(df["student"], return_inverse=True)[1]
	df["item_id"] = np.unique(df["problem"], return_inverse=True)[1]

	#df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
	#df["inter_id"] = df.index

	#df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id"]]
	df = df[['user_id', 'item_id', 'timestamp', 'correct']]
	# Remove potential duplicates
	df.drop_duplicates(inplace=True)
	if verbose:
		print("Removed {} duplicated samples.".format(df.shape[0] - initial_shape))
		initial_shape = df.shape[0]
	
	df.sort_values(by="timestamp",inplace=True)
	df.reset_index(inplace=True, drop=True)
	print("Data preprocessing done. Final output: {} samples.".format((df.shape[0])))

	# Sort q-matrix by item id
	Q_mat = pd.read_csv("data/robomission/qmatrix.csv")
	Q_mat.sort_values(by="id",inplace=True)
	Q_mat = Q_mat.values[:,1:]

	# Save data
	sparse.save_npz("data/robomission/q_mat.npz", sparse.csr_matrix(Q_mat))
	df.to_csv("data/robomission/preprocessed_data.csv", index=False)

	return df, Q_mat

if __name__ == "__main__":
	if options.dataset == "assistments12":
		df, Q_mat = prepare_assistments12(min_interactions_per_user=options.min_interactions,
										  remove_nan_skills=options.remove_nan_skills,
										  verbose=options.verbose)
	if options.dataset == "asssistments09":
		df, Q_mat = prepare_assistments09(min_interactions_per_user=options.min_interactions,
										  remove_nan_skills=options.remove_nan_skills,
										  verbose=options.verbose)
	elif options.dataset == "bridge_algebra06":
		df, Q_mat = prepare_kddcup10(data_name="bridge_algebra06",
									 min_interactions_per_user=options.min_interactions,
									 kc_col_name="KC(SubSkills)",
									 remove_nan_skills=options.remove_nan_skills,
									 verbose=options.verbose)
	elif options.dataset == "algebra05":
		df, Q_mat = prepare_kddcup10(data_name="algebra05",
									 min_interactions_per_user=options.min_interactions,
									 kc_col_name="KC(Default)",
									 remove_nan_skills=options.remove_nan_skills,
									 verbose=options.verbose)
	elif options.dataset == "robomission":
		df, Q_mat = prepare_robomission(min_interactions_per_user=options.min_interactions,
										verbose=options.verbose)
