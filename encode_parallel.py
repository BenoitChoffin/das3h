import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
from scipy import sparse
import argparse
import os
from tqdm import tqdm
import time
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Encode datasets.')
parser.add_argument('--dataset', type=str, nargs='?', default='assistments12')
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--attempts', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw_kc', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw_items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--log_counts', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

NB_OF_TIME_WINDOWS = 5

def encode_single_student(df, stud_id, Q_mat, active_features, NB_OF_TIME_WINDOWS, q, dict_q_mat, tw,
						  wf_counters, log_counts, X):
	df_stud = df[df["user_id"]==stud_id][["user_id", "item_id", "timestamp", "correct"]].copy()
	df_stud_indices = np.array(df_stud.index).reshape(-1,1)
	df_stud.sort_values(by="timestamp", inplace=True) # Sort values
	df_stud = np.array(df_stud)
	X['df'] = np.hstack((df_stud[:,[0,1,3]], df_stud_indices))

	skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
	if 'skills' in active_features:
		X['skills'] = sparse.csr_matrix(skills_temp)
	if "attempts" in active_features:
		if tw == "tw_kc":
			last_t = -1 ; list_of_skills = [] # in case multiple rows with the same timestamp
			attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*Q_mat.shape[1]))
			for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
				if (last_t != t) & (len(list_of_skills) > 0):
					for skill_id in list_of_skills:
						q[stud_id, skill_id].push(t)
					list_of_skills = []
				for skill_id in dict_q_mat[item_id]:
					attempts[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
						np.array(q[stud_id, skill_id].get_counters(t)))
					if last_t != t:
						q[stud_id, skill_id].push(t)
					else:
						list_of_skills.append(skill_id)
				last_t = t
		elif tw == "tw_items":
			last_t = -1 ; list_of_items = [] # in case multiple rows with the same timestamp
			attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
			for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
				if (last_t != t) & (len(list_of_items) > 0):
					for item in list_of_items:
						q[stud_id, item].push(t)
					list_of_items = []
				attempts[l] = np.log(1 + np.array(q[stud_id, item_id].get_counters(t)))
				if last_t != t:
					q[stud_id, item_id].push(t)
				else:
					list_of_items.append(item_id)
				last_t = t
		else:
			last_t = -1 ; list_of_skills = [] # in case multiple rows with the same timestamp
			attempts = np.zeros((df_stud.shape[0], Q_mat.shape[1]))
			for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
				if (last_t != t) & (len(list_of_skills) > 0):
					for skill_id in list_of_skills:
						wf_counters[stud_id, skill_id] += 1
					list_of_skills = []
				for skill_id in dict_q_mat[item_id]:
					if log_counts:
						attempts[l, skill_id] = np.log(1 + wf_counters[stud_id, skill_id])
					else:
						attempts[l, skill_id] = wf_counters[stud_id, skill_id]
					if last_t != t:
						wf_counters[stud_id, skill_id] += 1
					else:
						list_of_skills.append(skill_id)
				last_t = t
			#attempts = np.multiply(np.cumsum(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),0)[:-1],skills_temp)
		X['attempts'] = sparse.csr_matrix(attempts)
	if "wins" in active_features:
		#skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
		if tw == "tw_kc":
			last_t = -1 ; list_of_skills = [] # in case multiple rows with the same timestamp
			wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*Q_mat.shape[1]))
			for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
				if (last_t != t) & (len(list_of_skills) > 0):
					for skill_id in list_of_skills:
						q[stud_id, skill_id, "correct"].push(t)
					list_of_skills = []
				for skill_id in dict_q_mat[item_id]:
					wins[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
						np.array(q[stud_id, skill_id, "correct"].get_counters(t)))
					if correct:
						if last_t != t:
							q[stud_id, skill_id, "correct"].push(t)
						else:
							list_of_skills.append(skill_id)
				last_t = t
		elif tw == "tw_items":
			last_t = -1 ; list_of_items = [] # in case multiple rows with the same timestamp
			wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
			for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
				if (last_t != t) & (len(list_of_items) > 0):
					for item in list_of_items:
						q[stud_id, item].push(t)
					list_of_items = []
				wins[l] = np.log(1 + np.array(q[stud_id, item_id, "correct"].get_counters(t)))
				if correct:
					if last_t != t:
						q[stud_id, item_id, "correct"].push(t)
					else:
						list_of_items.append(item_id)
				last_t = t
		else:
			last_t = -1 ; list_of_skills = [] # in case multiple rows with the same timestamp
			wins = np.zeros((df_stud.shape[0], Q_mat.shape[1]))
			for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
				if (last_t != t) & (len(list_of_skills) > 0):
					for skill_id in list_of_skills:
						wf_counters[stud_id, skill_id, "correct"] += 1
					list_of_skills = []
				for skill_id in dict_q_mat[item_id]:
					if log_counts:
						wins[l, skill_id] = np.log(1 + wf_counters[stud_id, skill_id, "correct"])
					else:
						wins[l, skill_id] = wf_counters[stud_id, skill_id, "correct"]
					if correct:
						if last_t != t:
							wf_counters[stud_id, skill_id, "correct"] += 1
						else:
							list_of_skills.append(skill_id)
				last_t = t
			#wins = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),
			#	np.hstack((np.array([0]),df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp)
		X['wins'] = sparse.csr_matrix(wins)
	if "fails" in active_features:
		last_t = -1 ; list_of_skills = [] # in case multiple rows with the same timestamp
		fails = np.zeros((df_stud.shape[0], Q_mat.shape[1]))
		for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
			if (last_t != t) & (len(list_of_skills) > 0):
				for skill_id in list_of_skills:
					wf_counters[stud_id, skill_id, "incorrect"] += 1
				list_of_skills = []
			for skill_id in dict_q_mat[item_id]:
				fails[l, skill_id] = wf_counters[stud_id, skill_id, "incorrect"]
				if not correct:
					if last_t != t:
						wf_counters[stud_id, skill_id, "incorrect"] += 1
					else:
						list_of_skills.append(skill_id)
			last_t = t
		#skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
		#fails = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),
		#	np.hstack((np.array([0]),1-df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp)
		X["fails"] = sparse.csr_matrix(fails)
	#sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),
	#	sparse.hstack([X[agent] for agent in active_features if agent not in ["users","items"]])]).tocsr()
	#return sparse_df
	return X

def df_to_sparse(df, Q_mat, active_features, tw=None, skip_sucessive=True, log_counts=False):
	"""Build sparse features dataset from dense dataset and q-matrix.

	Arguments:
	df -- dense dataset, output from one function from prepare_data.py (pandas DataFrame)
	Q_mat -- q-matrix, output from one function from prepare_data.py (sparse array)
	active_features -- features used to build the dataset (list of strings)
	tw -- useful when script is *not* called from command line.

	Output:
	sparse_df -- sparse dataset. The 5 first columns of sparse_df are just the same columns as in df.

	Notes:
	* tw_kc and tw_items respectively encode time windows features instead of regular counter features
	  at the skill and at the item level for wins and attempts, as decribed in our paper. As a consequence,
	  these arguments can only be used along with the wins and/or attempts arguments. With tw_kc, one column
	  per time window x skill is encoded, whereas with tw_items, one column per time window is encoded (it is
	  assumed that items share the same time window biases).
	"""

	# Transform q-matrix into dictionary
	dt = time.time()
	dict_q_mat = {i:set() for i in range(Q_mat.shape[0])}
	for elt in np.argwhere(Q_mat == 1):
		dict_q_mat[elt[0]].add(elt[1])

	X={}
	if 'skills' in active_features:
		X["skills"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
	if 'attempts' in active_features:
		if tw == "tw_kc":
			X["attempts"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1]*NB_OF_TIME_WINDOWS)))
		elif tw == "tw_items":
			X["attempts"] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		else:
			X["attempts"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
	if 'wins' in active_features:
		if tw == "tw_kc":
			X["wins"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1]*NB_OF_TIME_WINDOWS)))
		elif tw == "tw_items":
			X["wins"] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		else:
			X["wins"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))
	if 'fails' in active_features:
		X["fails"] = sparse.csr_matrix(np.empty((0, Q_mat.shape[1])))

	X['df'] = np.empty((0,4)) # Keep only track of line index + user/item id + correctness

	q = defaultdict(lambda: OurQueue())  # Prepare counters for time windows
	wf_counters = defaultdict(lambda: 0)
	if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
		res = Parallel(n_jobs=-1,verbose=10)(delayed(encode_single_student)(df, stud_id, Q_mat, active_features, NB_OF_TIME_WINDOWS, q, dict_q_mat, tw,
			wf_counters, log_counts, X) for stud_id in df["user_id"].unique())
		for X_stud in res:
			for key in X_stud.keys():
				if key == "df":
					X[key] = np.vstack((X[key],X_stud[key]))
				else:
					X[key] = sparse.vstack([X[key],X_stud[key]]).tocsr()
		#sparse_df = sparse.vstack([sparse.csr_matrix(X_stud) for X_stud in res]).tocsr() #df["correct"].values.reshape(-1,1)),
		#		sparse.hstack([X[agent] for agent in active_features])]).tocsr()
		#sparse_df = sparse_df[np.argsort(sparse_df[:,3])] # sort matrix by original index
		#X_df = sparse_df[:,:5]
		#sparse_df = sparse_df[:,5:]
	onehot = OneHotEncoder()
	if 'users' in active_features:
		if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
			X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
		else:
			X['users'] = onehot.fit_transform(df["user_id"].values.reshape(-1,1))
	if 'items' in active_features:
		if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
			X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
		else:
			X['items'] = onehot.fit_transform(df["item_id"].values.reshape(-1,1))
	if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
		sparse_df = sparse.hstack([sparse.csr_matrix(X['df'])[:,-2].reshape(-1,1),
			sparse.hstack([X[agent] for agent in active_features])]).tocsr()
		#sparse_df = sparse_df[np.argsort(sparse.csr_matrix(X["df"])[:,-1])] # sort matrix by original index
		sparse_df = sparse_df[np.argsort(X["df"][:,-1])] # sort matrix by original index
	else:
		sparse_df = sparse.hstack([sparse.csr_matrix(df["correct"].values.reshape(-1,1)),
			sparse.hstack([X[agent] for agent in active_features])]).tocsr()
		# No need to sort sparse matrix here
	print("Preprocessed data in: ", time.time()-dt)
	#return sparse_df
	#if 'users' in active_features:
	#	if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
	#		sparse_df = sparse.hstack([onehot.fit_transform(X_df[:,0].reshape(-1,1))])
	#	else:
	#		X_users = onehot.fit_transform(df["user_id"].values.reshape(-1,1))
	#if 'items' in active_features:
	#	if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
	#		X_items = onehot.fit_transform(X_df[:,1].reshape(-1,1))
	#	else:
	#		X_items = onehot.fit_transform(df["item_id"].values.reshape(-1,1))
	#if len(set(active_features).intersection({"skills","attempts","wins","fails"})) > 0:
	#	sparse_df = sparse.hstack([])
	#	sparse_df = sparse.hstack([sparse.csr_matrix(X['df'][:,-2].reshape(-1,1)),
	#		sparse.hstack([X[agent] for agent in active_features])]).tocsr()
	#	sparse_df = sparse_df[np.argsort(X["df"][:,-1])] # sort matrix by original index
	#else:
	#	sparse_df = sparse.hstack([sparse.csr_matrix(df["correct"].values.reshape(-1,1)),
	#		sparse.hstack([X[agent] for agent in active_features])]).tocsr()
		# No need to sort sparse matrix here
	#print("Preprocessed data in: ", time.time()-dt)
	return sparse_df

if __name__ == "__main__":
	dt = time.time()
	os.chdir("data/"+options.dataset)
	all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
	active_features = [features for features in all_features if vars(options)[features]]
	features_suffix = ''.join([features[0] for features in active_features])
	if vars(options)["tw_kc"]:
		features_suffix += 't1'
		tw = "tw_kc"
	elif vars(options)["tw_items"]:
		features_suffix += 't2'
		tw = "tw_items"
	elif vars(options)["log_counts"]:
		features_suffix += 'l'
		tw = None
	else:
		tw = None

	df = pd.read_csv('preprocessed_data.csv')
	qmat = sparse.load_npz('q_mat.npz').toarray()
	print('Loading data:', df.shape[0], 'samples in ', time.time() - dt, "seconds")
	X  = df_to_sparse(df, qmat, active_features, tw=tw, log_counts=options.log_counts)
	sparse.save_npz('X-{:s}.npz'.format(features_suffix), X)
