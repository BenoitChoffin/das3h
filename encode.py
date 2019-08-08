import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
from scipy import sparse
import argparse
import os
from utils.this_queue import OurQueue

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
options = parser.parse_args()

NB_OF_TIME_WINDOWS = 5

def df_to_sparse(df, Q_mat, active_features, tw=None, verbose=True):
	"""Build sparse features dataset from dense dataset and q-matrix.

	Arguments:
	df -- dense dataset, output from one function from prepare_data.py (pandas DataFrame)
	Q_mat -- q-matrix, output from one function from prepare_data.py (sparse array)
	active_features -- features used to build the dataset (list of strings)
	verbose -- if True, print information on the encoding process (bool)

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

	X['df'] = np.empty((0,5)) # Keep track of the original dataset

	q = defaultdict(lambda: OurQueue())  # Prepare counters for time windows
	for stud_id in df["user_id"].unique():
		df_stud = df[df["user_id"]==stud_id][["user_id", "item_id", "timestamp", "correct", "inter_id"]].copy()
		df_stud.sort_values(by="timestamp", inplace=True) # Sort values 
		df_stud = np.array(df_stud)
		X['df'] = np.vstack((X['df'], df_stud))

		if 'skills' in active_features:
			skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
			X['skills'] = sparse.vstack([X["skills"],sparse.csr_matrix(skills_temp)])
		if "attempts" in active_features:
			skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
			if tw == "tw_kc":
				attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*Q_mat.shape[1]))
				for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
					for skill_id in dict_q_mat[item_id]:
						attempts[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
							np.array(q[stud_id, skill_id].get_counters(t)))
					q[stud_id, skill_id].push(t)
				#attempts = np.empty((df_stud.shape[0],0))
				#for l in LIST_OF_BOUNDARIES:
				#	attempts_temp = np.zeros((df_stud.shape[0],Q_mat.shape[1])) # a_sw array
				#	for i in range(1,attempts_temp.shape[0]): # 1st line is always full of zeros
				#		list_of_indices = np.where(df_stud[i,2] - df_stud[:i,2] < l)
				#		skills_temp = Q_mat[df_stud[:i,1].astype(int)][list_of_indices]
				#		attempts_temp[i] = np.sum(skills_temp,0)
				#	skills = Q_mat[df_stud[:,1].astype(int)]
				#	attempts_temp = np.log(1+np.multiply(attempts_temp,skills)) # only keep KCs involved
				#	attempts = np.hstack((attempts,attempts_temp))
			elif tw == "tw_items":
				attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
				for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
					attempts[l] = np.log(1 + np.array(q[stud_id, item_id].get_counters(t)))
					q[stud_id, item_id].push(t)
				#attempts = np.empty((df_stud.shape[0],0))
				#for l in LIST_OF_BOUNDARIES:
				#	attempts_temp = np.zeros(df_stud.shape[0]) # a_sw array
				#	for i in range(1,attempts_temp.shape[0]): # 1st line is always full of zeros
				#		list_of_indices = np.where((df_stud[i,2] - df_stud[:i,2] < l) & (df_stud[i,1] == df_stud[:i,1]))
				#		attempts_temp[i] = len(list_of_indices[0])
				#	attempts_temp = np.log(1+attempts_temp)
				#	attempts = np.hstack((attempts,attempts_temp.reshape(-1,1)))
			else:
				attempts = np.multiply(np.cumsum(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),0)[:-1],skills_temp)
			X['attempts'] = sparse.vstack([X['attempts'],sparse.csr_matrix(attempts)])
		if "wins" in active_features:
			skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
			if tw == "tw_kc":
				wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*Q_mat.shape[1]))
				for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
					for skill_id in dict_q_mat[item_id]:
						wins[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
							np.array(q[stud_id, skill_id, "correct"].get_counters(t)))
					if correct:
						q[stud_id, skill_id, "correct"].push(t)
				#wins = np.empty((df_stud.shape[0],0))
				#for l in LIST_OF_BOUNDARIES:
				#	wins_temp = np.zeros((df_stud.shape[0],Q_mat.shape[1])) # c_sw array
				#	for i in range(1,wins_temp.shape[0]): # 1st line is always full of zeros
				#		list_of_indices = np.where(df_stud[i,2] - df_stud[:i,2] < l)
				#		skills_temp = Q_mat[df_stud[:i,1].astype(int)][list_of_indices]
				#		wins_temp[i] = np.sum(np.multiply(skills_temp,df_stud[:i,3][list_of_indices].reshape(-1,1)),0)
				#	skills = Q_mat[df_stud[:,1].astype(int)]
				#	wins_temp = np.log(1+np.multiply(wins_temp,skills)) # only keep KCs involved
				#	wins = np.hstack((wins,wins_temp))
			elif tw == "tw_items":
				wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
				for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
					wins[l] = np.log(1 + np.array(q[stud_id, item_id, "correct"].get_counters(t)))
					if correct:
						q[stud_id, item_id, "correct"].push(t)
				#wins = np.empty((df_stud.shape[0],0))
				#for l in LIST_OF_BOUNDARIES:
				#	wins_temp = np.zeros(df_stud.shape[0]) # c_sw array
				#	for i in range(1,wins_temp.shape[0]): # 1st line is always full of zeros
				#		list_of_indices = np.where((df_stud[i,2] - df_stud[:i,2] < l) & (df_stud[i,1] == df_stud[:i,1]))
				#		wins_temp[i] = np.log(1+np.sum(df_stud[:i,3][list_of_indices]))
				#	wins = np.hstack((wins,wins_temp.reshape(-1,1)))
			else:
				wins = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),
					np.hstack((np.array([0]),df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp)
			X['wins'] = sparse.vstack([X['wins'],sparse.csr_matrix(wins)])
		if "fails" in active_features:
			skills_temp = Q_mat[df_stud[:,1].astype(int)].copy()
			fails = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),
				np.hstack((np.array([0]),1-df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp)
			X["fails"] = sparse.vstack([X["fails"],sparse.csr_matrix(fails)])
		if verbose:
			print(X["df"].shape)

	#onehot = OneHotEncoder(categories="auto")
	onehot = OneHotEncoder()
	if 'users' in active_features:
		X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
		if verbose:
			print("Users encoded.")
	if 'items' in active_features:
		X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
		if verbose:
			print("Items encoded.")
	sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X[agent] for agent in active_features])]).tocsr()
	return sparse_df

if __name__ == "__main__":
	os.chdir(os.path.join('data', options.dataset))
	all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
	active_features = [features for features in all_features if vars(options)[features]]
	features_suffix = ''.join([features[0] for features in active_features])
	if vars(options)["tw_kc"]:
		features_suffix += 't1'
		tw = "tw_kc"
	elif vars(options)["tw_items"]:
		features_suffix += 't2'
		tw = "tw_items"
	else:
		tw = None
	#LIST_OF_BOUNDARIES = [1/24, 1, 7, 30, np.inf]

	df = pd.read_csv('preprocessed_data.csv', sep="\t")
	qmat = sparse.load_npz('q_mat.npz').toarray()
	X  = df_to_sparse(df, qmat, active_features, tw=tw)
	sparse.save_npz('X-{:s}.npz'.format(features_suffix), X)

