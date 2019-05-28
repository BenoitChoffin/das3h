from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz, hstack, csr_matrix
import pandas as pd
import pywFM
import argparse
import numpy as np
import os
import sys
import dataio
import json

# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = os.path.join(os.path.dirname(__file__),
                                        'libfm/bin/')

parser = argparse.ArgumentParser(description='Run DAS3H.')
parser.add_argument('X_file', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='assistments12')
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--iter', type=int, nargs='?', default=300)
parser.add_argument('--users', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--items', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--wins', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--fails', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--attempts', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw_kc', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tw_items', type=bool, nargs='?', const=True, default=False)
options = parser.parse_args()

experiment_args = vars(options)
DATASET_NAME = options.dataset
CSV_FOLDER = dataio.build_new_paths(DATASET_NAME)

# Build legend
short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(experiment_args)

EXPERIMENT_FOLDER = os.path.join(CSV_FOLDER, "results", short_legend)
dataio.prepare_folder(EXPERIMENT_FOLDER)
maxRuns = 5
for run_id in range(maxRuns):
	dataio.prepare_folder(os.path.join(EXPERIMENT_FOLDER, str(run_id)))

# Load sparsely encoded datasets
X = csr_matrix(load_npz(options.X_file))
all_users = np.unique(X[:,0].toarray().flatten())
y = X[:,3].toarray().flatten()
qmat = load_npz(os.path.join(CSV_FOLDER, "q_mat.npz"))

# FM parameters
params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}

# Student-level train-test split
kf = KFold(n_splits=5, shuffle=True)
splits = kf.split(all_users)

for run_id, (i_user_train, i_user_test) in enumerate(splits):
	users_train = all_users[i_user_train]
	users_test = all_users[i_user_test]

	X_train = X[np.where(np.isin(X[:,0].toarray().flatten(),users_train))]
	y_train = X_train[:,3].toarray().flatten()
	X_test = X[np.where(np.isin(X[:,0].toarray().flatten(),users_test))]
	y_test = X_test[:,3].toarray().flatten()

	if options.d == 0:
		print('fitting...')
		model = LogisticRegression(solver="lbfgs", max_iter=400)
		model.fit(X_train[:,5:], y_train) # the 5 first columns are the non-sparse dataset
		y_pred_test = model.predict_proba(X_test[:,5:])[:, 1]

	else:
		fm = pywFM.FM(**params)
		model = fm.run(X_train[:,5:], y_train, X_test[:,5:], y_test)
		y_pred_test = np.array(model.predictions)
		model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'rlog.csv'))

	print(y_test)
	print(y_pred_test)
	ACC = accuracy_score(y_test, np.round(y_pred_test))
	AUC = roc_auc_score(y_test, y_pred_test)
	print('auc', AUC)
	NLL = log_loss(y_test, y_pred_test)
	print('nll', NLL)

	# Save experimental results
	with open(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'results.json'), 'w') as f:
		f.write(json.dumps({
			'args': experiment_args,
			'legends': {
				'short': short_legend,
				'full': full_legend,
				'latex': latex_legend
			},
			'metrics': {
				'ACC': ACC,
				'AUC': AUC,
				'NLL': NLL
			}
		}, indent=4))

