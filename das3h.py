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

parser = argparse.ArgumentParser(description='Run DAS3H and other student models.')
parser.add_argument('X_file', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='assistments12')
parser.add_argument('--generalization', type=str, nargs='?', default='strongest')
parser.add_argument('--d', type=int, nargs='?')
parser.add_argument('--C', type=float, nargs='?', default=1.)
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
today = datetime.datetime.now() # save date of experiment
DATASET_NAME = options.dataset
CSV_FOLDER = dataio.build_new_paths(DATASET_NAME)

# Build legend
short_legend, full_legend, latex_legend, active_agents = dataio.get_legend(experiment_args)

EXPERIMENT_FOLDER = os.path.join(CSV_FOLDER, options.generalization, "results", short_legend)
dataio.prepare_folder(EXPERIMENT_FOLDER)

# Load sparsely encoded datasets
X = csr_matrix(load_npz(options.X_file))
y = X[:,0].toarray().flatten()
qmat = load_npz(os.path.join(CSV_FOLDER, "q_mat.npz"))

# FM parameters
params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}

for i, folds_file in enumerate(sorted(glob.glob(os.path.join(CSV_FOLDER, options.generalization, "folds/test_fold*.npy")))):
	dataio.prepare_folder(os.path.join(EXPERIMENT_FOLDER, str(i)))
	dt = time.time()
	test_ids = np.load(folds_file)
	train_ids = list(set(range(X.shape[0])) - set(test_ids))
	
	X_train = X[train_ids,1:]
	y_train = y[train_ids]
	X_test = X[test_ids,1:]
	y_test = y[test_ids]

	if options.d == 0:
		print('fitting...')
		estimators = [
			('onehot', MaxAbsScaler()),
			('lr', LogisticRegression(solver="saga", max_iter=options.iter, C=options.C))
		]
		pipe = Pipeline(estimators)
		pipe.fit(X_train, y_train)
		y_pred_test = pipe.predict_proba(X_test)[:, 1]
	else:
		fm = pywFM.FM(**params)
		model = fm.run(X_train, y_train, X_test, y_test)
		y_pred_test = np.array(model.predictions)
		model.rlog.to_csv(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'rlog.csv'))
	
	print(y_test)
	print(y_pred_test)
	ACC = accuracy_score(y_test, np.round(y_pred_test))
	AUC = roc_auc_score(y_test, y_pred_test)
	print('auc', AUC)
	NLL = log_loss(y_test, y_pred_test)
	print('nll', NLL)
	RMSE = np.sqrt(mean_squared_error(y_test,y_pred_test))

	elapsed_time = np.around(time.time() - dt,3)
	# Save experimental results
	with open(os.path.join(EXPERIMENT_FOLDER, str(run_id), 'results.json'), 'w') as f:
		f.write(json.dumps({
			'date': str(today),
			'args': experiment_args,
			'legends': {
				'short': short_legend,
				'full': full_legend,
				'latex': latex_legend
			},
			'metrics': {
				'ACC': ACC,
				'AUC': AUC,
				'NLL': NLL,
				'RMSE': RMSE
			},
			'elapsed_time': elapsed_time
		}, indent=4))
