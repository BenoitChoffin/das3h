import os

def prepare_folder(path):
	"""Create folder from path."""
	if not os.path.isdir(path):
		os.makedirs(path)

def build_new_paths(DATASET_NAME):
	"""Create dataset folder path name."""
	DATA_FOLDER = './data'
	CSV_FOLDER = os.path.join(DATA_FOLDER, DATASET_NAME)
	return CSV_FOLDER

def get_legend(experiment_args):
	"""Generate legend for an experiment.

	Argument:
	experiment_args -- experiment arguments (from ArgumentParser)

	Outputs:
	short -- short legend (str)
	full -- full legend (str)
	latex -- latex legend (str)
	active -- list of active variables
	"""
	dim = experiment_args['d']
	short = ''
	full = ''
	agents = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
	active = []
	for agent in agents:
		if experiment_args.get(agent):
			short += agent[0]
			active.append(agent)
	if experiment_args.get('tw_kc'):
		short += 't1'
		active.append("tw_kc")
	elif experiment_args.get('tw_items'):
		short += 't2'
		active.append("tw_items")
	short += '_' # add embedding dimension after underscore
	short += str(dim)
	prefix = ''
	if set(active) == {'users', 'items'} and dim == 0:
		prefix = 'IRT: '
	elif set(active) == {'users', 'items'} and dim > 0:
		prefix = 'MIRTb: '
	elif set(active) == {'skills', 'attempts'}:
		prefix = 'AFM: '
	elif set(active) == {'skills', 'wins', 'fails'}:
		prefix = 'PFA: '
	elif set(active) == {'users', 'items', 'skills', 'wins', 'attempts', 'tw_kc'}:
		prefix = 'DAS3H: '
	elif set(active) == {'users', 'items', 'wins', 'attempts', 'tw_items'}:
		prefix = 'DASH: '
	full = prefix + ', '.join(active) + ' d = {:d}'.format(dim)
	latex = prefix + ', '.join(active)
	return short, full, latex, active

