import unittest
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from encode import df_to_sparse

class EncodeTestCase(unittest.TestCase):
	def setUp(self):
		self.data = pd.read_csv("data/dummy/preprocessed_data.csv")
		self.q_mat = load_npz("data/dummy/q_mat.npz").toarray()

	def test_ui(self):
		# Test IRT/MIRT encoding
		X_ui = df_to_sparse(self.data, self.q_mat, ["users", "items"]).toarray()
		# Sort array
		X_ui = X_ui[X_ui[:,4].argsort(),5:] # Collect only sparse columns
		irt_features = np.array(pd.read_csv("data/dummy/irt.csv", sep=';'))
		self.assertSequenceEqual(X_ui.tolist(), irt_features.tolist(),
			"Inconsistent IRT features")

	def test_afm(self):
		# Test AFM encoding
		X_afm = df_to_sparse(self.data, self.q_mat, ["skills", "attempts"]).toarray()
		# Sort array
		X_afm = X_afm[X_afm[:,4].argsort(),5:] # Collect only sparse columns
		afm_features = np.array(pd.read_csv("data/dummy/afm.csv", sep=';'))
		self.assertSequenceEqual(X_afm.tolist(), afm_features.tolist(),
			"Inconsistent AFM features")

	def test_pfa(self):
		# Test PFA encoding
		X_pfa = df_to_sparse(self.data, self.q_mat, ["skills", "wins", "fails"]).toarray()
		# Sort array
		X_pfa = X_pfa[X_pfa[:,4].argsort(),5:] # Collect only sparse columns
		pfa_features = np.array(pd.read_csv("data/dummy/pfa.csv", sep=';'))
		self.assertSequenceEqual(X_pfa.tolist(), pfa_features.tolist(),
			"Inconsistent PFA features")

	def test_dash(self):
		# Test DASH encoding
		X_uiwat2 = df_to_sparse(self.data, self.q_mat, ["users", "items", "wins", "attempts"], tw="tw_items").toarray()
		# Sort array
		X_uiwat2 = X_uiwat2[X_uiwat2[:,4].argsort(),5:] # Collect only sparse columns
		# Convert to simple counters to avoid using assertAlmostEqual and floats
		X_uiwat2[:,-10:] = np.exp(X_uiwat2[:,-10:])-1

		dash_features = np.array(pd.read_csv("data/dummy/dash.csv", sep=';'))
		self.assertSequenceEqual(X_uiwat2.tolist(), dash_features.tolist(),
			"Inconsistent DASH features")

	def test_das3h(self):
		# Test DAS3H encoding
		X_uiswat1 = df_to_sparse(self.data, self.q_mat, ["users", "items", "skills", "wins", "attempts"],
			tw="tw_kc").toarray()
		# Sort array
		X_uiswat1 = X_uiswat1[X_uiswat1[:,4].argsort(),5:] # Collect only sparse columns
		# Convert to simple counters to avoid using assertAlmostEqual and floats
		X_uiswat1[:,-30:] = np.exp(X_uiswat1[:,-30:])-1
		X_uiswat1 = X_uiswat1.astype(np.int64)

		das3h_features = np.array(pd.read_csv("data/dummy/das3h.csv", sep=';'))
		self.assertSequenceEqual(X_uiswat1.tolist(), das3h_features.tolist(),
			"Inconsistent DAS3H features")

if __name__ == '__main__':
	unittest.main()


