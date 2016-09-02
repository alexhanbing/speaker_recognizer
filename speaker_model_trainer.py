import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib

class SpeakerModelTrainer(object):
    def __init__(self, model_name='GMMHMM', n_components=4, n_mix=4, n_iter=1000, cov_type='diag'):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.n_mix = n_mix

        if self.model_name == 'GuassianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            self.model = hmm.GMMHMM(n_components=self.n_components,
                covariance_type=self.cov_type, n_mix=self.n_mix, n_iter=self.n_iter)

    def fit(self, X):
        np.seterr(all='ignore')
        self.model.fit(X)

    def score(self, input_data):
        return self.model.score(input_data)

    def get_monitor(self):
        return self.model.monitor_

    def get_converged(self):
        return self.model.monitor_.converged

    def save_model(self, file_name):
        joblib.dump(self.model, file_name)

    def load_model(self, file_name):
        self.model = joblib.load(file_name)
