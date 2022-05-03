import unittest
import logging
from gpc import GPCTrainer
import numpy as np
import GPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.getLogger().setLevel("INFO")


class GPCTrainerTests(unittest.TestCase):
    """ GPC Trainer Unit Test"""
    def setUp(self):
        self.random_state = 42
        self.n_dims = 1
        self.n_samples = 1000
        np.random.seed(self.random_state)
        self.model = GPCTrainer(self.random_state)

    def generate_data(self):
        # generate rando inputs
        X = np.random.rand(self.n_samples, self.n_dims)
        # draw the latent function value
        k = GPy.kern.RBF(self.n_dims, variance=7., lengthscale=0.2)
        f = np.random.multivariate_normal(mean=np.zeros(self.n_samples),
                                          cov=k.K(X))
        # generate samples from p(y|f)
        lik = GPy.likelihoods.Bernoulli()
        Y = lik.samples(f).reshape(-1, 1)
        return X, Y

    def get_kernel_params(self):
        """ set kernel parameters """
        # set kernel
        k1 = dict()
        k1.setdefault('active_dims', np.arange(self.n_dims))
        k1.setdefault('kern_func', 'rbf')
        k1.setdefault('kern_name', 'radial basis function')
        k1.setdefault('params', {})
        k1['params'].setdefault('ard', False)
        k1['params'].setdefault('variance', 1)
        k1['params'].setdefault('lengthscale', 1)
        return k1

    def test_gpc(self):
        """ Test GPC on simulated data """
        # generate data
        X, y = self.generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=int(self.n_samples*0.7),
                                                            random_state=self.random_state)
        # define kernel params
        kern_params = self.get_kernel_params()
        # train model
        self.model.train(X_train, y_train, kern_params, verbose=True)
        # predict
        y_prob = self.model.predict(X_test)
        y_pred = np.zeros_like(y_prob)
        y_pred[y_prob > 0.5] = 1
        # evaluate
        logging.info(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(acc, 0.85)
