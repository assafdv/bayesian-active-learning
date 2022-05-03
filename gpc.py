import logging
import numpy as np
from scipy import stats
import GPy

logging.getLogger().setLevel('INFO')


class GPCTrainer:
    """ Gaussian Process Classification Model """
    def __init__(self,
                 random_state=0):
        """
        init GPy model.
        :param random_state: float, random state.
        """
        self.random_state = random_state
        self.model = None

    def create_model(self, X, y, kern_params, verbose=False):
        """
        create GPy model
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples, 1], float). Targets.
        :param kern_params: dictionary with kernel parameters
        :return: GPy model object.
        """
        # create kernel
        kern = self.create_kernel_obj(**kern_params)

        # create GPC model
        self.model = GPy.models.GPClassification(X, y, kernel=kern)
        if verbose:
            assert(self.model.checkgrad(verbose=True))
            logging.info(self.model)

    @staticmethod
    def create_kernel_obj(kern_func, kern_name, active_dims, params):
        """
        create GPy kernel object.
        :param kern_func: string, kernel function (e.g. rbf)
        :param kern_name: string, kernel name.
        :param active_dims: array (shape=[nb_dims,_], int). active dimensions for slicing.
        :param params: dict, kernel params.
        :return: GPy kernel object
        """
        nb_dims = active_dims.size
        if kern_func == 'linear':
            var = params.get('variance', 1)
            ard = params.get('ard', False)
            if ard:
                variances = np.ones(nb_dims)*var
            else:
                variances = var
            logging.info("Create Linear kernel. dims = {}, variance = {}, ARD = {}.".format(nb_dims, var, ard))
            kern = GPy.kern.Linear(nb_dims, active_dims=active_dims, variances=variances, ARD=ard, name=kern_name)
        elif kern_func == 'rbf':
            var = params.get('variance', 1)
            ls = params.get('lengthscale', 1)
            ard = params.get('ard', False)
            fix_var = params.get('fix_variance', False)
            if ard:
                lengthscale = np.ones(nb_dims)*ls
            else:
                lengthscale = ls
            logging.info("Create RBF kernel. dims = {}, variance = {}, lengthscale = {}, ARD = {}."
                         .format(nb_dims, var, ls, ard))
            kern = GPy.kern.RBF(nb_dims,
                                active_dims=active_dims,
                                variance=var,
                                lengthscale=lengthscale,
                                ARD=ard,
                                name=kern_name)
            if fix_var:
                kern.variance.fix()
        return kern

    def train(self, X, y, kern_params, verbose=False):
        """
        train model
        :param X: array (shape=[n_samples, n_dims], float). Features.
        :param y: array (shape=[n_samples,_], float). Targets.
        :param kern_params: dictionary with kernel parameters
        :param verbose: if True, log model params.
        dimensions.
        """
        # create model
        logging.info('Create new GPy model')
        self.create_model(X, y, kern_params, verbose)

        # optimize model
        self.optimize_model(verbose)

    def optimize_model(self, n_iter=3, max_iter=100, verbose=False):
        """
        model adaptation in Bayesian fashion. Find optimal hyper-parameters
        that maximize the marginal likelihood.
        :param n_iter: int, number of parameters optimization + posterior approximation runs.
        :param max_iter: int, number of iteration for parameters optimization.
        :param verbose: bool, if True, log model params after every optimization iteration
        """
        for i in range(n_iter):
            self.model.optimize(optimizer="bfgs",
                                max_iters=max_iter)
            if verbose:
                logging.info(f"iteration: {i}")
                logging.info(self.model)
                logging.info("")

    def predict_gp_posterior_stats(self, Xnew):
        """
        compute the mean and variance of the posterior predictive of the GP given input samples X, p(f*|X*, D).
        :param Xnew:  array=[n_samples,n_dim], float. Input samples
        :return: tuple (mean, var). where mean is array (shape=[n_samples,1], float) with the mean values of p(f*|X*, D).
         var is array (shape=[n_samples,1], float) with the variance of p(f*|X*, D)
        """
        return self.model.predict_noiseless(Xnew)

    def predict(self, X):
        """
        predict
        :param X: array (shape=[n_samples, n_dims], float). input samples.
        :return: array (shape=[n_sample,_], float). poseterior predictive probabilities.
        """
        # compute mean and variance of the posterior
        mu_star, var_star = self.predict_gp_posterior_stats(X)

        # compute the posterior predicitive probability of y (p(y=1|D,x))
        y_prob = self.posterior_predictive_y(mu_star, var_star)
        return y_prob

    def posterior_predictive_y(self, mu_star, var_star, Y_metadata=None):
        """
        compute the posterior predictive distribution
        :param mu_star: array(shape=[n_sample,_], float ). mean values of the posterior predictive of f.
        :param var_star: array(shape=[n_sample,_], float ). variance values of the posterior predictive of f.
        :param Y_metadata: Y_metadata which is not used.
        :return: array (shape=[n_sample,_], float). posterior predictive for the target values
        """
        arg = mu_star/np.sqrt(1+var_star)
        p = stats.norm.cdf(arg)
        return p

    def compute_entropy_binary(self, p):
        """
        compute the entropy for the binary case.
        :param p: (array=[n,_], float). probabilities
        :return: (array=[n_,], float). entropies
        """
        # compute entropy
        p1 = p
        log_p1 = np.log(p1, out=np.zeros_like(p1), where=p1 != 0)

        p0 = 1 - p
        log_p0 = np.log(p0, out=np.zeros_like(p0), where=p0 != 0) 
        entropy = -p1*log_p1 - p0*log_p0
        return entropy

    def compute_entopy_y(self, X):
        """
        compute the entropy of y_i using the the posterior predictive distribution p(y_i|x_i,D).
        :param X: (array=[n_samples,n_dim], float). Input samples
        :return: (array=[n_samples,_], float). Entropy of y_i.
        """
        # compute the posterior predictive p(y_i|x_i,D)
        prob = self.predict(X)

        # compute entropy
        entropy = self.compute_entropy_binary(prob)

        return entropy

    def compute_entropy_y_cond_f(self, f):
        """
        compute the entropy of y given the latent varaible f (GP)
        :param f: array (shape=[n,1], float).  GP values.
        :return: array (shape=[n,1], float). Entropy values.
        """
        # compute  p(y=1|f_i, X_i).
        p_y_f = stats.norm.cdf(f)

        # compute entropies
        entropy = self.compute_entropy_binary(p_y_f)
        return entropy

    def compute_exected_entropy_y_cond_f(self, X, method='riemann_sum', n_samples=5000):
        """
        compute the expectation of the entropy of y given the latent f (GP). 
        The expectation is taken w.r.t to the latent GP f and is computed using numerical integration.
        :param X: (array=[Nx,n_dim], float). Input samples
        :param method: string, method used to approximate the expectation. Support values are 'sampling' and
        'riemann_sum'.
        :param n_samples: number of sampling points.
        :return: array (shape=[Nx,_], float). Expected entropy.
        """
        # init
        expetced_entropy = 0
        if method == 'sampling':
            # sample from the posterior predictive of f
            f = self.model.posterior_samples_f(X, size=n_samples).squeeze()  # [Nx, n_samples]
            # approximate the expectation by sum (SLLN)
            expetced_entropy = 0
            for n in range(n_samples):  # loop over f samples
                f_n = f[:, n].reshape(-1, 1)
                expetced_entropy += (1 / n_samples) * self.compute_entropy_y_cond_f(f_n)
        elif method == 'riemann_sum':
            # compute mean and variance of the posterior
            mean_f, var_f = self.predict_gp_posterior_stats(X)
            # approximate the expectation by Riemann sum
            std = 3
            z, delta_z = np.linspace(-std, std, n_samples, retstep=True)
            for n in range(n_samples):
                arg = np.sqrt(var_f)*z[n] + mean_f  # change of variables
                pdf_z = (1 / np.sqrt(2*np.pi))*np.exp(-0.5*z[n]**2)
                expetced_entropy += delta_z*pdf_z*self.compute_entropy_y_cond_f(arg)
        else:
            raise ValueError("unsupported method")
        return expetced_entropy

    def compute_informativeness(self, X):
        """
        compute the informativeness of a data sample X according to Bayesian Active Learning by Disagreement (BALD)
        criterion, Thus can be used as a Active Learning query criteria in Gaussian Process models as described in:
        Houlsby, Neil, et al. "Bayesian active learning for classification and preference learning." arXiv
         preprint arXiv:1112.5745(2011.
        :param X: (array=[n_samples,n_dim], float). Input samples
        :return: (array=[n_samples,_], float). informativeness for each input sample.
        """
        # compute entropy of y
        entropy_y = self.compute_entopy_y(X)

        # the expectation (w.r.t to the posterior predictive of f) of the entropy of y given f
        expected_entropy_y_f = self.compute_exected_entropy_y_cond_f(X)

        # information criterion
        informativeness = entropy_y - expected_entropy_y_f
        return informativeness
