import hashlib
import unittest
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from gpc import GPCTrainer

logging.getLogger().setLevel('INFO')


class BALDSamplerTests(unittest.TestCase):
    """ BALD Sampler Unit-Test """
    def setUp(self):
        self.random_state = 42
        self.n_dims = 2
        self.n_samples = 400
        self.model = GPCTrainer(random_state=self.random_state)
        np.random.seed(self.random_state)

    @staticmethod
    def generate_X_y_bic(n_samples, n_dims):
        """
        generate 'block in the corner' data set (see ...)
        :param n_samples: int, number of samples.
        :param n_dims: int, number of dimensions
        :return: array (shape=[n_samples,n_dims], float). data set.
        """
        # input validation
        if n_dims != 2:
            raise ValueError("support only 2-dimensions")

        # init
        n_samples_noise = int(0.7*n_samples)
        n_samples_signal = (n_samples - n_samples_noise)
        n1 = int(n_samples_signal/2)
        n2 = n_samples_signal - n1

        # generate noise 'block in corner'
        noise = np.random.uniform(low=(0, 0), high=(0.2, 0.2), size=(n_samples_noise, n_dims))

        # generate signal
        s1 = np.random.uniform(low=(0, 0), high=(0.6, 1), size=(n1, n_dims))
        s2 = np.random.uniform(low=(0.4, 0), high=(1, 1), size=(n2, n_dims))

        X = np.vstack([noise, s1, s2])
        y = np.zeros(n_samples, dtype=int)
        y[-n2:] = 1

        return X, y

    @staticmethod
    def generate_X_y_bim(n_samples, n_dims):
        """
        generate 'block in the middle' data set (see ...)
        :param n_samples: int, number of samples.
        :param n_dims: int, number of dimensions
        :return: array (shape=[n_samples,n_dims], float). data set.
        """
        # input validation
        if n_dims != 2:
            raise ValueError("support only 2-dimensions")

        # init
        n_1 = int(0.5*n_samples)
        n_2 = n_samples - n_1
        n_1_signal = int(0.5*n_1)
        n_1_noise = n_1 - n_1_signal
        n_2_signal = int(0.5*n_2)
        n_2_noise = n_2 - n_2_signal

        # generate noise 'block in corner'
        noise1 = np.random.uniform(low=(0.4, 0.4), high=(0.6, 0.6), size=(n_1_noise, n_dims))
        noise2 = np.random.uniform(low=(0.4, 0.4), high=(0.6, 0.6), size=(n_2_noise, n_dims))

        # generate signal
        signal1 = np.random.uniform(low=(0, 0), high=(0.5, 1), size=(n_1_signal, n_dims))
        signal2 = np.random.uniform(low=(0.5, 0), high=(1, 1), size=(n_2_signal, n_dims))

        X = np.vstack([noise1, signal1, noise2, signal2])
        y = np.zeros(n_samples, dtype=int)
        y[-n_2:] = 1

        return X, y

    def generate_data_set(self, n_samples, n_dims, data_set):
        """
        generate data set.
        :param n_samples: int, number of samples.
        :param n_dims: int, number of dimensions.
        :param data_set: sting, data set to generate.
        :return: pandas DataFrame with 'id', 'y', and 'X_{i}', columns.
        """
        # generate X, y
        if data_set == 'block_in_the_corner':
            X, y = self.generate_X_y_bic(n_samples, n_dims)
        elif data_set == 'block_in_the_middle':
            X, y = self.generate_X_y_bim(n_samples, n_dims)
        else:
            raise ValueError("unsupported data set")

        # generate ids
        ids = self.generate_ids(n_samples)

        # pack in DataFrame
        X_cols = ['X_{:d}'.format(i) for i in range(np.size(X, axis=1))]
        df = pd.DataFrame(data=X, columns=X_cols)
        df['y'] = y
        df['id'] = ids

        return df

    def get_X_y(self, df, ids):
        """
        get X,y tuples according to given ids.
        :param df: pandas DataFrame with 'id', 'y' and ['X_1','X_2',...,] columns
        :param ids: array (shape=[n_ids,_], string). requested ids.
        :return: X, y where X is array (shape=[n_ids,:], float or int) of inputs (features) and y is array
        (shape=[n_ids,_], int) of outputs.
        """
        X_cols = self.get_X_cols(df)
        df_s = df.loc[df['id'].isin(ids), :]
        X = df_s[X_cols].to_numpy()
        y = df_s['y'].to_numpy()
        return X, y

    @staticmethod
    def get_X_cols(df):
        """
        get X columns in DataFrame
        :param df: pandas DataFrame with 'id', 'y' and ['X_1','X_2',...,] columns
        :return: array (shape=[n_cols,_], string) column names for X data.
        """
        cols = [c for c in df.columns if c.startswith('X_')]
        return cols

    @staticmethod
    def generate_ids(n_ids, n_chars=16):
        """
        generate string ids.
        :param n_ids: int, number of string ids required.
        :param n_chars: int, number of chars for each id string.
        :return: array (shape=[n_ids,_], string). ids.
        """
        ids = [hashlib.sha1(str(n).encode()).hexdigest()[:n_chars] for n in range(n_ids)]
        return ids

    def get_kernel_params(self):
        """ set kernel parameters """
        # set kernel
        k1 = dict()
        k1.setdefault('active_dims', np.arange(self.n_dims))
        k1.setdefault('kern_func', 'rbf')
        k1.setdefault('kern_name', 'raidal basis function')
        k1.setdefault('params', {})
        k1['params'].setdefault('ard', False)
        k1['params'].setdefault('variance', 1)
        k1['params'].setdefault('lengthscale', 1)
        return k1

    def seed_sample(self, df, n_samples):
        """
        seed samples from data set.
        :param df: pandas DataFrame with 'id', 'y' and ['X_1','X_2',...,] columns
        :param n_samples: int, number of samples to select from each category.
        :return: array (shape=[2,_], selected ids
        """
        unq_y = df['y'].unique()
        ids = np.array([])
        for y in unq_y.tolist():
            ids_y = df.loc[df['y'] == y, 'id'].sample(n=n_samples,
                                                      random_state=self.random_state).to_numpy()
            ids = np.append(ids, ids_y)
        return ids

    def run_active_learning_iterations(self,
                                       df,
                                       n_iter,
                                       n_queries,
                                       query_method):
        """
        run active learning iteration
        :param df: pandas DataFrame with 'id', 'y' and ['X_1','X_2',...,] columns
        :param seed_ids: array (shape=[n_seed, _], string). ids of seed batch.
        :param X_test: array (shape=[n_ids,:], float or int). Inputs
        :param y_test: array (shape=[n_ids,_], int). Outputs.
        :param n_iter: int, number of active learning iterations.
        :param n_queries: int, number of queries to perform on each iteration.
        :param query_method: string, query method.
        :return: float, area under the learning curve (ALC).
        """
        # set kernel params
        kern_params = self.get_kernel_params()

        # init pool
        ids = df['id'].to_numpy()
        ids_queried = self.seed_sample(df, n_samples=1)
        ids_test = ids[np.isin(ids, ids_queried, invert=True)]

        # run active learning iterations
        f1_arr = np.zeros(n_iter)
        for n in range(n_iter):
            # get train\test data
            X_train, y_train = self.get_X_y(df, ids_queried)
            X_test, y_test = self.get_X_y(df, ids_test)
            y_train = y_train[:, np.newaxis]
            y_test = y_test[:, np.newaxis]

            # train model
            self.model.train(X_train, y_train, kern_params=kern_params)

            # predict
            y_prob = self.model.predict(X_test)
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > 0.5] = 1

            # evaluate
            f1 = f1_score(y_test, y_pred)
            logging.info('iter: {:d}, f1_score: {:.3f}'.format(n, f1))
            f1_arr[n] = f1

            # query new points
            if query_method == 'BALD':
                criteria = self.model.compute_informativeness(X_test)
                ids_sampled = ids_test[np.argpartition(criteria, -n_queries, axis=0)[-n_queries:]]
            elif query_method == 'RAND':
                ids_sampled = np.random.choice(ids_test, n_queries)
            else:
                raise ValueError("unsupported query method")

            # update pool
            ids_queried = np.append(ids_queried, ids_sampled)
            ids_test = ids_test[np.isin(ids_test, ids_queried, invert=True)]

        alc = np.trapz(f1_arr)
        return alc

    def test_bald_block_in_corner(self):
        """" Test BALD sampler on 'block in the corner' data set """
        # init
        n_iter = 20
        n_queries = 1

        # get data set
        df = self.generate_data_set(self.n_samples,
                                    self.n_dims,
                                    data_set='block_in_the_corner')

        # run active learning iteration using BALD criteria
        alc_bald = self.run_active_learning_iterations(df,
                                                       n_iter=n_iter,
                                                       n_queries=n_queries,
                                                       query_method='BALD')
        # run active learning iteration using random criteria
        alc_rand = self.run_active_learning_iterations(df,
                                                       n_iter=n_iter,
                                                       n_queries=n_queries,
                                                       query_method='RAND')
        # evaluate in terms of area under the learning curve
        logging.info('ALC results: bald - {:.3f}, rand - {:.3}'.format(alc_bald, alc_rand))
        self.assertGreater(alc_bald, alc_rand)

    def test_bald_block_in_middle(self):
        """" Test BALD sampler on 'block in the middle' data set """
        # init
        n_iter = 20
        n_queries = 1

        # get data set
        df = self.generate_data_set(self.n_samples, self.n_dims, data_set='block_in_the_middle')

        # run active learning iterations using BALD criteria
        alc_bald = self.run_active_learning_iterations(df,
                                                       n_iter=n_iter,
                                                       n_queries=n_queries,
                                                       query_method='BALD')
        # run active learning iterations using RANDOM criteria                                         
        alc_rand = self.run_active_learning_iterations(df,
                                                       n_iter=n_iter,
                                                       n_queries=n_queries,
                                                       query_method='RAND')
        # evaluate in terms of area under the learning curve
        logging.info('ALC results: bald - {:.3f}, rand - {:.3}'.format(alc_bald, alc_rand))
        self.assertGreater(alc_bald, alc_rand)


if __name__ == '__main__':
    unittest.main(verbosity=2)
