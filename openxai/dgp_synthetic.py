import numpy as np
class generate_gaussians:
    
    def __init__(self,
                 n_samples=10000,
                 dimensions=20,
                 n_clusters=100,
                 distance_to_center=5,
                 test_size=0.25,
                 upper_weight=1,
                 lower_weight=-1,
                 seed=567,
                 correlation_matrix=None,
                 sparsity=0.5):
        '''
        :param n_samples: int; # samples per cluster
        :param dim: int; # dimensions
        :param n_clusters: int; # clusters
        :param distance:  pos real; distance between cluster centers
        :param upper_weight: real; upper bound for largest weight
        :param lower_weight:  real; lower bound for lowest weight
        :param seed: series of ints;
        :param test_size: frac: size of the test set
        :param p: frac; determine the probability of bernoulli
        '''
        
        # Random Seed
        np.random.seed(seed)
        
        # SETUP
        self.n_samples = n_samples  # samples
        self.N_clusters = n_clusters  # clusters
        self.dim = dimensions  # dimensions
        self.distance = distance_to_center  # scaling distance for cluster centers
        self.test_size = test_size
        self.p = sparsity
        
        # True Model Coefficients in [lower, upper]
        self.w = np.random.uniform(upper_weight, lower_weight, dimensions)  # weight vector sampled uniformly at random
        # Constant covariance matrix for all clusters
        if correlation_matrix is None:
            self.sigma = np.eye(dimensions)  # identity covariance matrix for all clusters
        else:
            self.sigma = correlation_matrix
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _get_mus(self, multiplier=2):
        '''
        This function obtains the cluster centers (\mu) for the N clusters
        '''
        mus = []
        
        if self.dim >= self.N_clusters:
            
            fill_ins = np.ones(self.dim) * self.distance
            for i in range(self.N_clusters):
                filled = np.zeros(self.N_clusters)
                filled_rest = np.zeros(self.dim - self.N_clusters)
                filled[i] = fill_ins[i]
                filled = np.concatenate((filled, filled_rest), axis=None)
                mus.append(filled)
        
        elif self.N_clusters == self.dim * 2:
            
            fill_ins_pos = np.ones(self.dim) * self.distance
            fill_ins_neg = np.ones(self.dim) * self.distance * -1
            
            # fill positives
            for i in range(self.dim):
                filled = np.zeros(self.dim)
                filled[i] = fill_ins_pos[i]
                mus.append(filled)
            
            # fill negatives
            for i in range(self.dim):
                filled = np.zeros(self.dim)
                filled[i] = fill_ins_neg[i]
                mus.append(filled)
        
        else:
            # here: we will have d_effective >= 1.
            d_effective = int(np.floor(self.N_clusters / self.dim))
            multipliers = []
            multipliers.append(1)  # first multiplier should simply be 1
            
            for j in range(1, d_effective + 1):
                multipliers.append(j * multiplier)
            
            length_last = self.N_clusters % self.dim
            
            it = 0
            fill_ins = np.ones(self.dim) * self.distance
            while it <= d_effective:
                
                # decide end of loops
                if it == d_effective:
                    N_remain = length_last
                else:
                    N_remain = self.dim
                
                for i in range(N_remain):
                    filled = np.zeros(self.dim)
                    filled[i] = fill_ins[i] * multipliers[it]
                    mus.append(filled)
                
                it += 1
        
        return mus
    
    def _get_mask(self):
        '''
        This function produces the cluster dependent masking vectors
        '''
        
        mask = []
        for i in range(self.dim):
            rv = np.random.binomial(n=1, p=self.p)
            mask.append(rv)
        
        return np.array(mask)
    
    def _find_threshold(self, pis, size=1000, target=0.5):
        '''
        This function returns a threshold leading to 50/50 split between the 2 classes
        '''
        
        thresholds = np.linspace(0, 1, size)
        diffs = []
        
        for i in range(size):
            threshold = thresholds[i]
            y = (pis > threshold) * 1
            frac = y.sum() / pis.shape[0]
            diff = np.abs(target - frac)
            diffs.append(diff)
        
        best_idx = np.argmin(diffs)
        best_threshold = thresholds[best_idx]
        
        return best_threshold
    
    def dgp_vars(self):
        
        # obtain cluster centers
        mus = self._get_mus()
        
        df = []
        masks = []
        pis = []
        cluster_index = []
        w_eff_all = []
        
        w_all = np.repeat(self.w.reshape(1, -1), repeats=self.n_samples * self.N_clusters, axis=0)
        
        # Loop over all clusters
        # For each cluster, compute:
        # - masks,
        # - data,
        # - effective weights,
        # - prob. outputs
        # - cluster index
        
        for i in range(self.N_clusters):
            cluster_idx = (np.ones(self.n_samples) * i).astype(int)
            m = self._get_mask()  # compute mask
            X = np.random.multivariate_normal(mus[i], self.sigma, self.n_samples)  # get normally distributed sample
            w_eff = m * self.w  # compute masked explanation
            pi = self._sigmoid(X @ w_eff)  # compute probability output
            
            # reshape mask and effective explanation
            m = np.repeat(m.reshape(1, -1), repeats=self.n_samples, axis=0)
            w_eff = np.repeat(w_eff.reshape(1, -1), repeats=self.n_samples, axis=0)
            
            # Collect Variables of interest
            masks.append(m)
            df.append(X)
            pis.append(pi)
            cluster_index.append(cluster_idx)
            w_eff_all.append(w_eff)
        
        # Collect variables in appropriately sized numpy arrays
        pis = np.array(pis).reshape(-1)
        X = np.array(df).reshape(self.N_clusters * self.n_samples, -1)
        cluster_index = np.array(cluster_index).reshape(-1)
        masks = np.array(masks).reshape(self.N_clusters * self.n_samples, -1)
        w_eff_all = np.array(w_eff_all).reshape(self.N_clusters * self.n_samples, -1)
        
        # Find threshold for 50/50 split between class labels
        threshold = self._find_threshold(pis, size=1000)
        # Get class labels
        y = (pis > threshold) * 1
        
        # Sample train and test indices
        n_test = int(self.n_samples * self.N_clusters * self.test_size)
        all_idx = set(np.arange(0, self.n_samples * self.N_clusters))
        test_idx = set(np.random.choice(self.n_samples * self.N_clusters, n_test, replace=False))
        train_idx = list(all_idx - test_idx)
        test_idx = list(test_idx)
        
        var_dict = {
            'data': X,                     # dim = n*N x d
            'target': y,                   # dim = n*N
            'probs': pis,                  # dim = n*N
            'masks': masks,                # dim = n*N x d
            'weights': w_all,              # dim = n*N x d
            'masked_weights': w_eff_all,   # dim = n*N x d
            'cluster_idx': cluster_index   # dim = n*N
        }
        
        var_dict_test = {
            'data': X[test_idx, :],                     # dim = n*N x d
            'target': y[test_idx],                      # dim = n*N
            'probs': pis[test_idx],                     # dim = n*N
            'masks': masks[test_idx, :],                # dim = n*N x d
            'weights': w_all[test_idx, :],              # dim = n*N x d
            'masked_weights': w_eff_all[test_idx, :],   # dim = n*N x d
            'cluster_idx': cluster_index[test_idx]      # dim = n*N
        }
        
        var_dict_train = {
            'data': X[train_idx, :],                     # dim = n*N x d
            'target': y[train_idx],                      # dim = n*N
            'probs': pis[train_idx],                     # dim = n*N
            'masks': masks[train_idx, :],                # dim = n*N x d
            'weights': w_all[train_idx, :],              # dim = n*N x d
            'masked_weights': w_eff_all[train_idx, :],   # dim = n*N x d
            'cluster_idx': cluster_index[train_idx]      # dim = n*N
        }
        
        return var_dict, var_dict_train, var_dict_test
