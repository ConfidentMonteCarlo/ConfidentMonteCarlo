import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import math

# df columns for different usages
formats = {
    'dataset' : ['pwd', 'freq'],
    'sample' : ['pwd', 'logprob', 'freq'],
}

class ConfidentMonteCarlo():
    """
    An interface for the Confident Monte Carlo technique: rigorous statistical analysis on password guessing models, password datasets, and password policies.
    """
    def __init__(self, model, filename=None, policy=lambda x: True):
        """
        Constructor of the ConfidentMonteCarlo class.

        Parameters:
            model (pgm.Model):
                Password guessing model to be analyzed.
                Note: model must support functions 'generate()' and 'logprob()'. See file password_guessing_model.py for details.
            filename (str, optional):
                Password dataset file, user can omit this when instantiating the object and/or specify later with parse_file().
                Note: In csv format with columns 'pwd' and 'freq', representing the unique passowrds and the number of times each password occurs.
                      The two columns should be seperated by the \t character, see file io.py for helpers for compiling csv files in this format.
            policy (Callable[str, bool], optional):
                Password composition policy, user can omit this when instantiating the object and/or specify later with set_policy().
        """
        assert model is not None, "must specify password guessing model when initializing Confident Monte Carlo object"
        self.model = model
        self.policy = policy
        self.filename = filename
        if filename is None:
            self.df = pd.DataFrame()
        else:
            self.parse_file(filename)

        # initialize for assertion errors
        self.samples = []
        self.n_samples = []
        self.mesh_LBs_1 = []
        self.mesh_LBs_2 = []
        self.lam_ub3 = -1
        self.bound1_eps2 = -1
        self.bound2_eps2 = -1
        self.bound3_eps = -1

    def set_model(self, model):
        """
        Updates the password guesssing model to be analyzed.

        Parameters:
            model (pgm.Model):
                Password guessing model to be set and analyzed.
        """
        self.model = model

    def parse_file(self, filename):
        """
        Reads and parses a password dataset file then constructs a pandas dataframe for later use.

        Parameters:
            filename (str):
                The filename to be parsed.
                Note: In csv format with columns 'pwd' and 'freq', representing the unique passowrds and the number of times each password occurs.
                      The two columns should be seperated by the \t character, see file io.py for helpers for compiling csv files in this format.

        Returns:
            pd.DataFrame:
                pandas dataframe with columns ['pwd', 'logprob', 'prob', 'freq', 'cumfreq', 'cumperc']
        """
        assert self.model is not None, "must specify passowrd guessing model before parsing password dataset"
        self.filename = filename
        df = pd.read_csv(filename, sep = '\t', header = None, names = formats['dataset'], quoting=csv.QUOTE_NONE)
        df['pwd'] = df['pwd'].astype(str)
        df['logprob'] = df['pwd'].apply(lambda x: self.model.logprob(x))
        df['prob'] = np.power(2.0, df['logprob'])
        df.sort_values(by = ['logprob'], inplace = True, ignore_index = True)
        df['cumfreq'] = np.cumsum(df['freq'])
        self.total_accounts = df['freq'].sum()
        df['cumperc'] = df['cumfreq'] / self.total_accounts
        self.df = df

        return df

    def set_policy(self, policy):
        """
        Updates/sets the password policy to be enforced.
        
        Paremeters:
            policy (Callable[str, bool]):
                The password composition being enforced.
        """
        self.policy = policy

    def sample(self, k1):
        """
        Draws k1 samples from the model specified in the ConfidentMonteCarlo object.
        Stores samples and precomputes values for binary search and direct access to suffix sum values.

        Paremeters:
            k1 (int):
                Number of samples to draw.
            
        Returns:
            List[Tuple[str, float]]:
                k1 samples drawn with format (pwd, logprob).
        """
        assert k1 > 0, "must draw at least one sample"
        self.k1 = k1
        self.samples = self.model.sample(k1)
        self.allowed_samples = [samp for samp in self.samples if self.policy(samp[0])]

        assert len(self.allowed_samples) > 0, "no samples drawn are allowed by the policy"

        self.logprobs = np.sort(np.array([samp[1] for samp in self.allowed_samples]))
        invprobs = np.power(2, -self.logprobs)
        self.invprobs_suffsum = np.append(np.cumsum(invprobs[::-1])[::-1], 0) # append 0 to prevent out of bounds

        return self.samples

    def write_sample(self, filename):
        """
        Writes the sample drawn to a file.
        File would be in csv format with columns seperated with \t characters: ['pwd', 'logprob', 'freq']
        
        Paremeters:
            filename (str):
                File to write to.
        """
        assert len(self.samples) != 0, "must call sample() before writing sample" 

        freq = {}
        for samp in self.samples:
            freq[samp] = freq.get(samp, 0) + 1
        df = pd.DataFrame([(pwd[0], pwd[1], freq) for pwd, freq in freq.items()], columns=formats['sample'])
        df.to_csv(filename, sep='\t', index=False, quoting=csv.QUOTE_NONE)

    def read_sample(self, filename):
        """
        Reads samples from a file. The file must be in the exact same csv format as the file that write_sample() produced.
        The assumption is that the user calls this method directly with the file created with write_sample() to restore samples from a previous session.

        Parameters:
            filename (str):
                File to read from.
        """
        sample = pd.read_csv(filename, header=0)
        self.samples = []
        for i, row in df.iterrows():
            pwd = row['pwd']
            lp = row['logprob']
            freq = row['freq']
            self.samples.extend([(pwd, lp)] * freq)
        self.k1 = len(self.samples)
        self.allowed_samples = [samp for samp in self.samples if self.policy(samp[0])]

        assert len(self.allowed_samples) > 0, "no samples are allowed by the policy"

        self.logprobs = np.sort(np.array([samp[1] for samp in self.allowed_samples]))
        invprobs = np.power(2, -self.logprobs)
        self.invprobs_suffsum = np.append(np.cumsum(invprobs[::-1])[::-1], 0) # append 0 to prevent out of bounds

    def group_sample(self, n=0):
        """
        Randomly groups the samples drawn into n groups, each group is of size floor(k1/n).
        Note: If k1 is not a multiple of n, remainders are discarded.

        Parameters:
            n (int):
                Number of groups to seperate the samples into. If not set by user, defaults to sqrt(k) where k is the total number of samples drawn.

        Returns:
            List[List[Tuple[str, float]]]:
                Resulting grouping of samples in the format (pwd, logprob) and each group is stored in a list.
        """
        assert len(self.samples) > 0, "must call sample() before grouping samples"
        assert n <= len(self.samples), "too many groups"

        if n == 0:
            n = math.sqrt(self.k1) 
        self.n = n
        self.k2 = self.k1 // n

        random.shuffle(self.samples)
        self.n_samples = [self.samples[i : i+self.k2] for i in range(0, self.n * self.k2, self.k2)]
        self.n_allowed_samples = [[samp for samp in sampset if self.policy(samp[0])] for sampset in self.n_samples]
        self.n_logprobs = [np.sort(np.array([samp[1] for samp in sampset])) for sampset in self.n_allowed_samples]
        self.n_invprobs = [np.power(2, -logprobs) for logprobs in self.n_logprobs]
        self.n_invprobs_suffsum = [np.append(np.cumsum(invprobs[::-1])[::-1], 0) for invprobs in self.n_invprobs] # append 0 to prevent out of bounds

        return self.n_samples

    def n_sample(self, n, k2):
        """
        Draws n * k2 samples from the model specified in the ConfidentMonteCarlo object.
        Stores samples and precomputes values for binary search and direct access to suffix sum values.

        Paremeters:
            n (int):
                Number of groups.
            k2 (int):
                Number of samples in each group.
            
        Returns:
            List[List[Tuple[str, float]]]:
                n * k samples drawn in the format (pwd, logprob) and each group is stored in a list.
        """
        self.n = n
        self.k2 = k2
        samples = self.model.sample(n * k2)
        self.n_samples = [samples[i : i+self.k2] for i in range(0, self.n * self.k2, self.k2)]
        self.n_allowed_samples = [[samp for samp in sampset if self.policy(samp[0])] for sampset in self.n_samples]
        self.n_logprobs = [np.sort(np.array([samp[1] for samp in sampset])) for sampset in self.n_allowed_samples]
        self.n_invprobs = [np.power(2, -logprobs) for logprobs in self.n_logprobs]
        self.n_invprobs_suffsum = [np.append(np.cumsum(invprobs[::-1])[::-1], 0) for invprobs in self.n_invprobs] # append 0 to prevent out of bounds

        return self.n_samples

    def hoeffding_bound(self, q, err_rate):
        """
        Implements Theorem 1 to bound G^EX(q) and G^IN.
        
        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                desired error rate.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]:
                (lower bound, upper bound) for G^EX(q) and G^IN(q).
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"

        # calculate Ghat_IN and Ghat_EX
        idx_EX = np.searchsorted(self.logprobs, q, 'right')
        idx_IN = np.searchsorted(self.logprobs, q, 'left')
        Ghat_EX = self.invprobs_suffsum[idx_EX] / self.k1
        Ghat_IN = self.invprobs_suffsum[idx_IN] / self.k1

        # calculate required epsilon to get desired confidence level
        eps = math.sqrt(-math.log(err_rate) / (2 * self.k1))
        err = eps/math.pow(2, q)

        # bound G_EX and G_IN 
        return ((max(math.floor(Ghat_EX - err), 0), math.ceil(Ghat_EX + err)), (max(math.floor(Ghat_IN - err), 0), math.ceil(Ghat_IN + err)))

    def hoeffding_bound_EX(self, q, err_rate):
        """
        Implements Theorem 1 to bound G^EX(q).
        
        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                desired error rate.

        Returns:
            Tuple[int, int]:
                (lower bound, upper bound) for G^EX(q).
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"
        
        return self.hoeffding_bound(q, err_rate)[0]

    def hoeffding_bound_IN(self, q, err_rate):
        """
        Implements Theorem 1 to bound G^IN(q).
        
        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                desired error rate.

        Returns:
            Tuple[int, int]:
                (lower bound, upper bound) for G^IN(q).
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"
        
        return self.hoeffding_bound(q, err_rate)[1]

    def markov_lowerbound(self, q, err_rate):
        """
        Implements Theorem 2 to get a better lower bound on G^EX(q) and G^IN(q) for rare passwords.

        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                Desired error rate.

        Returns:
            Tuple[int, int]:
                Lower bound for G^EX(q) and G^IN(q).
                Note: the lower bound is 0 if there is no epsilon or delta satisfying the constraints and the confidence level.
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        idx_EXs = [np.searchsorted(row, q, side='right') for row in self.n_logprobs]
        idx_INs = [np.searchsorted(row, q, side='left') for row in self.n_logprobs]
        Ghat_EXs = np.array([self.n_invprobs_suffsum[i][idx_EXs[i]] / self.k2 for i in range(self.n)])
        Ghat_INs = np.array([self.n_invprobs_suffsum[i][idx_INs[i]] / self.k2 for i in range(self.n)])
        G_EX_med = np.median(Ghat_EXs)
        G_IN_med = np.median(Ghat_INs)

        # calculate required epsilon to get desired confidence level
        eps = math.sqrt(-math.log(err_rate) / (2 * self.n))
        if eps >= 0.5:
            return (0, 0) # can't give good lower bound with desired confidence level
        delta = 0.5 - eps

        return (math.floor(delta * G_EX_med), math.floor(delta * G_IN_med))

    def markov_lowerbound_EX(self, q, err_rate):
        """
        Implements Theorem 2 to get a better lower bound on G^EX(q) for rare passwords.

        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                Desired error rate.

        Returns:
            int:
                Lower bound for G^EX(q).
                Note: the lower bound is 0 if there is no epsilon or delta satisfying the constraints and the confidence level.
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        return self.markov_lowerbound(q, err_rate)[0]

    def markov_lowerbound_IN(self, q, err_rate):
        """
        Implements Theorem 2 to get a better lower bound on G^IN(q) for rare passwords.

        Paremeters:
            q (float):
                log probability of password being generated by model.
            err_rate (float):
                Desired error rate.

        Returns:
            int:
                Lower bound for G^IN(q).
                Note: the lower bound is 0 if there is no epsilon or delta satisfying the constraints and the confidence level.
        """
        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        return self.markov_lowerbound(q, err_rate)[1]

    def dataset_curve_bound1_fit(self, Q, err_rate):
        """
        Implements Theorem 12.
        For each log probability in Q, calculate LB and UB with the specified error rate as well as the true lambda values.

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate (float):
                Desired error rate.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"
        
        self.l = len(Q)
        Q = np.array(Q)
        np.sort(Q)

        # calculate required confidence level
        err_rate2 = err_rate / self.l

        # calculate LB and UB in Thm 12
        mesh_bounds = [(UB_G_EX, LB_G_IN) for q in Q for (_, UB_G_EX), (LB_G_IN, _) in [self.hoeffding_bound(q, err_rate2)]]

        # for each q in Q, calculate lambda_G(q)D by counting in the password dataset
        dataset_logprob = self.df['logprob'].to_numpy()
        dataset_cumperc = self.df['cumperc'].to_numpy()
        idx_EXs = np.searchsorted(dataset_logprob, Q, 'right')
        idx_INs = np.searchsorted(dataset_logprob, Q, 'left')
        cracked_fractions_EX = np.where(idx_EXs > 0, 1 - dataset_cumperc[idx_EXs - 1], 1.0)
        cracked_fractions_IN = np.where(idx_INs > 0, 1 - dataset_cumperc[idx_INs - 1], 1.0)

        # pair UB/LBs with logprobs and sort by UB/LB to prepare for binary search
        mesh_UB_lams = sorted([(x[0], lam) for x, lam in zip(mesh_bounds, cracked_fractions_EX)])
        mesh_LB_lams = sorted([(x[1], lam) for x, lam in zip(mesh_bounds, cracked_fractions_IN)])

        # retrive UB/LBs (for convenience)
        self.mesh_UBs_1 = [b for b, lam in mesh_UB_lams]
        self.mesh_LBs_1 = [b for b, lam in mesh_LB_lams]

        # following Thm 12 build prefix max array and suffix min array of lambdas
        self.lam_EX_prefmax_1 = np.maximum.accumulate(np.array([lam for _, lam in mesh_UB_lams]))
        lam_IN_suffmin_1 = np.minimum.accumulate(np.array([lam for _, lam in mesh_LB_lams][::-1]))[::-1]
        self.lam_IN_suffmin_1 = np.append(lam_IN_suffmin_1, 1.0) # append 1.0 to prevent out of bounds

        # build values and edges arrays for plotting
        self.lb1_vals = np.insert(self.lam_EX_prefmax_1, 0, 0.0)
        lb1_edges = np.append(np.insert(self.mesh_UBs_1, 0, 1), self.mesh_UBs_1[-1] * 10)
        lb1_edges = np.where(lb1_edges < 1, 1.0, lb1_edges).astype('float64')
        self.lb1_edges = np.log10(lb1_edges)

        self.ub1_vals = self.lam_IN_suffmin_1
        ub1_edges = np.append(np.insert(self.mesh_LBs_1, 0, 1), self.mesh_LBs_1[-1] * 10)
        ub1_edges = np.where(ub1_edges < 1, 1.0, ub1_edges).astype('float64')
        self.ub1_edges = np.log10(ub1_edges)

    def dataset_curve_bound1_query(self, B):
        """
        Implements Theorem 12.
        Bound the lambda values for the B value using the precomputed arrays during fit().

        Paremeters:
            B (int):
                Guessing number to be estimated.

        Returns:
            Tuple[float, float]:
                (LB, UB) for the fraction of cracked passwords for each guessing number in B.
        """
        assert len(self.mesh_LBs_1) != 0, "must call fit() before calling query()"

        # binary search for threshold
        idx_lb = np.searchsorted(self.mesh_UBs_1, B, 'right')
        idx_ub = np.searchsorted(self.mesh_LBs_1, B, 'left')

        # calculate bounds using the precomputed suffix max and prefix min arrays
        lb = self.lam_EX_prefmax_1[idx_lb - 1] if idx_lb > 0 else 0.0
        ub = self.lam_IN_suffmin_1[idx_ub] # don't need to check out of bounds since we already appended 1.0

        return lb, ub

    def dataset_curve_bound2_fit(self, Q, err_rate):
        """
        Implements Theorem 13.
        For each log probability in Q, calculate the median estimate with the specified error rate as well as the true lambda values.

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate (float):
                Desired error rate.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"
        
        Q.sort()
        self.l = len(Q)

        # calculate required confidence level
        err_rate2 = err_rate / self.l

        # calculate LB and UB in Thm 12
        mesh_bounds = [LB_G_IN_med for q in Q for (_, LB_G_IN_med) in [self.markov_lowerbound(q, err_rate2)]]

        # for each q in Q, calculate lambda_G(q)D by counting in the password dataset
        dataset_logprob = self.df['logprob'].to_numpy()
        dataset_cumperc = self.df['cumperc'].to_numpy()
        idx_INs = np.searchsorted(dataset_logprob, Q, 'left')
        cracked_fractions_IN = np.where(idx_INs > 0, 1 - dataset_cumperc[idx_INs - 1], 1.0)

        # pair LBs with logprobs and sort by LB to prepare for binary search
        mesh_LB_lams = sorted([(x, lam) for x, lam in zip(mesh_bounds, cracked_fractions_IN)])

        # retrive UB/LBs (for convenience)
        self.mesh_LBs_2 = [b for b, lam in mesh_LB_lams]

        # following Thm 12 build prefix max array and suffix min array of lambdas
        lam_IN_suffmin_2 = np.minimum.accumulate(np.array([lam for _, lam in mesh_LB_lams][::-1]))[::-1]
        self.lam_IN_suffmin_2 = np.append(lam_IN_suffmin_2, 1.0) # append 1.0 to prevent out of bounds
        
        # build values and edges arrays for plotting
        self.ub2_vals = self.lam_IN_suffmin_2
        ub2_edges = np.append(np.insert(self.mesh_LBs_2, 0, 1), self.mesh_LBs_2[-1] * 10)
        ub2_edges = np.where(ub2_edges < 1, 1.0, ub2_edges).astype('float64')
        self.ub2_edges = np.log10(ub2_edges)


    def dataset_curve_bound2_query(self, B):
        """
        Implements Theorem 13.
        Upper bound the lambda values for the B value using the precomputed arrays during fit().

        Paremeters:
            B (int):
                Guessing number to be estimated.

        Returns:
            Tuple[float, float]:
                Upper bound for the fraction of cracked passwords for each guessing number in B.
        """
        assert len(self.mesh_LBs_2) != 0, "must call fit() before calling query()"

        # binary search for threshold
        idx_ub = np.searchsorted(self.mesh_LBs_2, B, 'left')

        # calculate bounds using the precomputed suffix max and prefix min arrays
        ub = self.lam_IN_suffmin_2[idx_ub] # don't need np.where since we already appended 1.0

        return ub

    def dataset_curve_bound3_fit(self):
        """
        Implements Theorem 14.
        """
        self.lam_ub3 = (self.df['prob'] > 0).mean()
    
    def dataset_curve_bound3_query(self):
        """
        Implements Theorem 14.

        Returns
            float:
                the third (trivial) upper bound.
        """
        assert self.lam_ub3 != -1, "must call fit() before calling query()"

        return self.lam_ub3

    def dataset_curve_bound_fit(self, Q, err_rate):
        """
        Implements Theorems 12, 13, and 14.
        Fits each of the bounds (lb1, ub1, ub2, ub3).

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate (float):
                Desired error rate.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        self.dataset_curve_bound1_fit(Q, err_rate)
        self.dataset_curve_bound2_fit(Q, err_rate)
        self.dataset_curve_bound3_fit()

    def dataset_curve_bound_query(self, B):
        """
        Implements Theorems 12, 13, and 14.
        Returns each of the bounds (lb1, ub1, ub2, ub3).

        Returns
            Tuple[float, float, float, float]:
                (lb1, ub1, ub2, ub3).
        """
        assert len(self.mesh_LBs_1) != 0, "must call fit() before calling query()"
        assert len(self.mesh_LBs_2) != 0, "must call fit() before calling query()"
        assert self.lam_ub3 != -1, "must call fit() before calling query()"

        b1 = self.dataset_curve_bound1_query(B)
        b2 = self.dataset_curve_bound2_query(B)
        b3 = self.dataset_curve_bound3_query()

        return (b1[0], b1[1], b2, b3)

    def dataset_curve_bound_plot(self, bounds=['lb1', 'ub1', 'ub2', 'ub3'], show=False, savename='dataset_curve_bound.png', title=''):
        """
        Plots lb1, ub1, ub2, and/or ub3, optionally saves and/or displays the plot, and returns the values.

        Paremeters:
            bounds (List[str]):
                A list containing 'lb1', 'ub1', 'ub2', and/or 'ub3'. Will only plot and/or return the specified bounds.
            show (bool, optional):
                Whether to display the plot or not on the screen. (default False)
            savename (str, optional):
                File to save.
            title (str, optional):
                Title of the plot.

        Returns:
            Dict[str, Tuple[List[float], List[int]]]
                A dictionary of {bound : (values, edges)} with the requested bounds where bound is 'lb1', 'ub1', 'ub2', or 'ub3'.
        """
        if 'lb1' in bounds:
            assert len(self.mesh_LBs_1) != 0, "must call fit() for bound 1 before plotting"
        if 'ub1' in bounds:
            assert len(self.mesh_LBs_1) != 0, "must call fit() for bound 1 before plotting"
        if 'ub2' in bounds:
            assert len(self.mesh_LBs_2) != 0, "must call fit() for bound 2 before plotting"
        if 'ub3' in bounds:
            assert self.lam_ub3 != -1, "must call fit() for bound 3 before plotting"

        if savename != '' or show:
            fig, ax = plt.subplots()
            if 'lb1' in bounds:
                ax.stairs(self.lb1_vals, self.lb1_edges, linewidth=0.8, baseline=None, label='lb1', color='red')
            if 'ub1' in bounds:
                ax.stairs(self.ub1_vals, self.ub1_edges, linewidth=0.8, baseline=None, label='ub1', color='green')
            if 'ub2' in bounds:
                ax.stairs(self.ub2_vals, self.ub2_edges, linewidth=0.8, baseline=None, label='ub2', color='blue')
            if 'ub3' in bounds:
                ax.axhline(y=self.dataset_curve_bound3_query(), linewidth=0.8, label='ub3', color='orange')

            ax.set_xlabel('log10(guessing number)')
            ax.set_ylabel('fraction of cracked passwords')
            ax.set_title(title)
            ax.legend()
            if savename != '':
                fig.set_size_inches(12, 7)
                fig.savefig(savename, dpi=300)
            if show:
                plt.show()

        ret = {}
        if 'lb1' in bounds:
            ret['lb1'] = (self.lb1_vals, self.lb1_edges)
        if 'ub1' in bounds:
            ret['ub1'] = (self.ub1_vals, self.ub1_edges)
        if 'ub2' in bounds:
            ret['ub2'] = (self.ub2_vals, self.ub2_edges)
        if 'ub3' in bounds:
            ret['ub3'] = self.lam_ub3

        return ret

    def population_curve_bound1_fit(self, Q, err_rate1, err_rate2):
        """
        Implements Theorem 6.

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate1 (float):
                Desired error rate for sampling from the model.
            err_rate2 (float):
                Desired error rate for sampling the dataset from the population.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate1 >= 0 and err_rate1 <= 1, "err_rate1 must be a valid probability"
        assert err_rate2 >= 0 and err_rate2 <= 1, "err_rate2 must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"

        self.dataset_curve_bound1_fit(Q, err_rate1)
        self.bound1_eps2 = math.sqrt(-math.log(err_rate2) / (2 * self.df.shape[0]))

    def population_curve_bound1_query(self, B):
        """
        Implements Theorem 6.

        Paremeters:
            B (int):
                Guessing number to be estimated.

        Returns:
            Tuple[float, float]:
                Bounds for the estimated fraction of cracked passwords in the population for each guessing number in B.
        """
        assert self.bound1_eps2 != -1, "must call fit() before calling query()"

        lb, ub = self.dataset_curve_bound1_query(B)

        return max(lb - self.bound1_eps2, 0.0), min(ub + self.bound1_eps2, 1.0)

    def population_curve_bound2_fit(self, Q, err_rate1, err_rate2):
        """
        Implements Theorem 7.

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate1 (float):
                Desired error rate for sampling from the model.
            err_rate2 (float):
                Desired error rate for sampling the dataset from the population.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate1 >= 0 and err_rate1 <= 1, "err_rate1 must be a valid probability"
        assert err_rate2 >= 0 and err_rate2 <= 1, "err_rate2 must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        self.dataset_curve_bound2_fit(Q, err_rate1)
        self.bound2_eps2 = math.sqrt(-math.log(err_rate2) / (2 * self.df.shape[0]))

    def population_curve_bound2_query(self, B):
        """
        Implements Theorem 7.

        Paremeters:
            B (int):
                Guessing number to be estimated.

        Returns:
            float:
                Upper bound for the estimated fraction of cracked passwords in the population for each guessing number in B.
        """
        assert self.bound2_eps2 != -1, "must call fit() before calling query()"

        ub = self.dataset_curve_bound2_query(B)

        return min(ub + self.bound2_eps2, 1.0)

    def population_curve_bound3_fit(self, err_rate):
        """
        Implements Theorem 8.

        Paremeters:
            err_rate (float):
                Desired error rate for sampling the dataset from the population.
        """
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"

        self.dataset_curve_bound3_fit()
        self.bound3_eps = math.sqrt(-math.log(err_rate) / (2 * self.df.shape[0]))
    
    def population_curve_bound3_query(self):
        """
        Implements Theorem 8.

        Returns
            float:
                the third (trivial) upper bound.
        """
        assert self.bound3_eps != -1, "must call fit() before calling query()"

        return min(self.lam_ub3 + self.bound3_eps, 1.0)

    def population_curve_bound_fit(self, Q, err_rate1, err_rate2):
        """
        Implements Theorems 6, 7, and 8.
        Fits each of the bounds (lb1, ub1, ub2, ub3).

        Paremeters:
            Q (List[float]):
                Probability mesh points (in log2).
            err_rate1 (float):
                Desired error rate for sampling from the model.
            err_rate2 (float):
                Desired error rate for the dataset.
        """
        assert all(q <= 0 for q in Q), "Q must be a list of valid log probabilities"
        assert err_rate1 >= 0 and err_rate1 <= 1, "err_rate1 must be a valid probability"
        assert err_rate2 >= 0 and err_rate2 <= 1, "err_rate2 must be a valid probability"
        assert not self.df.empty, "must specify password dataset to use this function"
        assert len(self.samples) != 0, "must call sample() before calculating bounds"
        assert len(self.n_samples) != 0, "must call group_sample() or n_sample() before calculating bounds"

        self.population_curve_bound1_fit(Q, err_rate1, err_rate2)
        self.population_curve_bound2_fit(Q, err_rate1, err_rate2)
        self.population_curve_bound3_fit(err_rate2)

    def population_curve_bound_query(self, B):
        """
        Implements Theorems 6, 7, and 8.
        Returns each of the bounds (lb1, ub1, ub2, ub3).

        Returns
            Tuple[float, float, float, float]:
                (lb1, ub1, ub2, ub3).
        """
        assert self.bound1_eps2 != -1, "must call fit() before calling query()"
        assert self.bound2_eps2 != -1, "must call fit() before calling query()"
        assert self.bound3_eps != -1, "must call fit() before calling query()"

        b1 = self.population_curve_bound1_query(B)
        b2 = self.population_curve_bound2_query(B)
        b3 = self.population_curve_bound3_query()

        return (b1[0], b1[1], b2, b3)

    def population_curve_bound_plot(self, bounds=['lb1', 'ub1', 'ub2', 'ub3'], show=False, savename='population_curve_bound.png', title=''):
        """
        Plots lb1, ub1, ub2, and/or ub3, optionally saves and/or displays the plot.

        Paremeters:
            bounds (List[str]):
                A list containing 'lb1', 'ub1', 'ub2', and/or 'ub3'. Will only plot and/or return the specified bounds.
            show (bool, optional):
                Whether to display the plot or not on the screen. (default False)
            savename (str, optional):
                File to save.
            title (str, optional):
                Title of the plot.

        Returns:
            Dict[str, Tuple[List[float], List[int]]]
                A dictionary of {bound : (values, edges)} with the requested bounds where bound is 'lb1', 'ub1', 'ub2', or 'ub3'.
        """
        if 'lb1' in bounds:
            assert self.bound1_eps2 != -1, "must call fit() before calling query()"
        if 'ub1' in bounds:
            assert self.bound1_eps2 != -1, "must call fit() before calling query()"
        if 'ub2' in bounds:
            assert self.bound2_eps2 != -1, "must call fit() before calling query()"
        if 'ub3' in bounds:
            assert self.bound3_eps != -1, "must call fit() before calling query()"

        if savename != '' or show:
            fig, ax = plt.subplots()
            if 'lb1' in bounds:
                ax.stairs(np.maximum(self.lb1_vals - self.bound1_eps2, 0.0), self.lb1_edges, linewidth=0.8, baseline=None, label='lb1', color='red')
            if 'ub1' in bounds:
                ax.stairs(np.minimum(self.ub1_vals + self.bound1_eps2, 1.0), self.ub1_edges, linewidth=0.8, baseline=None, label='ub1', color='green')
            if 'ub2' in bounds:
                ax.stairs(np.minimum(self.ub2_vals + self.bound2_eps2, 1.0), self.ub2_edges, linewidth=0.8, baseline=None, label='ub2', color='blue')
            if 'ub3' in bounds:
                ax.axhline(y=self.population_curve_bound3_query(), linewidth=0.8, label='ub3', color='orange')

            ax.set_xlabel('log10(guessing number)')
            ax.set_ylabel('fraction of cracked passwords')
            ax.set_title(title)
            ax.legend()
            if savename != '':
                fig.set_size_inches(12, 7)
                fig.savefig(savename, dpi=300)
            if show:
                plt.show()

        ret = {}
        if 'lb1' in bounds:
            ret['lb1'] = (np.maximum(self.lb1_vals - self.bound1_eps2, 0.0), self.lb1_edges)
        if 'ub1' in bounds:
            ret['ub1'] = (np.minimum(self.ub1_vals + self.bound1_eps2, 1.0), self.ub1_edges)
        if 'ub2' in bounds:
            ret['ub2'] = (np.minimum(self.ub2_vals + self.bound2_eps2, 1.0), self.ub2_edges)
        if 'ub3' in bounds:
            ret['ub3'] = self.population_curve_bound3_query()

        return ret

    # Simple Interface. Automatic sample/fits.
    def auto_sample(self):
        """
        Helper for automated simple interface.
        Draws samples if haven't or sample size is too small.
        """
        if len(self.samples) < 5000000:
            self.sample(10000000)
            self.group_sample(4000)
        elif len(self.n_samples) == 0:
            self.group_sample(math.ceil(math.sqrt(len(self.samples))))

    def auto_fit(self, err_rate1, err_rate2):
        """
        Helper for automated simple interface.
        Fits curve bounds if haven't.

        Parameters:
            err_rate1 (float):
                Desired error rate for sampling from the model.
            err_rate2 (float):
                Desired error rate for sampling the dataset from the population.
        """
        self.auto_sample()
        mpg = MeshPointGenerator()
        # self.population_curve_bound_fit(mpg.from_sample(self.samples, 150), err_rate1, err_rate2)
        logprobs = [lp for pwd, lp in self.samples]
        self.population_curve_bound_fit(mpg.even_range(math.pow(2, self.logprobs[100000]), math.pow(2, self.logprobs[-1]), 150), err_rate1, err_rate2)

    def guessing_number_bound(self, q, err_rate=0.01):
        """
        Simple Interface.
        Implements Theorem 1 and 2 to bound the guessing number.
        Automatically draws samples if haven't yet or sample size is too small.

        Parameters:
            q (float OR str):
                log probability of password being generated by model OR a password.
            err_rate (float):
                Desired error rate.

        Returns:
            Tuple[int, int]:
                (lowerbound, upperbound) for guessing number.
                If the password specified is impossible for the model, returns (-1, -1).
        """
        if isinstance(q, str):
            q = self.model.logprob(q)
            if q == -float('inf'):
                return (-1, -1)

        assert q <= 0, "q must be a valid log probability"
        assert err_rate >= 0 and err_rate <= 1, "err_rate must be a valid probability"

        # have to do this since we're generating a confidence interval, not bounds individually, so must split error rates
        err_rate = err_rate / 2

        self.auto_sample()
        bounds1 = self.hoeffding_bound(q, err_rate)
        bounds2 = self.hoeffding_bound(q, err_rate/2)
        bounds3 = self.markov_lowerbound(q, err_rate/2)

        return max(bounds2[0][0], bounds3[0])+1, bounds1[1][1]

    def guessing_number_plot(self, err_rate=0.01, title='', savename='guessing_number_plot.png'):
        """
        Simple interface.
        Uses Theorem 1 and 2 to plot bounds for guessing number vs. probability.
        Saves and displays the plot.

        Parameters:
            err_rate (float):
                Desired error rate.
            title (str):
                Title of the plot.
            savename (str):
                Path of the file to save to plot.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Data that generates the plot in the format (logprobs, lb, ub)
        """
        self.auto_sample()

        if len(title) == 0:
            title = f'Bounds for Guessing Number with Confidence {1-err_rate}'

        logprobs = [lp for pwd, lp in self.samples]
        logprobs = np.linspace(self.logprobs[2000], self.logprobs[-1], 50)
        bounds = [self.guessing_number_bound(q, err_rate) for q in logprobs]
        lbs = np.log2(np.array([b[0] for b in bounds]).astype('float64'))
        ubs = np.log2(np.array([b[1] for b in bounds]).astype('float64'))

        fig, ax = plt.subplots()
        ax.plot(logprobs, lbs, linewidth=0.8, marker='.', label='lb', color='red')
        ax.plot(logprobs, ubs, linewidth=0.8, marker='.', label='ub', color='blue')
        ax.set_xlabel('log2(probability)')
        ax.set_ylabel('log2(guessing number)')
        ax.set_title(title)
        ax.legend()
        fig.set_size_inches(12, 7)
        fig.savefig(savename, dpi=300)
        plt.show()

        return logprobs, lbs, ubs

    def dataset_guessing_curve_bound(self, B, err_rate=0.01):
        """
        Simple interface.
        Uses Theorems 12, 13, and 14.
        Returns (lb, ub) for the dataset guessing curve.

        Parameters:
            B (int):
                Guessing number.
            err_rate (float, opitonal):
                Desired error rate for sampling from the model.

        Returns:
            Tuple[float, float]:
                (lb, ub).
        """
        assert not self.df.empty, "must specify password dataset to analyze guessing curve, use parse_file() to specify dataset"

        self.auto_fit(err_rate/2, 0.01)
        bounds = self.dataset_curve_bound_query(B)

        return (bounds[0], min(bounds[1], bounds[2], bounds[3]))

    def dataset_guessing_curve_plot(self, err_rate=0.01, title='', savename='dataset_guessing_curve.png'):
        """
        Plots lb1, ub1, ub2, and ub3, displays and saves the image, and returns the values.

        Paremeters:
            err_rate (float, optional):
                Desired error rate for sampling from the model.
            title (str, optional):
                Title of the plot.
            savename (str, optional):
                File to save.

        Returns:
            Dict[str, Tuple[List[float], List[int]]]
                A dictionary of {bound : (values, edges)} where bound is 'lb1', 'ub1', 'ub2', or 'ub3'.
        """
        assert not self.df.empty, "must specify password dataset to analyze guessing curve, use parse_file() to specify dataset"

        if len(title) == 0:
            title = f'Guessing Curve of Dataset with Confidence {1-err_rate}'

        self.auto_fit(err_rate, 0.01)

        return self.dataset_curve_bound_plot(show=True, title=title, savename=savename)

    def population_guessing_curve_bound(self, B, err_rate=0.01):
        """
        Simple interface.
        Uses Theorems 6, 7, and 8.
        Returns each of the bounds (lb, ub) for the population guessing curve.

        Parameters:
            B (int):
                Guessing number;
            err_rate (float, optional):
                Desired error rate.

        Returns:
            Tuple[float, float]:
                (lb, ub)
        """
        assert not self.df.empty, "must specify password dataset to analyze guessing curve, use parse_file() to specify dataset"

        err_rate1 = err_rate / 2
        err_rate2 = err_rate - err_rate1

        self.auto_fit(err_rate1/2, err_rate2/3)
        bounds = self.population_curve_bound_query(B)

        return (bounds[0], min(bounds[1], bounds[2], bounds[3]))

    def population_guessing_curve_plot(self, err_rate=0.01, title='', savename='population_guessing_curve.png'):
        """
        Plots lb1, ub1, ub2, and ub3, displays and saves the image, and returns the values.

        Paremeters:
            err_rate1 (float, optional):
                Desired error rate.
            title (str, optional):
                Title of the plot.
            savename (str, optional):
                File to save.

        Returns:
            Dict[str, Tuple[List[float], List[int]]]
                A dictionary of {bound : (values, edges)} where bound is 'lb1', 'ub1', 'ub2', or 'ub3'.
        """
        assert not self.df.empty, "must specify password dataset to analyze guessing curve, use parse_file() to specify dataset"

        if len(title) == 0:
            title = f'Guessing Curve of Population with Confidence {1-err_rate1-err_rate2}'

        err_rate1 = err_rate / 2
        err_rate2 = err_rate - err_rate1

        self.auto_fit(err_rate1, err_rate2)

        return self.population_curve_bound_plot(show=True, title=title, savename=savename)


class MeshPointGenerator():
    """
    Helper class that generates quality mesh points for Confident Monte Carlo analysis.
    """
    def even_range(self, low=1e-15, high=1e-5, num=50, scale='log'):
        """
        Generates evenly distributed probability mesh points between low and high.
        Scale can be 'normal' or 'log', and all mesh points are returned in log2.

        Paremeters:
            low (float):
                Lower end of the range.
            high (float):
                Upper end of the range.
            num (int):
                Number of mesh points to be generated.
            scale (str):
                'log' or 'normal'.

        Returns:
            List[float]:
                Probability mesh points generated (in log2).
        """
        assert num > 1, "must generate more than one mesh point"
        assert low > 0 and low <= 1, "low must be a non-zero probability"
        assert high > 0 and high <= 1, "high must be a non-zero probability"
        if scale == 'log':
            low = math.log2(low)
            high = math.log2(high)
            inc = (high - low) / (num - 1)
            return low + inc * np.arange(num)
        else:
            inc = (high - low) / (num - 1)
            return np.log2(low + inc * np.arange(num))

    def from_sample(self, sample, num=50, stop=1):
        """
        Generates evenly spread out percentiles of the sampled log probabilities.
        Discards any log probabilities less than stop.

        Paremeters:
            sample (List[str, float]):
                Sample drawn from password guessing model.
            num (int):
                Number of mesh points to generate.
            stop (float, optional):
                Threshold for log probabilities.
                Any log probabilities below it will be discarded and the percentiles will be calculated based on the remaining samples.
                Useful when plotting the guessing curves: prevents extremely small mesh points causing high guessing numbers that ruins the scale of the plot.

        Returns:
            List[float]:
                Probability mesh points generated (in log2).
        """
        assert num > 1, "must generate more than one mesh point"
        logprobs = np.array([lp for _, lp in sample])
        if stop != 1:
            logprobs = logprobs[np.where(logprobs >= stop)[0]]
        inc = 100 / (num - 1)
        percs = np.arange(num) * inc
        percs[-1] = 100

        return np.percentile(logprobs, percs)

