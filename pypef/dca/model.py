#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution

# Contains Python code used for our 'hybrid modeling' paper,
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

import os
import pickle
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray
from itertools import chain
from pypef.utils.variant_data import get_sequences_from_file
from pypef.utils.variant_data import remove_nan_encoded_positions
from pypef.dca.encoding import DCAEncoding, get_dca_data_parallel, get_encoded_sequence


np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
# how about new predictions? function that converts variant name to encoded sequence needed


class DCAHybridModel:
    alphas = np.logspace(-6, 6, 100)  # Grid for the parameter 'alpha'.
    parameter_range = [(0, 1), (0, 1)]  # Parameter range of 'beta_1' and 'beta_2' with lb <= x <= ub

    def __init__(
            self,
            sep: str = ';',
            alphas=alphas,
            parameter_range=None,
            X_train: np.ndarray = None,
            y_train: np.ndarray = None,
            X_test: np.ndarray = None,  # not necessary for training
            y_test: np.ndarray = None,  # not necessary for training
            X_wt = None
    ):
        if parameter_range is None:
            parameter_range = parameter_range
        self._alphas = alphas
        self._parameter_range = parameter_range
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = self.X_train
        self.y = self.y_train
        self.x_wild_type = X_wt
        self._spearmanr_dca = self._spearmanr_dca()

    @staticmethod
    def spearmanr(
            y1: np.ndarray,
            y2: np.ndarray
    ) -> float:
        """
        Parameters
        ----------
        y1 : np.ndarray
            Array of target fitness values.
        y2 : np.ndarray
            Array of predicted fitness values.

        Returns
        -------
        Spearman's rank correlation coefficient.
        """
        return spearmanr(y1, y2)[0]

    @staticmethod
    def _standardize(
            x: np.ndarray,
            axis=0
    ) -> np.ndarray:
        """
        Standardizes the input array x by subtracting the mean
        and dividing it by the (sample) standard deviation.

        Parameters
        ----------
        x : np.ndarray
            Array to be standardized.
        axis : integer (default=0)
            Axis to exectute operations on.

        Returns
        -------
        Standardized version of 'x'.
        """
        return np.subtract(x, np.mean(x, axis=axis)) / np.std(x, axis=axis, ddof=1)

    #@staticmethod
    #def _process_df_encoding(df_encoding) -> tuple:
    #    """
    #    Extracts the array of names, encoded sequences, and fitness values
    #    of the variants from the dataframe 'self.df_encoding'.
#
    #    It is mandatory that 'df_encoding' contains the names of the
    #    variants in the first column, the associated fitness value in the
    #    second column, and the encoded sequence starting from the third
    #    column.
#
    #    Returns
    #    -------
    #    Tuple of variant names, encoded sequences, and fitness values.
    #    """
    #    return (
    #        df_encoding.iloc[:, 0].to_numpy(),
    #        df_encoding.iloc[:, 2:].to_numpy(),
    #        df_encoding.iloc[:, 1].to_numpy()
    #    )

    def _delta_X(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Substracts for each variant the encoded wild-type sequence
        from its encoded sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Array of encoded variant sequences.

        Returns
        -------
        Array of encoded variant sequences with substracted encoded
        wild-type sequence.
        """
        # print('(sum(X[0]) - sum(self.x_wild_type)) =', sum(X[0]) - sum(self.x_wild_type))
        return np.subtract(X, self.x_wild_type)

    def _delta_E(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the difference of the statistical energy 'dE'
        of the variant and wild-type sequence.

        dE = E (variant) - E (wild-type)
        with E = \sum_{i} h_i (o_i) + \sum_{i<j} J_{ij} (o_i, o_j)

        Parameters
        ----------
        X : np.ndarray
            Array of the encoded variant sequences.

        Returns
        -------
        Difference of the statistical energy between variant 
        and wild-type.
        """
        # print('np.sum(self._delta_X(X), axis=1)[0] = ', np.sum(self._delta_X(X), axis=1)[:5])
        return np.sum(self._delta_X(X), axis=1)

    def _spearmanr_dca(self) -> float:
        """
        Returns
        -------
        Spearman's rank correlation coefficient of the (full)
        data and the DCA predictions (difference of statistical
        energies).
        """
        y_dca = self._delta_E(self.X)
        return self.spearmanr(self.y, y_dca)

    def ridge_predictor(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
    ) -> object:
        """
        Sets the parameter 'alpha' for ridge regression.

        Parameters
        ----------
        X_train : np.ndarray
            Array of the encoded sequences for training.
        y_train : np.ndarray
            Associated fitness values to the sequences present
            in 'X_train'.

        Returns
        -------
        Ridge object trained on 'X_train' and 'y_train' (cv=5)
        with optimized 'alpha'. 
        """
        grid = GridSearchCV(Ridge(), {'alpha': self._alphas}, cv=5)
        grid.fit(X_train, y_train)
        return Ridge(**grid.best_params_).fit(X_train, y_train)

    def _y_hybrid(
            self,
            y_dca: np.ndarray,
            y_ridge: np.ndarray,
            beta_1: float,
            beta_2: float
    ) -> np.ndarray:
        """
        Chooses sign for connecting the parts of the hybrid model.

        Parameters
        ----------
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.
        b1 : float
            Float between [0,1] coefficient for regulating DCA
            model contribution.
        b2 : float
            Float between [0,1] coefficient for regulating ML
            model contribution.

        Returns
        -------
        The predicted fitness value-representatives of the hybrid
        model.
        """
        if self._spearmanr_dca >= 0:
            return beta_1 * y_dca + beta_2 * y_ridge

        else:
            return beta_1 * y_dca - beta_2 * y_ridge

    def _adjust_betas(
            self,
            y: np.ndarray,
            y_dca: np.ndarray,
            y_ridge: np.ndarray
    ) -> np.ndarray:
        """
        Find parameters that maximize the absolut Spearman rank
        correlation coefficient using differential evolution.

        Parameters
        ----------
        y : np.ndarray
            Array of fitness values.
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.

        Returns
        -------
        'beta_1' and 'beta_2' that maximize the absolut Spearman rank correlation
        coefficient.
        """
        loss = lambda b: -np.abs(self.spearmanr(y, b[0] * y_dca + b[1] * y_ridge))
        minimizer = differential_evolution(loss, bounds=self.parameter_range, tol=1e-4)
        return minimizer.x

    def settings(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            train_size_train=0.66,
            random_state=42
    ) -> tuple:
        """
        Get the adjusted parameters 'beta_1', 'beta_2', and the
        tuned regressor of the hybrid model.

        Parameters
        ----------
        X_train : np.ndarray
            Encoded sequences of the variants in the training set.
        y_train : np.ndarray
            Fitness values of the variants in the training set.
        X_test : np.ndarray
            Encoded sequences of the variants in the testing set.
        y_test : np.ndarray
            Fitness values of the variants in the testing set.
        train_size_train : float [0,1] (default 0.66)
            Fraction to split training set into another
            training and testing set.
        random_state : int (default=224)
            Random state used to split.

        Returns
        -------
        Tuple containing the adjusted parameters 'beta_1' and 'beta_2',
        as well as the tuned regressor of the hybrid model.
        """
        try:
            X_ttrain, X_ttest, y_ttrain, y_ttest = train_test_split(
                X_train, y_train,
                train_size=train_size_train,
                random_state=random_state
            )

        except ValueError:
            """
            Not enough sequences to construct a sub-training and sub-testing 
            set when splitting the training set.

            Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
            possible -> return parameter setting for 'EVmutation' model.
            """
            return 1.0, 0.0, None

        """
        The sub-training set 'y_ttrain' is subjected to a five-fold cross 
        validation. This leads to the constraint that at least two sequences
        need to be in the 20 % of that set in order to allow a ranking. 

        If this is not given -> return parameter setting for 'EVmutation' model.
        """
        y_ttrain_min_cv = int(0.2 * len(y_ttrain))  # 0.2 because of five-fold cross validation (1/5)
        if y_ttrain_min_cv < 2:
            return 1.0, 0.0, None

        y_dca_ttest = self._delta_E(X_ttest)

        ridge = self.ridge_predictor(X_ttrain, y_ttrain)
        y_ridge_ttest = ridge.predict(X_ttest)

        beta1, beta2 = self._adjust_betas(y_ttest, y_dca_ttest, y_ridge_ttest)
        return beta1, beta2, ridge

    def predict(
            self,
            X: np.ndarray,
            reg: object,  # any regression-based estimator (from sklearn)
            beta_1: float,
            beta_2: float
    ) -> np.ndarray:
        """
        Use the regressor 'reg' and the parameters 'beta_1'
        and 'beta_2' for constructing a hybrid model and
        predicting the fitness associates of 'X'.

        Parameters
        ----------
        X : np.ndarray
            Encoded sequences used for prediction.
        reg : object
            Tuned ridge regressor for the hybrid model.
        beta_1 : float
            Float for regulating EVmutation model contribution.
        beta_2 : float
            Float for regulating Ridge regressor contribution.

        Returns
        -------
        Predicted fitness associates of 'X' using the
        hybrid model.
        """
        y_dca = self._delta_E(X)
        if reg is None:
            y_ridge = np.random.random(len(y_dca))  # in order to suppress error
        else:
            y_ridge = reg.predict(X)
        return self._y_hybrid(y_dca, y_ridge, beta_1, beta_2)

    def split_performance(
            self,
            train_size: float = 0.8,
            n_runs: int = 10,
            seed: int = 42,
            save_model: bool = False
    ) -> dict:
        """
        Estimates performance of the model.

        Parameters
        ----------
        train_size : int or float (default=0.8)
            Number of samples in the training dataset
            or fraction of full dataset used for training.
        n_runs : int (default=10)
            Number of different splits to perform.
        seed : int (default=42)
            Seed for random generator.
        verbose : bool (default=False)
            Controls information content to be returned.
        save_model : bool (default=False)
            If True, model is saved using pickle, else not.

        Returns
        -------
        Returns array of spearman rank correlation coefficients
        if verbose=False, otherwise returns array of spearman
        rank correlation coefficients, cs, alphas, number of 
        samples in the training and testing set, respectively.
        """
        data = {}
        np.random.seed(seed)

        for r, random_state in enumerate(np.random.randint(100, size=n_runs)):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=train_size, random_state=random_state)
            beta_1, beta_2, reg = self.settings(X_train, y_train)
            print(beta_1, beta_2, reg)
            if beta_2 == 0.0:
                alpha = np.nan
            else:
                if save_model:
                    pickle.dumps(reg)
                alpha = reg.alpha
            data.update(
                {f'{len(y_train)}_{r}':
                    {
                        'no_run': r,
                        'n_y_train': len(y_train),
                        'n_y_test': len(y_test),
                        'rnd_state': random_state,
                        'spearman_rho': self.spearmanr(y_test, self.predict(X_test, reg, beta_1, beta_2)),
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'alpha': alpha
                    }
                }
            )

        return data

    def ls_ts_performance(
            self,
            data=None
    ):
        if data is None:
            data = {}
        beta_1, beta_2, reg = self.settings(self.X_train, self.y_train)
        print(f'beta_1 (DCA): {beta_1:.3f}, beta_2 (ML): {beta_2:.3f}, '
              f'regressor: Ridge(alpha={reg.alpha:.3f})')
        if beta_2 == 0.0:
            alpha = np.nan
        else:
            alpha = reg.alpha
        data.update(
            {f'ls_ts':
                {
                    'n_y_train': len(self.y_train),
                    'n_y_test': len(self.y_test),
                    'spearman_rho': self.spearmanr(
                        self.y_test, self.predict(self.X_test, reg, beta_1, beta_2)),
                    'beta_1': beta_1,
                    'beta_2': beta_2,
                    'alpha': alpha
                }
            }
        )

        return data, self.predict(self.X_test, reg, beta_1, beta_2)

    def train_and_save_pkl(
            self,
            train_percent_train: float = 0.66,
            random_state: int = 42
    ):
        """
        Description
        ----------
        Trains the hybrid model on a relative number of all variants
        and returns the individual model contribution weights beta_1 (DCA)
        and beta_2 (ML) as well as the hyperparameter-tuned regression model,
        e.g. to save all the hybrid model parameters for later loading as
        Pickle file.

        Parameters
        ----------
        train_percent_train: float (default = 0.66)
            Relative number of variants used for model fitting (not
            hyperparameter validation. Default of 0.66 and overall train
            size of 0.8 means the total size for least squares fitting
            is 0.8 * 0.66 = 0.528, thus for hyperparameter validation
            the size is 0.8 * 0.33 = 0.264 and for testing the size is
            1 - 0.528 - 0.264 = 0.208.
        random_state: int (default = 42)
            Random state for splitting (and reproduction of results).

        Returns
        ----------
        beta_1: float
            DCA model contribution to hybrid model predictions.
        beta_2: float
            ML model contribution to hybrid model predictions.
        reg: object
            sklearn Estimator class, e.g. sklearn.linear_model.Ridge
            fitted and with optimized hyperparameters (e.g. alpha).
        self._spearmanr_dca: float
            To determine, if spearmanr_dca (i.e. DCA correlation) and measured
            fitness values is positive (>= 0) or negative (< 0).
        test_spearman_r : float
            Achieved performance in terms of Spearman's rank correlation
            between measured and predicted test set variant fitness values.
        """
        beta_1, beta_2, reg = self.settings(
            X_train=self.X_train,
            y_train=self.y_train,
            train_size_train=train_percent_train,
            random_state=random_state
        )

        if len(self.y_test) > 0:
            test_spearman_r = self.spearmanr(
                self.y_test,
                self.predict(self.X_test, reg, beta_1, beta_2)
            )
        else:
            test_spearman_r = None

        return beta_1, beta_2, reg, self._spearmanr_dca, test_spearman_r

    def get_train_sizes(self) -> np.ndarray:
        """
        Generates a list of train sizes to perform low-n with.

        Returns
        -------
        Numpy array of train sizes up to 80% (i.e. 0.8 * N_variants).
        """
        eighty_percent = int(len(self.y) * 0.8)

        train_sizes = np.sort(np.concatenate([
            np.arange(15, 50, 5), np.arange(50, 100, 10),
            np.arange(100, 150, 20), [160, 200, 250, 300, eighty_percent],
            np.arange(400, 1100, 100)
        ]))

        idx_max = np.where(train_sizes >= eighty_percent)[0][0] + 1
        return train_sizes[:idx_max]

    def run(
            self,
            train_sizes: list = None,
            n_runs: int = 10
    ) -> dict:
        """

        Returns
        ----------
        data: dict
            Performances of the split with size of the
            training set = train_size and size of the
            test set = N_variants - train_size.
        """
        data = {}
        for t, train_size in enumerate(train_sizes):
            print(t + 1, '/', len(train_sizes), ':', train_size)
            data.update(self.split_performance(train_size=train_size, n_runs=n_runs))

        return data


"""
Below: Some helper functions that call or are dependent on the DCAHybridModel class.
"""


def get_model_and_save_pkl(
        Xs,
        ys,
        dca_encoding_instance: DCAEncoding,
        train_percent_fitting: float = 0.66,  # percent of all data: 0.8 * 0.66
        test_percent: float = 0.2,
        random_state: int = 42
):
    """
    Description
    -----------
    Save (Ridge) regression model (trained and with tuned alpha parameter)
    with betas (beta_1 and beta_2) as dictionary-structured pickle file.

    Parameters
    ----------
    test_percent:
    df: pandas.DataFrame
        DataFrame of form Substitution;Fitness;Encoding_Features.
    train_percent: float
        Percent of DataFrame data to train on.
        The remaining data is used for validation.
    random_state: int
        Random seed for splitting in train and test data for reproducing results.

    Returns
    ----------
    None
        Just saving model parameters as pickle file.
    """

    # getting target (WT) sequence and encoding it to provide it as
    # relative value for pure DCA based predictions (difference in sums
    # of sequence encodings: variant - WT)
    target_seq, index = dca_encoding_instance.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    wt_X = get_encoded_sequence(wt_name, dca_encoding_instance)

    print(f'Train size (fitting) :{train_percent_fitting} % '
          f'(overall {(1 - test_percent)*train_percent_fitting} %), '
          f'Train size validation: {1 - train_percent_fitting} % '
          f'(overall {(1 - test_percent)*(1 - train_percent_fitting)} %), '
          f'Test size: {test_percent} %, '
          f'(overall {test_percent} %), '
          f'Random state: {random_state}...\n')

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=test_percent)

    model = DCAHybridModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_wt=wt_X
    )

    beta_1, beta_2, reg, spearman_dca, test_spearman_r = model.train_and_save_pkl(
        train_percent_train=train_percent_fitting, random_state=random_state)
    print(f'Individual model weigths and regressor hyperparameters:')
    print(f'beta_1 (DCA): {beta_1:.3f}, beta_2 (ML): {beta_2:.3f}, '
          f'regressor: Ridge(alpha={reg.alpha:.3f})')
    print('Testing performance...')
    print(f'Spearman\'s rho = {test_spearman_r:.4f}')
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass
    print(f'Save model as Pickle file... HYBRIDMODEL')  # CONSTRUCTION : Name Pickle file as PLMC paramas file?
    pickle.dump(
        {'dca_hybrid_model': model, 'beta_1': beta_1, 'beta_2': beta_2,
         'spearman_dca': spearman_dca, 'reg': reg},
        open('Pickles/HYBRIDMODEL', 'wb')
    )


def plot_y_true_vs_y_pred(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        variants: np.ndarray,  # just required for labeling
        label=False
):
    spearman_rho = spearmanr(y_true, y_pred)[0]
    print(spearman_rho)

    figure, ax = plt.subplots()
    ax.scatter(y_true, y_pred, marker='o', s=20, linewidths=0.5, edgecolor='black',
               label=fr'$\rho$ = {spearman_rho:.3f}')
    ax.legend()
    ax.set_xlabel(r'$y_\mathrm{true}$' + fr' ($N$ = {len(y_true)})')
    ax.set_ylabel(r'$y_\mathrm{pred}$' + fr' ($N$ = {len(y_pred)})')
    print('Plotting...')
    if label:
        from adjustText import adjust_text
        print('Adjusting variant labels for plotting can take some '
              'time (the limit for labeling is 150 data points)...')
        if len(y_true) < 150:
            texts = [ax.text(y_true[i], y_pred[i], txt, fontsize=4)
                     for i, txt in enumerate(variants)]
            adjust_text(
                texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5, lim=250)
        else:
            print("Terminating label process. Too many variants "
                  "(> 150) for plotting (labels would overlap).")
    file_name = 'DCA_Hybrid_Model_LS_TS_Performance.png'
    i = 1
    while os.path.isfile(file_name):
        i += 1  # iterate until finding an unused file name
        file_name = f'DCA_Hybrid_Model_LS_TS_Performance({i}).png'
    plt.savefig(file_name, dpi=500)


def performance_ls_ts(
        ls_fasta: str,
        ts_fasta: str,
        threads: int,
        starting_position: int,
        params_file: str,
        separator: str,
        label: bool
):
    """
    Description
    -----------
    Computes performance based on a (linear) regression model trained
    on the training set by optimizing model hyperparameters based on
    validation performances on training subsets (default: 5-fold CV)
    and predicting test set entries using the hyperparmeter-tuned model
    to estimate performance for model generalization.

    Parameters
    -----------
    ls_fasta: str
        Fasta-like file with fitness values. Will be read and extracted
        for training the regressor.
    ts_fasta: str
        Fasta-like file with fitness values. Used for computing performance
        of the tuned regressor for test set entries (performance metric of
        measured and predicted fitness values).
    n_cores: int
        Number of threads to use for parallel computing using Ray.
    starting_position: int
        Shift os position (not really needed, error tells user wrong input).
    params_file: str
        PLMC parameter file (containing evolutionary, i.e. MSA-based local
        and coupling terms.
    separator: str
        Character to split the variant to obtain the single substitutions
        (default='/').
    label: bool
        Labeling plotted predicted vs. measured variants.
    y_wt: float = None
        Wild-type fitness.

    Returns
    -----------
    None
        Just plots test results (predicted fitness vs. measured fitness)
        using def plot_y_true_vs_y_pred.
    """
    _, train_variants, y_train = get_sequences_from_file(ls_fasta)
    _, test_variants, y_test = get_sequences_from_file(ts_fasta)

    dca_encode = DCAEncoding(
        params_file=params_file,
        starting_position=starting_position,
        separator=separator
    )

    # DCA prediction: delta E = np.subtract(X, self.x_wild_type),
    # with X = encoded sequence of any variant -->
    # getting wild-type name und subsequently x_wild_type
    # to provide it for the DCAHybridModel
    target_seq, index = dca_encode.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    x_wt = get_encoded_sequence(wt_name, dca_encode)
    if threads > 1:
        # NaNs are already being removed by the called function
        train_variants, x_train, y_train = get_dca_data_parallel(
            train_variants, y_train, dca_encode, threads)
        test_variants, x_test, y_test = get_dca_data_parallel(
            test_variants, y_test, dca_encode, threads)
    else:
        x_train_ = dca_encode.collect_encoded_sequences(train_variants)
        x_test_ = dca_encode.collect_encoded_sequences(test_variants)
        # NaNs must still be removed
        x_train, train_variants = remove_nan_encoded_positions(x_train_, train_variants)
        x_train, y_train = remove_nan_encoded_positions(x_train_, y_train)
        x_test, test_variants = remove_nan_encoded_positions(x_test_, test_variants)
        x_test, y_test = remove_nan_encoded_positions(x_test_, y_test)
        assert len(x_train) == len(train_variants) == len(y_train)
        assert len(x_test) == len(test_variants) == len(y_test)

    hybrid_model = DCAHybridModel(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        X_wt=x_wt
    )

    data, y_pred = hybrid_model.ls_ts_performance()
    y_pred = np.abs(y_pred)

    plot_y_true_vs_y_pred(y_test, y_pred, test_variants, label)
    hybrid_model.train_and_save_pkl()


def predict_ps(  # "predict pmult"
        prediction_dict: dict,
        params_file: str,
        prediction_set: str,
        test_set: str,
        threads: int,
        starting_position: int,
        separator: str,
        regressor_pkl: str
):
    """
    Description
    -----------
    Predicting the fitness of sequences of a prediction set
    or multiple prediction sets that were exemplary created with
    'pypef mkps' based on single substitutional variant data
    provided in a CSV and the wild type sequence:
        pypef mkps --wt WT_SEQ --input CSV_FILE
        [--drop THRESHOLD] [--drecomb] [--trecomb] [--qarecomb] [--qirecomb]
        [--ddiverse] [--tdiverse] [--qdiverse]

    Parameters
    -----------
    prediction_dict: dict,
    params_file: str,
    prediction_set: str,
    validation_set: str,
    n_cores: int,
    starting_position: int,
    separator: str,
    regressor_pkl: str

    Returns
    -----------
    None
        Writes sorted predictions to files (for [--drecomb] [--trecomb]
        [--qarecomb] [--qirecomb] [--ddiverse] [--tdiverse] [--qdiverse]
        in the respective created folders).

    """
    dca_encode = DCAEncoding(
        starting_position=starting_position,
        params_file=params_file,
        separator=separator
    )
    reg = pickle.load(open('Pickles/' + regressor_pkl, "rb"))
    if type(reg) == dict:
        reg = reg['reg']    # CONSTRUCTION: PREDICT USING HYBRID MODEL PREDICTION FUNCTION
    print(f'Taking regression model from Pickle file: {reg}...')
    write_to_file = lambda f, ff: open(f'{f}/variants_fitness.csv', 'a').write(ff)
    pmult = ['Recomb_Double_Split', 'Recomb_Triple_Split', 'Recomb_Quadruple_Split', 'Recomb_Quintuple_Split',
             'Diverse_Double_Split', 'Diverse_Triple_Split', 'Diverse_Quadruple_Split']
    print(prediction_dict.values(), pmult)
    if True in prediction_dict.values():
        for ps, path in zip(prediction_dict.values(), pmult):
            print(str(path) + ': ' + str(ps))
            if ps:
                write_to_file(path, 'variant;y_pred\n')
                print('ENTERING:', str(path) + ': ' + str(ps))
                for file in os.listdir(path):
                    print('file:', file)
                    if not file.endswith('.csv'):
                        file_path = os.path.join(path, file)
                        sequences, variants, _ = get_sequences_from_file(file_path)
                        if threads > 1:
                            # NaNs are already being removed by the called function
                            variants, xs, _ = get_dca_data_parallel(
                                variants, list(np.zeros(len(variants))), dca_encode, threads)
                        else:
                            xs = dca_encode.collect_encoded_sequences(variants)
                            # NaNs must still be removed
                            xs, variants = remove_nan_encoded_positions(xs, variants)
                        for i, (variant, x) in enumerate(zip(variants, xs)):
                            write_to_file(path, f'{variant};{reg.predict([x])[0]}\n')
            else:
                continue
    elif prediction_set:
        sequences, variants, _ = get_sequences_from_file(prediction_set)
        # NaNs are already being removed by the called function
        if threads > 1:
            # NaNs are already being removed by the called function
            variants, xs, _ = get_dca_data_parallel(
                variants, list(np.zeros(len(variants))), dca_encode, threads)
        else:
            xs = dca_encode.collect_encoded_sequences(variants)
            # NaNs must still be removed
            xs, variants = remove_nan_encoded_positions(xs, variants)
        for i, (variant, x) in enumerate(zip(variants, xs)):
            write_to_file(prediction_set + 'predicted.txt', f'{variant};{reg.predict([x])[0]}\n')

    elif test_set:
        sequences, variants, y_true = get_sequences_from_file(test_set)
        # NaNs are already being removed by the called function
        if threads > 1:
            variants, xs, y_true = get_dca_data_parallel(
                variants, y_true, dca_encode, threads)
        else:
            # NaNs must still be removed
            xs_ = dca_encode.collect_encoded_sequences(variants)
            xs, variants = remove_nan_encoded_positions(xs_, variants)
            xs, y_test = remove_nan_encoded_positions(xs_, y_true)
            assert len(xs) == len(variants) == len(y_test)

        y_pred = reg.predict(xs)


        #n_y_true_dict = dict(zip(variants, y_true))
        #n_y_pred_dict = dict(zip(variants, y_pred))
        #y_true = []
        #y_pred = []
        #variants = []
        #for key_true, value_true in n_y_true_dict.items():
        #    for key_pred, value_pred in n_y_pred_dict.items():
        #        if key_true == key_pred:
        #            y_true.append(value_true)
        #            y_pred.append(value_pred)
        #            variants.append(key_true)

        print(f'Spearman\'s rho = {spearmanr(y_true, y_pred)[0]:.3f}')
        plot_y_true_vs_y_pred(np.array(y_true), np.array(y_pred), np.array(variants))


def predict_directed_evolution(
        encoder: DCAEncoding,
        variant: str,
        regressor_pkl: str
):
    """

    """
    X = encoder.encode_variant(variant)

    #print(variant[0] + variant[1:-1] + variant[0])
    model_dict = pickle.load(open('Pickles/' + regressor_pkl, "rb"))
    model = model_dict['dca_hybrid_model']
    ridge = model_dict['reg']
    #print(ridge.coef_)
    #print(ridge.intercept_)
    beta_1 = model_dict['beta_1']
    beta_2 = model_dict['beta_2']

    y_pred = model.predict(  # 2d as only single variant
        X=np.atleast_2d(X),  # e.g., np.atleast_2d(3.0) --> array([[3.]])
        reg=ridge,  # RidgeRegressor
        beta_1=beta_1,  # DCA model prediction weight
        beta_2=beta_2  # ML model prediction weight
    )[0]

    return [(y_pred, variant[1:])]
