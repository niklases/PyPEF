#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <niklas.siedhoff@rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF
# Licensed under Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099

# Contains Python code used for the approach presented in our 'hybrid modeling' paper
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

from __future__ import annotations


import pickle
import copy
import logging
logger = logging.getLogger('pypef.dca.hybrid_model')

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.optimize import differential_evolution
import torch
from peft import LoraConfig, get_peft_model
from transformers import EsmForMaskedLM, EsmTokenizer

from esm1v_contrastive_learning import get_encoded_seqs, get_batches, train, test, infer, corr_loss



def reduce_by_batch_modulo(a: np.ndarray, batch_size=5) -> np.ndarray:
    reduce = len(a) - (len(a) % batch_size)
    return a[:reduce]



# TODO: Implementation of other regression techniques (CVRegression models)
# TODO: Differential evolution of multiple Zero Shot predictors
#       (and supervised model predictions thereof) and y_true
class DCAESMHybridModel:
    def __init__(
            self,
            x_train_dca: np.ndarray,               # DCA-encoded sequences
            x_train_esm: np.ndarray, 
            x_train_esm_attention_masks: np.ndarray,
            y_train: np.ndarray,                   # true labels
            esm_base_model,
            esm_model,
            esm_optimizer,
            x_wt: np.ndarray | None = None,        # Wild type encoding
            alphas: np.ndarray | None = None,      # Ridge regression grid for the parameter 'alpha'
            parameter_range: list | None = None    # Parameter range of 'beta_1' and 'beta_2' with lower bound <= x <= upper bound,
    ):
        if parameter_range is None:
            parameter_range = [(0, 1), (0, 1), (0, 1), (0, 1)] 
        if alphas is None:
            alphas = np.logspace(-6, 6, 100)
        self.parameter_range = parameter_range
        self.alphas = alphas
        self.x_train_dca = x_train_dca
        self.y_train = y_train
        self.x_wild_type = x_wt
        self.x_train_esm = x_train_esm
        self.x_train_esm_attention_masks = x_train_esm_attention_masks
        self.ridge_opt, self.beta1, self.beta2, self.beta3, self.beta4 = None, None, None, None, None
        self.esm_base_model = esm_base_model
        self.esm_model = esm_model
        self.esm_optimizer = esm_optimizer
        self.train_and_optimize()

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

    def _delta_x(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Substracts for each variant the encoded wild-type sequence
        from its encoded sequence.
        
        Parameters
        ----------
        x : np.ndarray
            Array of encoded variant sequences (matrix X).

        Returns
        -------
        Array of encoded variant sequences with substracted encoded
        wild-type sequence.
        """
        return np.subtract(x, self.x_wild_type)

    def _delta_e(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the difference of the statistical energy 'dE'
        of the variant and wild-type sequence.

        dE = E (variant) - E (wild-type)
        with E = sum_{i} h_i (o_i) + sum_{i<j} J_{ij} (o_i, o_j)

        Parameters
        ----------
        X : np.ndarray
            Array of the encoded variant sequences.

        Returns
        -------
        Difference of the statistical energy between variant 
        and wild-type.
        """
        return np.sum(self._delta_x(x), axis=1)

    def _spearmanr_dca(self) -> float:
        """
        Returns
        -------
        Spearman's rank correlation coefficient of the full
        data and the statistical DCA predictions (difference
        of statistical energies). Used to adjust the sign
        of hybrid predictions, i.e.
            beta_1 * y_dca + beta_2 * y_ridge
        or
            beta_1 * y_dca - beta_2 * y_ridge.
        """
        y_dca = self._delta_e(self.x_train_dca)
        return self.spearmanr(self.y_train, y_dca)

    def ridge_predictor(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
    ) -> object:
        """
        Sets the parameter 'alpha' for ridge regression.

        Parameters
        ----------
        x_train : np.ndarray
            Array of the encoded sequences for training.
        y_train : np.ndarray
            Associated fitness values to the sequences present
            in 'x_train'.

        Returns
        -------
        Ridge object trained on 'x_train' and 'y_train' (cv=5)
        with optimized 'alpha'. 
        """
        grid = GridSearchCV(Ridge(), {'alpha': self.alphas}, cv=5)
        grid.fit(x_train, y_train)
        return Ridge(**grid.best_params_).fit(x_train, y_train)

    def _adjust_betas(
            self,
            y: np.ndarray,
            y_dca: np.ndarray,
            y_ridge: np.ndarray,
            y_esm1v: np.ndarray,
            y_esm1v_lora: np.ndarray
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
        loss = lambda params: -np.abs(self.spearmanr(y, params[0] * y_dca + params[1] * y_ridge + params[2] * y_esm1v + y_esm1v_lora * params[3]))
        minimizer = differential_evolution(loss, bounds=self.parameter_range, tol=1e-4)
        return minimizer.x

    def train_and_optimize(
            self,
            train_size_fit: float = 0.66,
            random_state: int = 42,
            batch_size: int = 5
    ) -> tuple:
        """
        Get the adjusted parameters 'beta_1', 'beta_2', and the
        tuned regressor of the hybrid model.

        Parameters
        ----------
        x_train : np.ndarray
            Encoded sequences of the variants in the training set.
        y_train : np.ndarray
            Fitness values of the variants in the training set.
        train_size_fit : float [0,1] (default 0.66)
            Fraction to split training set into another
            training and testing set.
        random_state : int (default=224)
            Random state used to split.

        Returns
        -------
        Tuple containing the adjusted parameters 'beta_1' and 'beta_2',
        as well as the tuned regressor of the hybrid model.
        """
        #try:
        print('Orig. train size:', int(train_size_fit * len(self.y_train)))
        train_size_fit = int((train_size_fit * len(self.y_train)) - ((train_size_fit * len(self.y_train)) % batch_size))
        print('New train size:', train_size_fit)
        print('Remaining for testing:', len(self.y_train) - train_size_fit)
        train_test_size = int((len(self.y_train) - train_size_fit) - ((len(self.y_train) - train_size_fit) % batch_size))
        print('New test size:', train_test_size)


        (
                x_dca_ttrain, x_dca_ttest, 
                x_esm1v_ttrain, x_esm1v_ttest,
                attn_esm_1v_ttrain, attn_esm_1v_ttest,
                y_ttrain, y_ttest
        ) = train_test_split(
                self.x_train_dca, 
                self.x_train_esm,
                self.x_train_esm_attention_masks,
                self.y_train, 
                train_size=train_size_fit,
                random_state=random_state
        )
        x_dca_ttest = x_dca_ttest[:train_test_size]
        x_esm1v_ttest = x_esm1v_ttest[:train_test_size]
        attn_esm_1v_ttest = attn_esm_1v_ttest[:train_test_size]
        y_ttest = y_ttest[:train_test_size]

        #except ValueError:
        """
            Not enough sequences to construct a sub-training and sub-testing 
            set when splitting the training set.

            Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
            possible -> return parameter setting for 'EVmutation/GREMLIN' model.
        """
            #return 1.0, 0.0, 1.0, 0.0, None

        """
        The sub-training set 'y_ttrain' is subjected to a five-fold cross 
        validation. This leads to the constraint that at least two sequences
        need to be in the 20 % of that set in order to allow a ranking. 

        If this is not given -> return parameter setting for 'EVmutation/GREMLIN' model.
        """
        # int(0.2 * len(y_ttrain)) due to 5-fold-CV for adjusting the (Ridge) regressor
        y_ttrain_min_cv = int(0.2 * len(y_ttrain))
        if y_ttrain_min_cv < 5:
            return 1.0, 0.0, 1.0, 0.0, None

        y_dca_ttest = self._delta_e(x_dca_ttest)
        self.ridge_opt = self.ridge_predictor(x_dca_ttrain, y_ttrain)
        y_ridge_ttest = self.ridge_opt.predict(x_dca_ttest)

        # LoRA training on y_esm1v_ttrain
        # --> Testing on y_esm1v_ttest 
        # for ß optimization y_esm1v_ttest (raw ESM1v-LLM predicted values)
        # and y_esm1v_lora_ttest (LoRA-optimized predicted ESM1v values)


        x_esm1v_ttrain_b, attns_ttrain_b, scores_ttrain_b = (
            get_batches(x_esm1v_ttrain, batch_size=batch_size), 
            get_batches(attn_esm_1v_ttrain, batch_size=batch_size), 
            get_batches(y_ttrain, batch_size=batch_size)
        )
        x_esm1v_ttest_b, attns_ttest_b, scores_ttest_b = (
            get_batches(x_esm1v_ttest, batch_size=batch_size), 
            get_batches(attn_esm_1v_ttest, batch_size=batch_size), 
            get_batches(y_ttest, batch_size=batch_size)
        )

        y_ttest_, y_esm_ttest = test(x_esm1v_ttest_b, attns_ttest_b, scores_ttest_b, loss_fn=corr_loss, model=self.esm_base_model)
        print(f'Hybrid opt. Test-perf. (untrained, N={len(y_ttest_)}):', spearmanr(y_ttest_.cpu(), y_esm_ttest.cpu()))

        train(x_esm1v_ttrain_b, attns_ttrain_b, scores_ttrain_b, loss_fn=corr_loss, model=self.esm_model, optimizer=self.esm_optimizer, n_epochs=5)
        y_ttrain_, y_ttrain_esm1v_pred = test(x_esm1v_ttrain_b, attns_ttrain_b, scores_ttrain_b, loss_fn=corr_loss, model=self.esm_model)
        print(f'Hybrid opt. LoRA Train-perf., N={len(y_ttrain_.cpu())}:', spearmanr(y_ttrain_.cpu(), y_ttrain_esm1v_pred.cpu()))
        y_ttest_, y_esm_lora_ttest = test(x_esm1v_ttest_b, attns_ttest_b, scores_ttest_b, loss_fn=corr_loss, model=self.esm_model)
        print(f'Hybrid opt. LoRA Test-perf., N={len(y_ttest_.cpu())}:', spearmanr(y_ttest_.cpu(), y_esm_lora_ttest.cpu()))

        # Hybrid DCA model performance (N Train: 200, N Test: 6027). Spearman's rho: 0.519 (Unnorm.), 0.496 normal., 0.496 min-max norm.
        self.beta1, self.beta2, self.beta3, self.beta4 = self._adjust_betas(
            y_ttest, y_dca_ttest, y_ridge_ttest, y_esm_ttest.detach().cpu().numpy(), y_esm_lora_ttest.detach().cpu().numpy()
        )  # .cpu() ?
        return self.beta1, self.beta2, self.beta3, self.beta4, self.ridge_opt  #, raw and optimized ESM1v model

    def hybrid_prediction(
            self,
            x_dca: np.ndarray,
            x_esm: np.ndarray,
            attns_esm: np.ndarray,
            batch_size: int = 5
    ) -> np.ndarray:
        """
        Use the regressor 'reg' and the parameters 'beta_1'
        and 'beta_2' for constructing a hybrid model and
        predicting the fitness associates of 'X'.

        Parameters
        ----------
        x : np.ndarray
            Encoded sequences X used for prediction.
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
        y_dca = self._delta_e(x_dca)
        if self.ridge_opt is None:
            y_ridge = np.zeros(len(y_dca))  # in order to suppress error
        else:
            y_ridge = self.ridge_opt.predict(x_dca)

        x_esm_b, attns_b = (
            get_batches(x_esm, batch_size=batch_size), 
            get_batches(attns_esm, batch_size=batch_size)
        )
        
        y_esm = infer(x_esm_b, attns_b, self.esm_base_model, desc='Infering base model').detach().cpu().numpy()
        y_esm_lora = infer(x_esm_b, attns_b, self.esm_model, desc='Infering LoRA-tuned model').detach().cpu().numpy()
        
        y_dca, y_ridge, y_esm, y_esm_lora = (
            reduce_by_batch_modulo(y_dca), 
            reduce_by_batch_modulo(y_ridge), 
            reduce_by_batch_modulo(y_esm), 
            reduce_by_batch_modulo(y_esm_lora)
        )
        
        # adjusting: + or - on train data --> +-beta_1 * y_dca + beta_2 * y_ridge
        return self.beta1 * y_dca + self.beta2 * y_ridge + self.beta3 * y_esm + self.beta4 * y_esm_lora

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
        save_model : bool (default=False)
            If True, model is saved using pickle, else not.

        Returns
        -------
        data : dict
            Contains information about hybrid model parameters
            and performance results.
        """
        data = {}
        np.random.seed(seed)

        for r, random_state in enumerate(np.random.randint(100, size=n_runs)):
            x_train, x_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=train_size, random_state=random_state)
            beta_1, beta_2, reg = self.settings(x_train, y_train)
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
                        'spearman_rho': self.spearmanr(
                            y_test, self.hybrid_prediction(
                                x_test, reg, beta_1, beta_2
                            )
                        ),
                        'beta_1': beta_1,
                        'beta_2': beta_2,
                        'alpha': alpha
                    }
                }
            )

        return data

    def ls_ts_performance(self):
        beta_1, beta_2, reg = self.settings(
            x_train=self.x_train,
            y_train=self.y_train
        )
        spearman_r = self.spearmanr(
            self.y_test,
            self.hybrid_prediction(self.x_test, reg, beta_1, beta_2)
        )
        self.beta_1, self.beta_2, self.regressor = beta_1, beta_2, reg
        return spearman_r, reg, beta_1, beta_2

    def train_and_test(
            self,
            train_percent_fit: float = 0.66,
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
        train_percent_fit: float (default = 0.66)
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
            x_train=self.x_train,
            y_train=self.y_train,
            train_size_fit=train_percent_fit,
            random_state=random_state
        )

        if len(self.y_test) > 0:
            test_spearman_r = self.spearmanr(
                self.y_test,
                self.hybrid_prediction(
                    self.x_test, reg, beta_1, beta_2
                )
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
            logger.info(f'{t + 1}/{len(train_sizes)}:{train_size}')
            data.update(self.split_performance(train_size=train_size, n_runs=n_runs))
        return data


def get_delta_e_statistical_model(
        x_test: np.ndarray,
        x_wt: np.ndarray
):
    """
    Description
    -----------
    Delta_E means difference in evolutionary energy in plmc terms.
    In other words, this is the delta of the sum of Hamiltonian-encoded
    sequences of local fields and couplings of encoded sequence and wild-type
    sequence in GREMLIN terms.

    Parameters
    -----------
    x_test: np.ndarray [2-dim]
        Encoded sequences to be subtracted by x_wt to compute delta E.
    x_wt: np.ndarray [1-dim]
        Encoded wild-type sequence.

    Returns
    -----------
    delta_e: np.ndarray [1-dim]
        Summed subtracted encoded sequences.

    """
    delta_x = np.subtract(x_test, x_wt)
    delta_e = np.sum(delta_x, axis=1)
    return delta_e


if __name__ == '__main__':
    import pandas as pd
    from pypef.utils.variant_data import get_seqs_from_var_name
    from pypef.dca.gremlin_inference import GREMLIN
    from pypef.hybrid.hybrid_model import get_delta_e_statistical_model

        # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"hybrid_model.py: Using {device} device")
    basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    model = get_peft_model(basemodel, peft_config)
    model = model.to(device)
    baseline_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    MAX_WT_SEQUENCE_LENGTH = 500
    N_EPOCHS = 5
    BATCH_SIZE = 5

    # Get cpu, gpu or mps device for training.
    #device = (
    #    "cuda"
    #    if torch.cuda.is_available()
    #    else "mps"
    #    if torch.backends.mps.is_available()
    #    else "cpu"
    #)
    #print(f"Using {device} device")
    #basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    #model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    #tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_3')
    #peft_config = LoraConfig(r=8, target_modules=["query", "value"])
    #model = get_peft_model(basemodel, peft_config)
    #model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #MAX_WT_SEQUENCE_LENGTH = 500
    #N_EPOCHS = 5
    #BATCH_SIZE = 5


    k = ['/mnt/d/dev/', '/home/niklas/Data/dev/'][0]
    csv_path = f'{k}contrastive_learning_test/DMS_ProteinGym_substitutions/AMIE_PSEAE_Wrenbeck_2017.csv'
    msa_path= f'{k}contrastive_learning_test/DMS_msa_files/AMIE_PSEAE_full_11-26-2021_b02.a2m'
    # MSA start: 1 - MSA end: 346
    wt_seq = 'MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNARKIAEMIVGMKQGLPGMDLVVFPEYSLQGIMYDPAEMMETAVAIPGEETEIFSRACRKANVWGVFSLTGERHEEHPRKAPYNTLVLIDNNGEIVQKYRKIIPWCPIEGWYPGGQTYVSEGPKGMKISLIICDDGNYPEIWRDCAMKGAELIVRCQGYMYPAKDQQVMMAKAMAWANNCYVAVANAAGFDGVYSYFGHSAIIGFDGRTLGECGEEEMGIQYAQLSLSQIRDARANDQSQNHLFKILHRGYSGLQASGDGDRGLAECPFEFYRTWVTDAEKARENVERLTRSTTGVAQCPVGRLPYEGLEKEA'
    variant_fitness_data = pd.read_csv(csv_path, sep=',')
    print('N_variant-fitness-tuples:', np.shape(variant_fitness_data)[0])
    #if np.shape(variant_fitness_data)[0] > 400000:
    #    print('More than 400000 variant-fitness pairs which represents a '
    #          'potential out-of-memory risk, skipping dataset...')
    #    continue
    variants = variant_fitness_data['mutant'].to_numpy()  # [200:500]
    fitnesses = variant_fitness_data['DMS_score'].to_numpy()  # [200:500]
    variants_split = []
    for variant in variants:
        # Split double and higher substituted variants to multiple single substitutions; 
        # e.g. separated by ':' or '/'
        variants_split.append(variant.split(':'))
    variants, fitnesses, sequences = get_seqs_from_var_name(
        wt_seq, variants_split, fitnesses, shift_pos=0)
    # Only model sequences with length of max. 800 amino acids to avoid out of memory errors 
    print('Sequence length:', len(wt_seq))

    gremlin = GREMLIN(alignment=msa_path, wt_seq=wt_seq, opt_iter=100, max_msa_seqs=10000)
    x_dca = gremlin.collect_encoded_sequences(sequences)
    x_wt = gremlin.x_wt
    y_pred = get_delta_e_statistical_model(x_dca, x_wt)
    print(f'\nDCA performance: {spearmanr(fitnesses, y_pred)[0]:.3f}  N={len(fitnesses)}\n')

    encoded_seqs_, attention_masks_ = get_encoded_seqs(sequences, tokenizer, max_length=len(wt_seq))
    xs, attns, scores = get_batches(encoded_seqs_, batch_size=BATCH_SIZE), get_batches(attention_masks_, batch_size=BATCH_SIZE), get_batches(fitnesses, batch_size=BATCH_SIZE)
    y_true, y_pred_esm1v = test(xs, attns, scores, loss_fn=corr_loss, model=model)
    y_true, y_pred_esm1v = y_true.cpu(), y_pred_esm1v.cpu()
    print(f'\nESM1v performance: {spearmanr(y_true, y_pred_esm1v)[0]:.3f} N={len(y_true)}\n')


    (
        x_train_dca, x_test_dca,
        x_train_esm1v, x_test_esm,
        attn_train_esm, attn_test_esm,
        y_train, y_test
    ) = train_test_split(
        x_dca,
        encoded_seqs_,
        attention_masks_,
        fitnesses,
        train_size=200,
        random_state=42
    )

    hm = DCAESMHybridModel(
        x_train_dca=np.array(x_train_dca), 
        x_train_esm=np.array(x_train_esm1v), 
        x_train_esm_attention_masks=np.array(attn_train_esm), 
        y_train=y_train,
        esm_model=model,
        x_wt=x_wt
    )

    y_test = reduce_by_batch_modulo(y_test)

    y_test_pred = hm.hybrid_prediction(x_dca=np.array(x_test_dca), x_esm=np.array(x_test_esm), attns_esm=np.array(attn_test_esm))
    print(f'Hybrid DCA model performance (N Train: {len(y_train)}, N Test: {len(y_test)}). '
          f'Spearman\'s rho: {abs(spearmanr(y_test, y_test_pred)[0]):.3f}')
