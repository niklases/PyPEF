# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Contains Python code used for the approach presented in our 'hybrid modeling' paper
# Preprint available at: https://doi.org/10.1101/2022.06.07.495081
# Code available at: https://github.com/Protein-Engineering-Framework/Hybrid_Model

from __future__ import annotations

import logging

import gpytorch

from pypef.gaussian_process.gauss_opt import get_gp_kernel_model
logger = logging.getLogger('pypef.hybrid.hybrid_model')

import os
import pickle
from os import listdir
from os.path import isfile, join
from typing import Union
import warnings
import gc
import torch

import numpy as np
import sklearn.base
from scipy.stats import spearmanr
#from sklearnex import patch_sklearn
#patch_sklearn(verbose=False)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.optimize import differential_evolution

from pypef.settings import USE_RAY
from pypef.utils.variant_data import get_sequences_from_file, remove_nan_encoded_positions
import pypef.dca.plmc_encoding
from pypef.dca.plmc_encoding import PLMC, get_dca_data_parallel, get_encoded_sequence
from pypef.utils.to_file import predictions_out
from pypef.utils.helpers import get_device
from pypef.utils.plot import plot_y_true_vs_y_pred
import pypef.dca.gremlin_inference
from pypef.plm.utils import hybrid_corr_mse_loss
from pypef.dca.gremlin_inference import GREMLIN, get_delta_e_statistical_model
from pypef.plm.esm_lora_tune import get_esm_models
from pypef.plm.prosst_lora_tune import get_prosst_models
from pypef.plm.inference import esm_setup, prosst_setup, tokenize_sequences, plm_inference

# sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and 
# will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function 
# becomes public and is part of the scikit-learn developer API.
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')


def reduce_by_batch_modulo(a: np.ndarray, batch_size=5) -> np.ndarray:
    """
    Cuts input array by batch size modulo.
    """
    reduce = len(a) - (len(a) % batch_size)
    return a[:reduce]


# TODO: Add meta-learning model (e.g., learn2learn MAML learning option on PGym dataset)?
class DCALLMHybridModel:
    def __init__(
            self,
            x_train_dca: np.ndarray,
            y_train: np.ndarray,
            llm_model_input: dict | None = None,
            x_wt: np.ndarray | None = None,
            alphas: np.ndarray | None = None,
            parameter_range: list[tuple] | None = None,
            ensemble_func: str = 'torch',
            lora_train: bool = True,
            gauss_opt: bool = False,
            batch_size: int | None = None,
            n_epochs: int | None = None,
            device: str | None = None,
            seed: int | None = None,
            verbose: bool = True,
            progress_cb=None, 
            abort_cb=None
    ):
        if llm_model_input is not None:
            if not isinstance(llm_model_input, dict):
                raise RuntimeError("Model input must be in form of a dictionary.")
            
            # Get the list of provided models
            self.llm_keys = list(llm_model_input.keys())
            supported_models = {'esm1v', 'prosst'}
            
            # Check for unsupported models
            unsupported = set(self.llm_keys) - supported_models
            if unsupported:
                raise RuntimeError(f"LLM input models {unsupported} not supported. "
                                   f"Currently supported models are {supported_models}")

            logger.info(f"Using LLM(s) ({', '.join(self.llm_keys)}) next to DCA for hybrid modeling...")
            
            # Store the entire dictionary so we can loop through it later
            self.llm_data = llm_model_input
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1), (0, 1), (0, 1)] 
        else:
            logger.info("No LLM inputs were defined for hybrid modelling. "
                  "Using only DCA for hybrid modeling...")
            self.llm_keys = None
            self.llm_model_input = None
            self.llm_attention_mask = None
            if parameter_range is None:
                parameter_range = [(0, 1), (0, 1)]
        if alphas is None:
            alphas = np.logspace(-6, 6, 100)
        self.parameter_range = parameter_range
        self.ensemble_func = ensemble_func
        self.gauss_opt = gauss_opt
        self.lora_train = lora_train
        self.alphas = alphas
        self.x_train_dca = x_train_dca
        self.y_train = y_train
        self.x_wild_type = x_wt
        if device is None:
            device = get_device()
        self.device = device
        logger.info(f'Using device {device.upper()} for hybrid modeling...')
        self.seed = seed
        if batch_size is None:
            batch_size = 5
        self.batch_size = batch_size
        if n_epochs is None:
            n_epochs = 50
        self.n_epochs = n_epochs
        self.verbose = verbose
        (
            self.ridge_opt, 
            self.betas,
            self.y_dca_ttest,
            self.y_dca_ridge_ttest,
            self.y_llm_ttest,
            self.y_llm_lora_ttest,
            self.y_llm_ttrain,
            self.y_llm_lora_ttrain,
            self.y_gp_opt_ttest
        ) = None, None, None, None, None, None, None, None, None
        self.progress_cb = progress_cb
        self.abort_cb = abort_cb
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
        grid = GridSearchCV(
            Ridge(), 
            {'alpha': self.alphas, 'random_state': [self.seed]}, 
            cv=5
        )
        grid.fit(x_train, y_train)
        return Ridge(**grid.best_params_).fit(x_train, y_train)
    
    def optimize_ensemble_weights(
            self,
            y_true_np: np.ndarray, 
            *y_preds_np: np.ndarray | None, 
            method: str = "spearman-hybrid",
            alpha: float = 0.5
        ) -> np.ndarray:

            valid_indices = []
            valid_preds_list = []

            # Prepare and Flip Predictors
            for i, p in enumerate(y_preds_np):
                if p is not None:
                    # OPTIONAL: If a predictor is naturally anti-correlated, 
                    # one might want to flip it here: p = -p
                    p_std = (p - np.mean(p)) / (np.std(p) + 1e-8)
                    valid_preds_list.append(torch.from_numpy(p_std).float())
                    valid_indices.append(i)

            X = torch.stack(valid_preds_list, dim=-1)
            y_true_std = torch.from_numpy((y_true_np - np.mean(y_true_np)) / (np.std(y_true_np) + 1e-8)).float()

            # Leaf Tensor initialization
            betas = torch.ones(X.shape[1], requires_grad=True)
            with torch.no_grad():
                betas /= X.shape[1]

            optimizer = torch.optim.LBFGS([betas], lr=0.1, max_iter=20)

            def closure():
                optimizer.zero_grad()
                # SOFTPLUS: Forces weights to be positive, but allowed to be > 1
                weights = torch.nn.functional.softplus(betas)
                y_hat = X @ weights
                loss = hybrid_corr_mse_loss(y_true_std, y_hat, method=method, alpha=alpha)
                loss.backward()
                return loss

            for _ in range(25): # L-BFGS is efficient, 25 steps is usually plenty
                optimizer.step(closure)

            # Transform the raw betas into the EFFECTIVE weights
            with torch.no_grad():
                # You must apply softplus here too!
                effective_weights = torch.nn.functional.softplus(betas).numpy()

            # Map back to original predictor slots
            final_betas = np.zeros(len(y_preds_np))
            for idx, w in zip(valid_indices, effective_weights):
                final_betas[idx] = w

            # Internal validation check
            y_ensemble = np.zeros_like(y_true_np)
            for i, p in enumerate(y_preds_np):
                if p is not None:
                    p_scaled = (p - np.mean(p)) / (np.std(p) + 1e-8)
                    y_ensemble += final_betas[i] * p_scaled

            final_corr = self.spearmanr(y_true_np, y_ensemble)
            logger.info(f"Ensemble Opt. Spearman: {final_corr:.3f} | Ensemble "
                        f"weights ({len(final_betas)}): {final_betas}")
            self.betas = final_betas
            return final_betas
    
    def adjust_betas(self, y: np.ndarray, *predictions: np.ndarray | None) -> np.ndarray:
        """
        Find parameters that maximize the absolute Spearman rank
        correlation coefficient using differential evolution 
        (pos-hoc ensemble weighting).

        Parameters
        ----------
        y : np.ndarray
            Array of fitness values.
        *predictions : np.ndarray | None
            A variable number of prediction arrays to balance. 
            None values are safely filtered out.

        Returns
        -------
        np.ndarray
            Array of beta weights corresponding to each valid prediction input, 
            maximizing the absolute Spearman rank correlation coefficient.
        """
        valid_preds = [p for p in predictions if p is not None]
        if not valid_preds:
            raise ValueError("At least one valid prediction array must be provided.")

        num_preds = len(valid_preds)

        # Dynamically identify which predictors contain NaNs
        clean_indices = []
        clean_preds = []

        for i, p in enumerate(valid_preds):
            if np.any(np.isnan(p)):
                logger.warning(f"Predictor at index {i} contains NaNs. Weighting parameter with zero...")
            else:
                clean_indices.append(i)
                clean_preds.append(p)

        # If all predictors were full of NaNs, return an array of zeros
        if not clean_preds:
            return np.zeros(num_preds)

        # Stack clean predictions into a 2D matrix for fast vectorized multiplication
        # Shape: (n_samples, n_clean_predictors)
        X_clean = np.column_stack(clean_preds)

        # 4. Define the objective function using matrix multiplication (@)
        def loss(params):
            # This dynamically replaces params[0]*y0 + params[1]*y1 + ...
            y_pred = X_clean @ params
            return -np.abs(self.spearmanr(y, y_pred))

        # Handle bounds dynamically to match the number of clean parameters
        # If self.parameter_range is a list of bounds, map it to the clean indices.
        # Otherwise, duplicate the base tuple (e.g., (-1, 1)) for each parameter.
        if isinstance(self.parameter_range, list):
            bounds = [self.parameter_range[i] for i in clean_indices]
        else:
            bounds = [self.parameter_range] * len(clean_indices)

        # Run the optimizer
        try:
            minimizer = differential_evolution(
                loss, bounds=bounds, tol=1e-4, rng=self.seed
            )
        except TypeError:  # SciPy v. 1.15.0 change: `seed` -> `rng` keyword
            minimizer = differential_evolution(
                loss, bounds=bounds, tol=1e-4, seed=self.seed
            )

        # Reconstruct the final weights array, leaving NaN predictors as 0.0
        final_betas = np.zeros(num_preds)
        for clean_idx, opt_weight in zip(clean_indices, minimizer.x):
            final_betas[clean_idx] = opt_weight
        
        y_ensemble = np.zeros_like(y, dtype=np.float64)

        # Iterate through predictors and apply weights
        for i, p in enumerate(predictions):
            if p is not None:
                # Standardize the predictor to match the scale used during optimization
                # (Crucial if your loss function used MSE)
                p_mean = np.mean(p)
                p_std = np.std(p) + 1e-8
                p_scaled = (p - p_mean) / p_std

                # Add weighted contribution
                y_ensemble += final_betas[i] * p_scaled

        # Validate the final ensemble performance
        final_corr = self.spearmanr(y, y_ensemble)
        logger.info(f"Ensemble-optimized Spearman: {final_corr:.3f} (N_opt={len(y)}) "
                    f"| Weights: {final_betas} (N_predictors={len(predictions)})")
        self.betas = final_betas
        return final_betas

    def get_subsplits_train(self, train_size_fit: float = 0.66):
        logger.info("Getting subsplits for supervised (re-)training of models "
              "and for adjustment of hybrid component contribution "
              "weights (\"beta's\")..."
        )
        train_size_fit = int(train_size_fit * len(self.y_train))
        train_size_beta_adjustment = len(self.y_train) - train_size_fit
        logger.info(f"Splitting training data of size {len(self.y_train)} "
              f"into {train_size_fit} variants for model tuning and "
              f"{train_size_beta_adjustment} variants for hybrid model "
              f"beta adjustment...")
        #if len(self.parameter_range) >= 4:
        # Reduce sizes by batch modulo
        n_drop = train_size_fit % self.batch_size
        if n_drop > 0:
            train_size_fit = train_size_fit - n_drop
            train_size_beta_adjustment = len(self.y_train) - train_size_fit
            logger.info(
                  f"Shifting {n_drop} variants from training set to "
                  f"beta adjustment set to match batch requirements "
                  f"of batch size {self.batch_size} for LLM retraining "
                  f"resulting in {train_size_fit} variants for model "
                  f"tuning and {train_size_beta_adjustment} variants "
                  f"for hybrid model beta adjustment..."
            )
        arrays_to_split = [self.x_train_dca, self.y_train]

        if self.llm_keys is not None:
            for llm_name in self.llm_keys:
                arrays_to_split.append(self.llm_data[llm_name]['x_llm'])
            
        splits = train_test_split(
            *arrays_to_split, 
            train_size=train_size_fit,
            random_state=self.seed
        )
        
        self.x_dca_ttrain = splits[0]
        self.x_dca_ttest = splits[1]
        self.y_ttrain = splits[2]
        self.y_ttest = splits[3]
        
        if self.llm_keys is not None:
            current_idx = 4
            for llm_name in self.llm_keys:
                self.llm_data[llm_name]['x_llm_ttrain'] = splits[current_idx]
                self.llm_data[llm_name]['x_llm_ttest'] = splits[current_idx + 1]
                current_idx += 2
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
        #y_ttrain_min_cv = int(0.2 * len(y_ttrain))
        #if y_ttrain_min_cv < 5:
        #    return 1.0, 0.0, 1.0, 0.0, None

    def train_llm(self):
        # LoRA training on y_llm_ttrain --> Testing on y_llm_ttest 
        # Here, just getting the unsupervised scores and correlations on ttrain and ttest splits
        self.y_llm_preds = {}       # e.g., {'esm1v': array, 'prosst': array}
        self.y_llm_lora_preds = {} 
        self.embeddings = {}        # e.g., {'esm1v': emb, 'prosst': emb}
        self.all_llm_ttest_scores = []
        # Initialize dictionaries for embeddings
        self.embs_ttrain = {}
        self.embs_ttest = {}

        # Loop through whatever models were passed in __init__
        for llm_name in self.llm_keys:
            logger.info(f"Processing LLM {llm_name.upper()}...")
            # Extract this specific model's data
            current_llm = self.llm_data[llm_name]
            base_model = current_llm['llm_base_model']
            lora_model = current_llm['llm_model']
            inference_fn = current_llm['llm_inference_function']
            training_fn = current_llm['llm_train_function']
            loss_fn = current_llm['llm_loss_function']
            optimizer = current_llm['llm_optimizer']
            x_llm_ttrain = current_llm['x_llm_ttrain']
            x_llm_ttest = current_llm['x_llm_ttest']
            wt_input_ids = current_llm['wt_input_ids']
            attention_mask = current_llm['llm_attention_mask']
            wt_struct_ids = current_llm.get('wt_structure_input_ids')
            y_llm_ttest = inference_fn(
                tokenized_sequences=x_llm_ttest,
                model=base_model,  # Before training, LoRa and Base model (should) yield the same results
                wt_input_ids=wt_input_ids,
                attention_mask=attention_mask,
                device=self.device,
                wt_structure_input_ids=wt_struct_ids
            )
            y_llm_ttrain = inference_fn(
                tokenized_sequences=x_llm_ttrain,
                model=base_model,
                wt_input_ids=wt_input_ids,
                attention_mask=attention_mask,
                device=self.device,
                wt_structure_input_ids=wt_struct_ids
            )
            if self.gauss_opt is True:
                self.embs_ttest[llm_name] = inference_fn(
                    tokenized_sequences=x_llm_ttest,
                    model=base_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    extract_emb=True,
                    device=self.device,
                    wt_structure_input_ids=wt_struct_ids
                )
                self.embs_ttrain[llm_name] = inference_fn(
                    tokenized_sequences=x_llm_ttrain,
                    model=base_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    extract_emb=True,
                    device=self.device,
                    wt_structure_input_ids=wt_struct_ids
                )
            logger.info(
                f"{llm_name} unsupervised performance: "
                f"Train set = {spearmanr(self.y_ttrain, y_llm_ttrain.detach().cpu())[0]:.3f}"
                f" (N={len(self.y_ttrain)}), "
                f"Test set = {spearmanr(self.y_ttest, y_llm_ttest.detach().cpu())[0]:.3f}"
                f" (N={len(self.y_ttest)})"
            )
            self.y_llm_ttrain = y_llm_ttrain.detach().cpu().numpy()
            self.y_llm_ttest = y_llm_ttest.detach().cpu().numpy()
            self.all_llm_ttest_scores.append(self.y_llm_ttest)

            if self.lora_train:
                logger.info('Refining/training the model... gradient calculation adds a computational '
                      'graph that requires quite some memory - if you are facing an (out of memory) '
                      'error, try reducing the batch size or sticking to CPU device...')
                training_fn(
                    x_sequences=x_llm_ttrain, 
                    scores=self.y_ttrain,
                    loss_fn=loss_fn,
                    model=lora_model,
                    optimizer=optimizer, 
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    n_epochs=self.n_epochs,
                    device=self.device,
                    verbose=self.verbose,
                    raise_error_on_train_fail=False,
                    progress_cb=self.progress_cb, 
                    abort_cb=self.abort_cb,
                    wt_structure_input_ids=wt_struct_ids
                )
                y_llm_lora_ttrain = inference_fn(
                    tokenized_sequences=x_llm_ttrain,
                    model=lora_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    device=self.device,
                    verbose=self.verbose,
                    wt_structure_input_ids=wt_struct_ids
                )
                y_llm_lora_ttest = inference_fn(
                    tokenized_sequences=x_llm_ttest,
                    model=lora_model,
                    wt_input_ids=wt_input_ids,
                    attention_mask=attention_mask,
                    device=self.device,
                    verbose=self.verbose,
                    wt_structure_input_ids=wt_struct_ids
                )
                logger.info(
                    f"{llm_name.upper()} supervised tuned performance: "
                    f"Train = {spearmanr(self.y_ttrain, y_llm_lora_ttrain.detach().cpu())[0]:.3f} "
                    f"(N={len(self.y_ttrain)}), "
                    f"Test = {spearmanr(self.y_ttest, y_llm_lora_ttest.detach().cpu())[0]:.3f} "
                    f"(N={len(self.y_ttest)})"
                )
                self.y_llm_lora_ttrain = y_llm_lora_ttrain.detach().cpu().numpy()
                self.y_llm_lora_ttest = y_llm_lora_ttest.detach().cpu().numpy()
                self.all_llm_ttest_scores.append(self.y_llm_lora_ttest)
        
        if self.gauss_opt:
            emb_esm_ttrain = self.embs_ttrain.get('esm1v')
            emb_prosst_ttrain = self.embs_ttrain.get('prosst')
            
            emb_esm_ttest = self.embs_ttest.get('esm1v')
            emb_prosst_ttest = self.embs_ttest.get('prosst')

            # 4. Use your original, explicit multi-kernel setup
            if emb_esm_ttrain is not None and emb_prosst_ttrain is not None:
                self.gp_model = get_gp_kernel_model(
                    y_train=self.y_ttrain, 
                    x_tokseqs_seq_kernel_train=emb_esm_ttrain, 
                    x_tokseqs_struct_kernel_train=emb_prosst_ttrain, 
                    device=self.device, train=True
                )
                emb_ttrain = torch.cat([emb_esm_ttrain, emb_prosst_ttrain], dim=-1)
                emb_ttest = torch.cat([emb_esm_ttest, emb_prosst_ttest], dim=-1)
            elif emb_esm_ttrain is not None:
                self.gp_model = get_gp_kernel_model(
                    y_train=self.y_ttrain, 
                    x_tokseqs_seq_kernel_train=emb_esm_ttrain, 
                    device=self.device, train=True
                )
                emb_ttrain = emb_esm_ttrain
                emb_ttest = emb_esm_ttest
            elif emb_prosst_ttrain is not None:
                self.gp_model = get_gp_kernel_model(
                    y_train=self.y_ttrain, 
                    x_tokseqs_seq_kernel_train=emb_prosst_ttrain, 
                    device=self.device, train=True
                )
                emb_ttrain = emb_prosst_ttrain
                emb_ttest = emb_prosst_ttest
            else:
                raise RuntimeError("No valid embeddings found for GP optimization.")
                
            likelihood = self.gp_model.likelihood
            self.gp_model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp_pred_ttrain = likelihood(self.gp_model(emb_ttrain)).mean.detach().cpu().numpy()  # site-packages\gpytorch\models\exact_gp.py:299: GPInputWarning: The input matches the stored training data. Did you forget to call model.train()?
                gp_pred_ttest = likelihood(self.gp_model(emb_ttest))
                self.y_gp_opt_ttest = gp_pred_ttest.mean.detach().cpu().numpy()
                logger.info(
                    f"{llm_name.upper()} supervised Gaussian process optimized performance: "
                    f"Train = {spearmanr(self.y_ttrain, gp_pred_ttrain)[0]:.3f} "
                    f"(N={len(self.y_ttrain)}), "
                    f"Test = {spearmanr(self.y_ttest, self.y_gp_opt_ttest)[0]:.3f} "
                    f"(N={len(self.y_ttest)})"
                )
            

    def train_and_optimize(self) -> tuple:
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
        self.get_subsplits_train()
        self.y_dca_ttest = self._delta_e(self.x_dca_ttest)
        self.ridge_opt = self.ridge_predictor(self.x_dca_ttrain, self.y_ttrain)
        self.y_dca_ridge_ttest = self.ridge_opt.predict(self.x_dca_ttest)

        predictors = [self.y_dca_ttest, self.y_dca_ridge_ttest]

        if len(self.parameter_range) >= 4:
            self.train_llm()
            # Add LLM predictors to the list
            predictors.extend(self.all_llm_ttest_scores)
            if self.gauss_opt:
                predictors.append(self.y_gp_opt_ttest)
    
        if self.ensemble_func == 'torch':
            self.all_betas = self.optimize_ensemble_weights(self.y_ttest, *predictors)
        else:
            self.all_betas = self.adjust_betas(self.y_ttest, *predictors)
        return (*self.all_betas, self.ridge_opt)

    def hybrid_prediction(
            self,
            x_dca: np.ndarray,
            x_llm_dict: dict | None = None,
            verbose: bool = True
    ) -> np.ndarray:
        logger.info(f"Hybrid prediction with N_individual model weights ('betas') = {self.betas}...")
        y_dca = self._delta_e(x_dca)
        y_ridge = self.ridge_opt.predict(x_dca) if self.ridge_opt is not None else np.zeros(len(y_dca))

        predictors = [y_dca, y_ridge]
        llm_embs_ttest = {}

        if self.llm_keys is not None:
            for llm_name in self.llm_keys:
                current_llm = self.llm_data[llm_name]
                x_input = x_llm_dict.get(llm_name) if x_llm_dict else None

                # Must append something to keep beta indices correct
                if x_input is None:
                    predictors.extend([np.zeros(len(y_dca)), np.zeros(len(y_dca))])
                    continue
                
                common_args = {
                    'tokenized_sequences': x_input,
                    'wt_input_ids': current_llm['wt_input_ids'],
                    'attention_mask': current_llm['llm_attention_mask'],
                    'device': self.device,
                    'verbose': verbose,
                    'wt_structure_input_ids': current_llm.get('wt_structure_input_ids')
                }

                y_base = current_llm['llm_inference_function'](model=current_llm['llm_base_model'], **common_args)
                predictors.append(y_base.detach().cpu().numpy())
                if self.lora_train:
                    y_lora = current_llm['llm_inference_function'](model=current_llm['llm_model'], **common_args)
                    predictors.append(y_lora.detach().cpu().numpy())

                if self.gauss_opt:
                    emb_args = {**common_args, 'model': current_llm['llm_base_model'], 'extract_emb': True}
                    llm_embs_ttest[llm_name] = current_llm['llm_inference_function'](**emb_args)

        if self.gauss_opt:
            self.gp_model.eval()
            esm_emb = llm_embs_ttest.get('esm1v')
            prosst_emb = llm_embs_ttest.get('prosst')

            if esm_emb is not None and prosst_emb is not None:
                gp_input = torch.cat([esm_emb, prosst_emb], dim=-1)
            elif esm_emb is not None:
                gp_input = esm_emb
            else:
                gp_input = prosst_emb

            y_gp_list = []
            predict_batch_size = 100  # Adjust based on VRAM, 100 is very safe
            
            gp_input_batches = torch.split(gp_input, predict_batch_size)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for batch in gp_input_batches:
                    batch = batch.to(self.device)
                    batch_output = self.gp_model.likelihood(self.gp_model(batch))
                    y_gp_list.append(batch_output.mean.cpu().numpy())
            
            y_gp = np.concatenate(y_gp_list)
            predictors.append(y_gp)

        y_final = np.zeros_like(y_dca)
        for beta, p in zip(self.all_betas, predictors, strict=True):
            std_val = np.std(p)
            p_std = (p - np.mean(p)) / (std_val + 1e-8) if std_val > 1e-8 else p
            y_final += beta * p_std
        return y_final

    def ls_ts_performance(self):
        beta_1, beta_2, reg = self.settings(
            x_train=self.x_train,
            y_train=self.y_traing
        )
        spearman_r = self.spearmanr(
            self.y_test,
            self.hybrid_prediction(self.x_test, reg, beta_1, beta_2)
        )
        self.beta_1, self.beta_2, self.regressor = beta_1, beta_2, reg
        return spearman_r, reg, beta_1, beta_2


""" 
###########################################################################################
# Below: Some helper functions that call or are dependent on the DCALLMHybridModel class. #
###########################################################################################
""" 


def check_model_type(model: dict | DCALLMHybridModel | PLMC | GREMLIN):
    """
    Checks type/instance of model.
    """
    if type(model) == dict:
        try:
            model = model['model']
        except KeyError:
            raise RuntimeError("Unknown model dictionary taken from Pickle file.")
    if type(model) == pypef.dca.plmc_encoding.PLMC:
        return 'PLMC'
    elif type(model) == pypef.hybrid.hybrid_model.DCALLMHybridModel:
        return 'Hybrid'
    elif type(model) == pypef.dca.gremlin_inference.GREMLIN:
        return 'GREMLIN'
    elif isinstance(model, sklearn.base.BaseEstimator):
        raise RuntimeError("Loaded an sklearn ML model. For pure ML-based modeling the "
                          "\'ml\' flag has to be used instead of the \'hybrid\' flag.")
    else:
        raise RuntimeError('Unknown model/unknown Pickle file.')


def get_model_path(model: str):
    """
    Checks if model Pickle files exits in CWD 
    and then in ./Pickles directory.
    """
    # Not capitalizing model names here as PLMC params file names are not capitalized
    # model = os.path.splitext(model)[0].upper() + os.path.splitext(model)[1]
    try:
        if isfile(model):
            model_path = model
        elif isfile(f'Pickles/{model}'):
            model_path = f'Pickles/{model}'
        else:
            raise RuntimeError(
                f"Did not find specified model file ({model}) in current "
                "working directory or /Pickles subdirectory. Make sure "
                "to train/save a model first (e.g., for saving a GREMLIN "
                "model, type \"pypef param_inference --msa TARGET_MSA.a2m\" "
                "or, for saving a plmc model, type \"pypef param_inference "
                "--params TARGET_PLMC.params\")."
            )
        return model_path
    except TypeError:
        raise RuntimeError(
            "No provided model. Specify a " \
            "model for DCA-based encoding."
        )


def get_model_and_type(
        params_file: str, 
        substitution_sep: str = '/'
):
    """
    Tries to load/unpickle model to identify the model type 
    and to load the model from the identified plmc pickle file 
    or from the loaded pickle dictionary.
    """
    file_path = get_model_path(params_file)
    logger.info(f"Unpickling file {os.path.abspath(file_path)}...")
    if type(params_file) == pypef.dca.gremlin_inference.GREMLIN:
        logger.info("Found GREMLIN model...")
        return params_file, 'GREMLIN'
    if type(params_file) == pypef.dca.plmc_encoding.PLMC:
        logger.info("Found PLMC model...")
        return params_file, 'PLMC'
    try:
        with open(file_path, 'rb') as read_pkl_file:
            model = pickle.load(read_pkl_file)
            model_type = check_model_type(model)
    except pickle.UnpicklingError:
        logger.info("Unpickling error... assuming PLMC parameters found...")
        model_type = 'PLMC_Params'

    if model_type == 'PLMC_Params':
        model = PLMC(
            params_file=params_file,
            separator=substitution_sep,
            verbose=False
        )
        model_type = 'PLMC'

    else:  # --> elif model_type in ['PLMC', 'GREMLIN', 'Hybrid']:
        model = model['model']
    if model_type == 'Hybrid':
        if model.llm_key == 'esm1v':
            logger.info("Found hybrid model with ESM1v LLM model...")
            base_model, lora_model, _tokenizer, _optimizer = get_esm_models()
            model_type += '_ESM1v'
        elif model.llm_key == 'prosst':
            logger.info("Found hybrid model with ProSST LLM model...")
            base_model, lora_model, _tokenizer, _optimizer = get_prosst_models()
            model_type += '_ProSST'
        else:
            logger.info("Found hybrid model without LLM model...")
            return model, model_type
        base_model.load_state_dict(model.llm_base_model)
        lora_model.load_state_dict(model.llm_model)
        model.llm_model = lora_model
        model.llm_base_model = base_model
        model.llm_model.eval()
        model.llm_base_model.eval()

    return model, model_type


def save_model_to_dict_pickle(
        model: DCALLMHybridModel | PLMC | GREMLIN,
        model_type: str | None = None
):
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass

    if model_type is None:
        model_type = 'MODEL'
    
    # For Hybrid LLM models save as model.state_dict()
    if model_type.lower().startswith('hybrid'):
        if model.llm_key is not None:
            logger.info(f"Storing LLM model {model.llm_key.upper()} "
                  f"of hybrid model as state dictionaries...")
            model.llm_model = model.llm_model.to('cpu')
            model.llm_model = model.llm_model.state_dict()
            model.llm_base_model = model.llm_base_model.to('cpu')
            model.llm_base_model = model.llm_base_model.state_dict()
            model.llm_model_input[model.llm_key]['llm_base_model'] = None
            model.llm_model_input[model.llm_key]['llm_model'] = None
            model.progress_cb = None
            model.abort_cb = None
            model_type += model.llm_key.upper()
    pkl_path = os.path.abspath(f'Pickles/{model_type.upper()}')
    pickle.dump(
        {
            'model': model,
            'model_type': model_type
        },
        open(pkl_path, 'wb')
    )
    logger.info(f'Saved model as Pickle file ({pkl_path})...')
    # Free up memory (needed when running from Qt GUI threads?) 
    del model
    torch.cuda.empty_cache()
    gc.collect()


global_model = None
global_model_type = None


def plmc_or_gremlin_encoding(
        variants,
        sequences,
        ys_true,
        params_file,
        substitution_sep='/',
        threads=1,
        verbose=True,
        use_global_model=False
):
    """
    Decides based on the params file input type which DCA encoding 
    to be performed, i.e., GREMLIN or PLMC.
    If use_global_model==True, to avoid each time pickle model 
    file getting loaded, which is quite inefficient when performing 
    directed evolution, i.e., encoding of single sequences, a 
    global model is stored at the first evolution step and used 
    in the subsequent steps.
    """
    global global_model, global_model_type
    if ys_true is None:
        ys_true = np.zeros(np.shape(sequences))
    if use_global_model:
        if global_model is None:
            global_model, global_model_type = get_model_and_type(
                params_file, substitution_sep)
            model, model_type = global_model, global_model_type
        else:
            model, model_type = global_model, global_model_type
    else:
        model, model_type = get_model_and_type(
            params_file, substitution_sep)
    if model_type == 'PLMC':
        xs, x_wt, variants, sequences, ys_true = plmc_encoding(
            model, variants, sequences, ys_true, threads, verbose
        )
    elif model_type == 'GREMLIN':
        if verbose:
            logger.info(
                f"Following positions are frequent gap positions "
                f"in the MSA and cannot be considered for effective "
                f"modeling, i.e., substitutions at these positions "
                f"are removed as these would be predicted with "
                f"wild-type fitness:"
                f"\n{[int(gap) + 1 for gap in model.gaps]}.\n"
                f"Effective positions (N={len(model.v_idx)}) are:\n"
                f"{[int(v_pos) + 1 for v_pos in model.v_idx]}"
            )
        xs, x_wt, variants, sequences, ys_true = gremlin_encoding(
            model, variants, sequences, ys_true,
            shift_pos=1, substitution_sep=substitution_sep
        )
    else:
        raise RuntimeError(
            f"Found a {model_type.lower()} model as input. Please "
            f"train a new hybrid model on the provided LS/TS datasets."
        )
    assert len(xs) == len(variants) == len(sequences) == len(ys_true)
    return xs, variants, sequences, ys_true, x_wt, model, model_type


def gremlin_encoding(gremlin: GREMLIN, variants, sequences, ys_true, 
                     shift_pos=1, substitution_sep='/'):
    """
    Gets X and x_wt for DCA prediction: delta_Hamiltonian respectively
    delta_E = np.subtract(X, x_wt), with X = encoded sequences of variants.
    Also removes variants, sequences, and y_trues at MSA gap positions.
    """
    variants, sequences, ys_true = (
        np.atleast_1d(variants), 
        np.atleast_1d(sequences), 
        np.atleast_1d(ys_true)
    )
    variants, sequences, ys_true = remove_gap_pos(
        gremlin.gaps, variants, sequences, ys_true,
        shift_pos=shift_pos, substitution_sep=substitution_sep
    )
    if not sequences:
        xs = []
    else:
        try:
            xs = gremlin.get_scores(sequences, encode=True)
        except RuntimeError:
            xs = []
    x_wt = gremlin.get_scores(np.atleast_1d(gremlin.wt_seq), encode=True)
    return xs, x_wt, variants, sequences, ys_true


def plmc_encoding(plmc: PLMC, variants, sequences, ys_true, threads=1, verbose=False):
    """
    Gets X and x_wt for DCA prediction: delta_E = np.subtract(X, x_wt),
    with X = encoded sequences of variants.
    Also removes variants, sequences, and y_trues at MSA gap positions.
    """
    target_seq, index = plmc.get_target_seq_and_index()
    wt_name = target_seq[0] + str(index[0]) + target_seq[0]
    if verbose:
        logger.info(f"Using to-self-substitution '{wt_name}' as wild type reference. "
              f"Encoding variant sequences. This might take some time...")
    x_wt = get_encoded_sequence(wt_name, plmc)
    if threads > 1 and USE_RAY:
        # Hyperthreading, NaNs are already being removed by the called function
        variants, sequences, xs, ys_true = get_dca_data_parallel(
            variants, sequences, ys_true, plmc, threads, verbose=verbose)
    else:
        x_ = plmc.collect_encoded_sequences(variants)
        # NaNs must still be removed
        xs, variants, sequences, ys_true = remove_nan_encoded_positions(
            x_, variants, sequences, ys_true
        )
    return xs, x_wt, variants, sequences, ys_true


def remove_gap_pos(
        gaps,
        variants,
        sequences,
        fitnesses,
        shift_pos=1,
        substitution_sep='/'
):
    """
    Remove gap postions from input variants, sequences, and fitness values
    based on input gaps (gap positions).
    Note that by default, gap positions are shifted by +1 to match the input
    variant identifiers (e.g., variant A123C is removed if gap pos is 122; (122 += 1).

    Returns
    -----------
    variants_v
        Variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    sequences_v
        Sequences of variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    fitnesses_v
        Fitness values of variants with substitutions at valid sequence positions, 
        i.e., at non-gap positions
    """
    variants_v, sequences_v, fitnesses_v = [], [], []
    valid = []
    for i, variant in enumerate(variants):
        variant = variant.split(substitution_sep)
        for var in variant:
            if int(var[1:-1]) not in [gap + shift_pos for gap in gaps]:
                if i not in valid:
                    valid.append(i)
                    variants_v.append(variants[i])
                    sequences_v.append(sequences[i])
                    fitnesses_v.append(fitnesses[i])
    return variants_v, sequences_v, fitnesses_v


def performance_ls_ts(
        ls_fasta: str | None,
        ts_fasta: str | None,
        threads: int,
        params_file: str,
        model_pickle_file: str | None = None,
        llm: str | None = None,
        pdb_file: str | None = None,
        wt_seq: str | None = None,
        substitution_sep: str = '/',
        label=False,
        device: str| None = None,
        progress_cb=None, 
        abort_cb=None
):
    test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)

    if ls_fasta is not None and ts_fasta is not None:
        train_sequences, train_variants, y_train = get_sequences_from_file(
            ls_fasta)
        (
            x_train, train_variants, train_sequences, 
            y_train, x_wt, _, model_type 
        ) = plmc_or_gremlin_encoding(
            train_variants, train_sequences, y_train, 
            params_file, substitution_sep, threads
        )

        (
            x_test, test_variants, test_sequences, y_test, *_
        ) = plmc_or_gremlin_encoding(
            test_variants, test_sequences, y_test, params_file, 
            substitution_sep, threads, verbose=False
        )

        logger.info(f"Initial training set variants: {len(train_sequences)}. "
                    f"Remaining: {len(train_variants)} (after removing "
                    f"substitutions at gap positions).\nInitial test set "
                    f"variants: {len(test_sequences)}. Remaining: " 
                    f"{len(test_variants)} (after removing substitutions "
                    f"at gap positions)."
        )
        if llm is not None:
            if llm.lower().startswith('esm'):
                llm_dict = esm_setup(train_sequences)
                x_llm_test = tokenize_sequences(test_sequences, llm_dict['esm1v']['llm_tokenizer'])
            elif llm.lower() == 'prosst':
                llm_dict = prosst_setup(
                    wt_seq, pdb_file, sequences=train_sequences)
                x_llm_test = tokenize_sequences(test_sequences, llm_dict['prosst']['llm_tokenizer'])
        else:
            llm_dict = None
            x_llm_test = None
            llm = ''
        hybrid_model = DCALLMHybridModel(
            x_train_dca=np.array(x_train),
            y_train=np.array(y_train),
            llm_model_input=llm_dict,
            x_wt=x_wt,
            device=device,
            progress_cb=progress_cb, 
            abort_cb=abort_cb
        )
        y_test_pred = hybrid_model.hybrid_prediction(np.array(x_test), x_llm_test)
        logger.info(f'Hybrid performance: {spearmanr(y_test, y_test_pred)[0]:.3f} N={len(y_test)}')
        save_model_to_dict_pickle(hybrid_model, f'HYBRID{model_type}')

    elif (
        ts_fasta is not None and 
        model_pickle_file is not None 
        and params_file is not None
    ):
        # no LS provided but hybrid model provided for 
        # individual beta contributed zero shot predictions
        logger.info(f'Taking model from saved model (Pickle file): {model_pickle_file}...')
        model, model_type = get_model_and_type(model_pickle_file)
        if not model_type.startswith('Hybrid'):  # same as below in next elif
            (
                x_test, test_variants, test_sequences, 
                y_test, x_wt, *_
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, model_pickle_file, 
                substitution_sep, threads, False
            )
            y_test_pred = get_delta_e_statistical_model(x_test, x_wt)
        else:  # Hybrid model input requires params from plmc or GREMLIN model
            (
                x_test, test_variants, test_sequences, 
                y_test, *_
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, params_file,
                substitution_sep, threads, False
            )
            if model.llm_model_input is not None:
                llm_ = list(model.llm_model_input.keys())[0]
                tokenizer = model.llm_model_input[llm_]['llm_tokenizer']
                logger.info(f"Found hybrid model with LLM {llm_}...")
                x_llm_test = tokenize_sequences(test_sequences, tokenizer)
                y_test_pred = model.hybrid_prediction(x_test, x_llm_test)
            else:
                y_test_pred = model.hybrid_prediction(x_test)
    
    elif ts_fasta is not None and model_pickle_file is None:
        # no LS and *no hybrid model* provided:
        # statistical modeling / no ML / zero-shot LLM predictions
        logger.info(
            f"No learning set provided, falling back to statistical DCA model: "
            f"no adjustments of individual hybrid model parameters (\"beta's\")."
        )
        test_sequences, test_variants, y_test = get_sequences_from_file(ts_fasta)
        logger.info(
            f"Initial test set variants: {len(test_sequences)}. "
            f"Remaining: {len(test_variants)} (after removing "
            f"substitutions at gap positions)."
        )
        if params_file is not None:
            logger.info("DCA inference on test set...")
            (
                x_test, test_variants, test_sequences, 
                y_test, x_wt, model, model_type
            ) = plmc_or_gremlin_encoding(
                test_variants, test_sequences, y_test, 
                params_file, substitution_sep, threads
            )
            y_test_pred = get_delta_e_statistical_model(x_test, x_wt)
            save_model_to_dict_pickle(model, model_type)
            model_type = f'{model_type}_no_ML'
        else:
            model_type = 'LLM'
            if llm == 'esm':
                llm_dict = esm_setup(test_sequences[0], test_sequences)  # TODO: Improve wt_seq input workaround
                logger.info("Zero-shot LLM inference on test set using ESM1v...")
                y_test_pred = plm_inference(
                    tokenized_sequences = llm_dict['esm1v']['x_llm'],
                    wt_input_ids=llm_dict['esm1v']['wt_input_ids'],
                    model=llm_dict['esm1v']['llm_base_model']
                )
            elif llm == 'prosst':
                llm_dict = prosst_setup(test_sequences[0], test_sequences)  # TODO: Improve wt_seq input workaround
                logger.info("Zero-shot LLM inference on test set using ProSST...")
                y_test_pred = plm_inference(
                    tokenized_sequences = llm_dict['prosst']['x_llm'],
                    wt_input_ids=llm_dict['prosst']['wt_input_ids'],
                    model=llm_dict['prosst']['llm_base_model'],
                    wt_structure_input_ids=llm_dict['prosst']['wt_structure_input_ids']
                    
                )
            else:
                raise RuntimeError("Unknown --llm flag option.")
    else:
        raise RuntimeError('No test set given for performance estimation.')
    if llm is None or llm == '':
        llm = ''
    else:
        llm = f"_{llm.upper()}"
    plot_y_true_vs_y_pred(
        np.array(y_test), np.array(y_test_pred), np.array(test_variants), 
        label=label, hybrid=True, name=f'{model_type}{llm}'
    )


def predict_ps(
        prediction_dict: dict,
        threads: int,
        separator: str,
        model_pickle_file: str | None = None,
        params_file: str | None = None,
        prediction_set: str | None = None,
        llm: str | None = None,
        pdb_file: str | None = None,
        wt_seq: str | None = None,
        negative: bool = False
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
    prediction_dict: dict
        Contains arguments which directory to predict, e.g. {'drecomb': True},
        than predicts prediction files that are present in this directory, e.g.
        in directory './Recomb_Double_Split'.
    params_file: str
        PLMC/GREMLIN couplings parameter file
    threads: int
        Threads used for parallelization for DCA-based sequence encoding
    separator: str
        Separator of individual substitution of variants, default '/'
    model_pickle_file: str
        Pickle file containing the hybrid model and model parameters in
        a dictionary format
    test_set: str = None
        Test set for prediction and plotting of predictions (contains
        true fitness values of variants).
    prediction_set: str = None
        Prediction set for prediction, does not contain true fitness values.
    figure: str = None
        Plotting the test set predictions and the corresponding true fitness
        values.
    label: bool = False
        If True, plots associated variant names of predicted variants.
    negative: bool = False
        If true, negative defines improved variants having a reduced/negative
        fitness compared to wild type.


    Returns
    -----------
    ()
        Writes sorted predictions to files (for [--drecomb] [--trecomb]
        [--qarecomb] [--qirecomb] [--ddiverse] [--tdiverse] [--qdiverse]
        in the respective created folders).

    """
    dca_modeling = False
    if model_pickle_file is None and params_file is not None:
        model_pickle_file = params_file
        logger.info(f'Trying to load model from saved parameters (Pickle file): {model_pickle_file}...')
        dca_modeling = True
    elif params_file is not None:
        logger.info(f'Loading model from saved model (Pickle file {model_pickle_file})...')
        dca_modeling = True
    if dca_modeling:
        model, model_type = get_model_and_type(model_pickle_file)
        if model_type == 'PLMC' or model_type == 'GREMLIN':
            logger.info(f'Found {model_type} model file. No hybrid model provided - '
                        f'falling back to a statistical DCA model...')

    pmult = [
        'Recomb_Double_Split', 'Recomb_Triple_Split', 'Recomb_Quadruple_Split',
        'Recomb_Quintuple_Split', 'Diverse_Double_Split', 'Diverse_Triple_Split',
        'Diverse_Quadruple_Split'
    ]
    if True in prediction_dict.values():
        for ps, path in zip(prediction_dict.values(), pmult):
            if ps:  # if True, run prediction in this directory, e.g. for drecomb
                logger.info(f'Running predictions for variant-sequence files in directory {path}...')
                all_y_v_pred = []
                files = [f for f in listdir(path) if isfile(join(path, f)) if f.endswith('.fasta')]
                for i, file in enumerate(files):  # collect and predict for each file in the directory
                    logger.info(f'Encoding files ({i + 1}/{len(files)}) for prediction...')
                    file_path = os.path.join(path, file)
                    sequences, variants, _ = get_sequences_from_file(file_path)
                    if not model_type.startswith('Hybrid'):
                        x_test, _, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, model, threads=threads, verbose=False,
                            substitution_sep=separator)
                        ys_pred = get_delta_e_statistical_model(x_test, x_wt)
                    else:  # Hybrid model input requires params from plmc or GREMLIN model plus optional LLM input
                        x_test, _test_variants, test_sequences, *_ = plmc_or_gremlin_encoding(
                            variants, sequences, None, params_file,
                            threads=threads, verbose=False, substitution_sep=separator
                        )
                        if model.llm_key is None:  # TODO: Check llm_key
                            ys_pred = model.hybrid_prediction(x_test)
                        else:
                            sequences = [str(seq) for seq in test_sequences]
                            llm_ = list(model.llm_model_input.keys())[0]
                            tokenizer = model.llm_model_input[llm_]['llm_tokenizer']
                            x_llm_test = tokenize_sequences(sequences, tokenizer)
                            ys_pred = model.hybrid_prediction(np.asarray(x_test), np.asarray(x_llm_test))
                    for k, y in enumerate(ys_pred):
                        all_y_v_pred.append((ys_pred[k], variants[k]))
                if negative:  # sort by fitness value
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=False)
                else:
                    all_y_v_pred = sorted(all_y_v_pred, key=lambda x: x[0], reverse=True)
                predictions_out(
                    predictions=all_y_v_pred,
                    model=model_type,
                    prediction_set=f'Top{path}',
                    path=path
                )
            else:  # check next task to do, e.g., predicting triple substituted variants, e.g. trecomb
                continue

    elif prediction_set is not None:  # Predicting single FASTA file sequences
        sequences, variants, _ = get_sequences_from_file(prediction_set)
        # NaNs are already being removed by the called function
        if not dca_modeling:  # model_pickle_file is None and params_file is None:
            # *No hybrid model* and no DCA params provided:
            # Zero-shot LLM predictions
            if llm == 'esm':
                model_type = 'LLM_ESM1v'
                logger.info("Zero-shot LLM inference on test set using ESM1v...")
                ys_pred = plm_inference(sequences, llm)  # TODO
            elif llm == 'prosst':
                model_type = 'LLM_ProSST'
                logger.info("Zero-shot LLM inference on test set using ProSST...")
                ys_pred = plm_inference(sequences, llm, pdb_file=pdb_file, wt_seq=wt_seq)  # TODO
        else:
            if not model_type.startswith('Hybrid'):  # statistical DCA model
                xs, variants, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=False, substitution_sep=separator
                )
                ys_pred = get_delta_e_statistical_model(xs, x_wt)
            else:  # Hybrid model input requires params from plmc or GREMLIN model plus optional LLM input
                xs, variants, sequences, *_ = plmc_or_gremlin_encoding(
                    variants, sequences, None, params_file,
                    threads=threads, verbose=True, substitution_sep=separator
                )
                if model.llm_key is None:
                    ys_pred = model.hybrid_prediction(xs)
                else:
                    sequences = [str(seq) for seq in sequences]
                    llm_ = list(model.llm_model_input.keys())[0]
                    tokenizer = model.llm_model_input[llm_]['llm_tokenizer']
                    xs_llm = tokenize_sequences(sequences, tokenizer)
                    ys_pred = model.hybrid_prediction(np.asarray(xs), np.asarray(xs_llm))
            assert len(xs) == len(variants) == len(ys_pred)
        y_v_pred = zip(ys_pred, variants)
        y_v_pred = sorted(y_v_pred, key=lambda x: x[0], reverse=True)
        predictions_out(
            predictions=y_v_pred,
            model=model_type,
            prediction_set=f'Top{prediction_set}'
        )


global_hybrid_model = None
global_hybrid_model_type = None


def predict_directed_evolution(
        encoder: str,
        variant: str,
        variant_sequence: str,
        hybrid_model_data_pkl: None | str
) -> Union[str, list]:
    """
    Perform directed in silico evolution and predict the fitness of a
    (randomly) selected variant using the hybrid model. This function opens
    the stored DCALLMHybridModel and the model parameters to predict the fitness
    of the variant encoded herein using the PLMC class. If the variant
    cannot be encoded (based on the PLMC params file), returns 'skip'. Else,
    returning the predicted fitness value and the variant name.
    """
    global global_hybrid_model, global_hybrid_model_type
    if hybrid_model_data_pkl is not None:
        if global_hybrid_model is None:
            global_hybrid_model, global_hybrid_model_type = get_model_and_type(
                hybrid_model_data_pkl)
            model, model_type = global_hybrid_model, global_hybrid_model_type
        else:
            model, model_type = global_hybrid_model, global_hybrid_model_type
    else:
        model_type = 'StatisticalModel'  # any name != 'Hybrid'

    if not model_type.startswith('Hybrid'):  # statistical DCA model
        xs, variant, _, _, x_wt, *_ = plmc_or_gremlin_encoding(
            variant, variant_sequence, None, encoder, 
            verbose=False, use_global_model=True)
        if not list(xs):
            return 'skip'
        y_pred = get_delta_e_statistical_model(xs, x_wt)
    else:  # model_type == 'Hybrid': Hybrid model input requires params 
        # from PLMC or GREMLIN model plus optional LLM input
        xs, variant, variant_sequence, *_ = plmc_or_gremlin_encoding(
            variant, variant_sequence, None, encoder, 
            verbose=False, use_global_model=True
        )
        if not list(xs):
            return 'skip'
        try:
            if model.llm_model_input is None:
                y_pred = model.hybrid_prediction(xs)
            else:
                x_llm = tokenize_sequences(model.llm_model_input, 
                                     variant_sequence, verbose=False)

                y_pred = model.hybrid_prediction(
                    np.atleast_2d(xs), 
                    np.atleast_2d(x_llm), verbose=False
                )[0]
        except ValueError as e:
            raise RuntimeError(
                f"Error: {e}\nProbably a different model was used for encoding than "
                "for modeling; e.g. using a HYBRIDgremlin model in "
                "combination with parameters taken from a PLMC file."
            )
    y_pred = float(y_pred)

    return [(y_pred, variant[0][1:])]
