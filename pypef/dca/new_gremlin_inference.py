#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 17 May 2023
# @authors: Niklas Siedhoff, Alexander-Maurice Illig
# @contact: <n.siedhoff@biotec.rwth-aachen.de>
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

"""
Code taken from GREMLIN repository available at https://github.com/sokrypton/GREMLIN_CPP/
and adapted (put functions into a class termed GREMLIN) and used under the
"THE BEER-WARE LICENSE" (Revision 42):
 ----------------------------------------------------------------------------------------
 "THE BEER-WARE LICENSE" (Revision 42):
 <so@fas.harvard.edu> wrote this file.  As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.  Sergey Ovchinnikov
 ----------------------------------------------------------------------------------------
--> Thanks for sharing the great code, I will gladly provide you a beer or two. (Niklas)
Code mainly taken from
https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb

References:
[1] Kamisetty, H., Ovchinnikov, S., & Baker, D.
    Assessing the utility of coevolution-based residue–residue contact predictions in a
    sequence- and structure-rich era.
    Proceedings of the National Academy of Sciences, 2013, 110, 15674-15679
    https://www.pnas.org/doi/10.1073/pnas.1314045110
[2] Balakrishnan, S., Kamisetty, H., Carbonell, J. G., Lee, S.-I., & Langmead, C. J.
    Learning generative models for protein fold families.
    Proteins, 79(4), 2011, 1061–78.
    https://doi.org/10.1002/prot.22934
[3] Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E.
    Improved contact prediction in proteins: Using pseudolikelihoods to infer Potts models.
    Physical Review E, 87(1), 2013, 012707. doi:10.1103/PhysRevE.87.012707
    https://doi.org/10.1103/PhysRevE.87.012707
"""

import logging
logger = logging.getLogger('pypef.dca.params_inference')

from os import mkdir, PathLike
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp
from scipy.stats import boxcox
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')

from pypef.utils.variant_data import get_sequences_from_file


class GREMLIN:
    """
    Alphabet char order in GREMLIN: "ARNDCQEGHILKMFPSTWYV-".
    gap_cutoff = 0.5 and eff_cutoff = 0.8 is proposed by Hopf et al.;
    here, by default all columns are used (gap_cutoff >= 1.0 and eff_cutoff > 1.0).
    v_ini represent the weighted frequency of each amino acid at each position, i.e.,
    np.log(np.sum(onehot_cat_msa.T * self.msa_weights, -1).T + pseudo_count).
    """
    def __init__(
            self,
            alignment: str | PathLike,
            char_alphabet: str = "ACDEFGHIKLMNPQRSTVWY-X",
            wt_seq=None,
            offset=0,
            optimize=True,
            gap_cutoff=1.0,
            eff_cutoff=1.0,
            opt_iter=100,
            max_msa_seqs: int | None = 10000,
            seqs=None
    ):
        self.char_alphabet = char_alphabet
        self.allowed_chars = "ABCDEFGHIKLMNPQRSTVWYX-."
        self.allowed_chars += self.allowed_chars.lower()
        self.offset = offset
        self.gap_cutoff = gap_cutoff
        self.eff_cutoff = eff_cutoff
        self.opt_iter = opt_iter
        if max_msa_seqs == None:
            self.max_msa_seqs = 1E9
        else:
            self.max_msa_seqs = max_msa_seqs
        self.states = len(self.char_alphabet)
        print('Loading MSA...')
        if seqs is None:
            self.seqs, _ = self.get_sequences_from_msa(alignment)
        else:
            self.seqs = seqs
        print(f'Found {len(self.seqs)} sequences in the MSA...')
        self.msa_ori = self.get_msa_ori()
        self.n_col_ori = self.msa_ori.shape[1]
        if wt_seq is not None:
            self.wt_seq = wt_seq
        else:  # Taking the first sequence in the MSA as wild type sequence
            self.wt_seq = "".join([self.char_alphabet[i] for i in self.msa_ori[0]])
            print("No wild-type sequence provided: The first sequence "
                  f"in the MSA is considered the wild-type sequence "
                  f"(Length: {len(self.wt_seq)}):\n{self.wt_seq}\n")
        if len(self.wt_seq) != self.n_col_ori:
            raise SystemError(f"Length of (provided) wild-type sequence ({len(self.wt_seq)}) "
                              f"does not match number of MSA columns ({self.n_col_ori}), "
                              f"i.e., common MSA sequence length.")
        print('Filtering gaps...')
        self.msa_trimmed, self.v_idx, self.w_idx, self.w_rel_idx, self.gaps = self.filt_gaps(self.msa_ori)
        print('Getting effective sequence weights...')
        self.msa_weights = self.get_eff_msa_weights(self.msa_trimmed)
        self.n_eff = np.sum(self.msa_weights)
        self.n_row = self.msa_trimmed.shape[0]
        self.n_col = self.msa_trimmed.shape[1]
        print('Initializing v and W terms based on MSA frequencies...')
        self.v_ini, self.w_ini, self.aa_counts = self.initialize_v_w(remove_gap_entries=False)
        self.aa_freqs = self.aa_counts / self.n_row
        self.optimize = optimize
        if self.optimize:
            self.v_opt_with_gaps, self.w_opt_with_gaps = self.run_opt_tf()
            no_gap_states = self.states - 1
            self.v_opt = self.v_opt_with_gaps[:, :no_gap_states],
            self.w_opt = self.w_opt_with_gaps[:, :no_gap_states, :, :no_gap_states]
        self.x_wt = self.collect_encoded_sequences(np.atleast_1d(self.wt_seq))

    def get_sequences_from_msa(self, msa_file: str):
        """
        "Get_Sequences" reads (learning and test) .fasta and
        .fasta-like ".fasl" format files and extracts the name,
        the target value and the sequence of the protein.
        Only takes one-liner sequences for correct input.
        See example directory for required fasta file format.
        Make sure every marker (> and ;) is seperated by a
        space ' ' from the value respectively name.
        Trimming MSA sequences starting at offset.

        msa_file: str
            Path to MSA in FASTA or A2M format.
        """
        sequences = []
        names_of_seqs = []
        with open(msa_file, 'r') as f:
            words = ""
            for line in f:
                if line.startswith('>'):
                    if words != "":
                        sequences.append(words[self.offset:])
                    words = line.split('>')
                    names_of_seqs.append(words[1].strip())
                    words = ""
                elif line.startswith('#'):
                    pass  # are comments
                else:
                    line = line.strip()
                    words += line
            if words != "":
                sequences.append(words[self.offset:])
        assert len(sequences) == len(names_of_seqs), f"{len(sequences)}, {len(names_of_seqs)}"
        return np.array(sequences), np.array(names_of_seqs)

    def a2n_dict(self):
        """convert alphabet to numerical integer values, e.g.:
        {"A": 0, "C": 1, "D": 2, ...}"""
        a2n = {}
        for a, n in zip(self.char_alphabet, range(self.states)):
            a2n[a] = n
        return a2n

    def aa2int(self, aa):
        """convert single aa into numerical integer value, e.g.:
        "A" -> 0 or "-" to 21 dependent on char_alphabet"""
        a2n = self.a2n_dict()
        if aa in a2n:
            return a2n[aa]
        else:  # for unknown characters insert Gap character
            return a2n['-']

    def seq2int(self, aa_seqs):
        """
        convert a single sequence or a list of sequences into a list of integer sequences, e.g.:
        ["ACD","EFG"] -> [[0,4,3], [6,13,7]]
        """
        if type(aa_seqs) == str:
            aa_seqs = np.array(aa_seqs)
        if type(aa_seqs) == list:
            aa_seqs = np.array(aa_seqs)
        if aa_seqs.dtype.type is np.str_:
            if aa_seqs.ndim == 0:  # single seq
                return np.array([self.aa2int(aa) for aa in str(aa_seqs)])
            else:  # list of seqs
                return np.array([[self.aa2int(aa) for aa in seq] for seq in aa_seqs])
        else:
            return aa_seqs

    @property
    def get_v_idx_w_idx(self):
        return self.v_idx, self.w_idx

    def get_msa_ori(self):
        """
        Converts list of sequences to MSA.
        Also checks for unknown amino acid characters and removes those sequences from the MSA.
        """
        n_skipped = 0
        msa_ori = []
        for i, seq in enumerate(self.seqs):
            skip = False
            for aa in seq:
                if aa not in self.allowed_chars:
                    if n_skipped == 0:
                        f"The input file(s) (MSA or train/test sets) contain(s) "
                        f"unknown protein sequence characters "
                        f"(e.g.: \"{aa}\" in sequence {i + 1}). "
                        f"Will remove those sequences from MSA!"
                    skip = True
                if skip:
                    n_skipped += 1
            if i <= self.max_msa_seqs and not skip:
                msa_ori.append([self.aa2int(aa.upper()) for aa in seq])
            elif i >= self.max_msa_seqs:
                print(f'Reached the number of maximal MSA sequences ({self.max_msa_seqs}), '
                       'skipping the rest of the MSA sequences...')
                break
        try:
            msa_ori = np.array(msa_ori)
        except ValueError:
            raise ValueError("The provided MSA seems to have inhomogeneous "
                             "shape, i.e., unequal sequence length.")
        return msa_ori

    def filt_gaps(self, msa_ori):
        """filters alignment to remove gappy positions"""
        tmp = (msa_ori == self.states - 1).astype(float)
        non_gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] < self.gap_cutoff)[0]

        gaps = np.where(np.sum(tmp.T, -1).T / msa_ori.shape[0] >= self.gap_cutoff)[0]
        print(f'Gap positions (removed from MSA; 0-indexed):\n{gaps}')
        ncol_trimmed = len(non_gaps)
        v_idx = non_gaps
        w_idx = v_idx[np.stack(np.triu_indices(ncol_trimmed, 1), -1)]
        w_rel_idx = np.stack(np.triu_indices(ncol_trimmed, 1), -1)
        return msa_ori[:, non_gaps], v_idx, w_idx, w_rel_idx, gaps

    def get_eff_msa_weights(self, msa):
        """compute effective weight for each sequence"""
        # pairwise identity
        pdistance_msa = pdist(msa, "hamming")
        msa_sm = 1.0 - squareform(pdistance_msa)
        # weight for each sequence
        msa_w = (msa_sm >= self.eff_cutoff).astype(float)
        msa_w = 1 / np.sum(msa_w, -1)
        return msa_w

    @staticmethod
    def l2(x):
        return np.sum(np.square(x))

    def objective(self, v, w=None, flattened=True):
        """Same objective function as used in run_opt_tf below
        but here only using numpy not TensorFlow functions.
        Potentially helpful for implementing SciPy optimizers."""
        if w is None:
            w = self.w_ini
        onehot_cat_msa = np.eye(self.states)[self.msa_trimmed]
        if flattened:
            v = np.reshape(v, (self.n_col, self.states))
            w = np.reshape(w, (self.n_col, self.states, self.n_col, self.states))
        ########################################
        # Pseudo-Log-Likelihood
        ########################################
        # v + w
        vw = v + np.tensordot(onehot_cat_msa, w, 2)
        # Hamiltonian
        h = np.sum(np.multiply(onehot_cat_msa, vw), axis=(1, 2))
        # local z (partition function)
        z = np.sum(np.log(np.sum(np.exp(vw), axis=2)), axis=1)
        # Pseudo-Log-Likelihood
        pll = h - z
        ########################################
        # Regularization
        ########################################
        l2_v = 0.01 * self.l2(v)
        l2_w = 0.01 * self.l2(w) * 0.5 * (self.n_col - 1) * (self.states - 1)
        # loss function to minimize
        loss = -np.sum(pll * self.msa_weights) / np.sum(self.msa_weights)
        loss = loss + (l2_v + l2_w) / self.n_eff
        return loss

    @staticmethod
    def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
        """
        Adam optimizer [https://arxiv.org/abs/1412.6980] with first and second moments
            mt          and
            vt          (greek letter nu) at time steps t, respectively.
        Note by GREMLIN authors: this is a modified version of adam optimizer.
        More specifically, we replace "vt" with sum(g*g) instead of (g*g).
        Furthermore, we find that disabling the bias correction
        (b_fix=False) speeds up convergence for our case.
        """
        if var_list is None:
            var_list = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(loss, var_list)
        if b_fix:
            t = tf.Variable(0.0, "t")
        opt = []
        for n, (x, g) in enumerate(zip(var_list, gradients)):
            if g is not None:
                ini = dict(initializer=tf.zeros_initializer, trainable=False)
                mt = tf.compat.v1.get_variable(name + "_mt_" + str(n), shape=list(x.shape), **ini)
                vt = tf.compat.v1.get_variable(name + "_vt_" + str(n), shape=[], **ini)

                mt_tmp = b1 * mt + (1 - b1) * g
                vt_tmp = b2 * vt + (1 - b2) * tf.reduce_sum(tf.square(g))
                lr_tmp = lr / (tf.sqrt(vt_tmp) + 1e-8)

                if b_fix:
                    lr_tmp = lr_tmp * tf.sqrt(1 - tf.pow(b2, t)) / (1 - tf.pow(b1, t))

                opt.append(x.assign_add(-lr_tmp * mt_tmp))
                opt.append(vt.assign(vt_tmp))
                opt.append(mt.assign(mt_tmp))

        if b_fix:
            opt.append(t.assign_add(1.0))
        return tf.group(opt)

    @staticmethod
    def sym_w(w):
        """
        Symmetrize input matrix of shape (x,y,x,y)
        As the full couplings matrix W might/will be slightly "unsymmetrical"
        it will be symmetrized according to one half being "mirrored".
        """
        x = w.shape[0]
        w = w * np.reshape(1 - np.eye(x), (x, 1, x, 1))
        w = w + tf.transpose(w, [2, 3, 0, 1])
        return w

    @staticmethod
    def l2_tf(x):
        return tf.reduce_sum(tf.square(x))

    def run_opt_tf(self, opt_rate=1.0, batch_size=None):
        """
        For optimization of v and w ADAM is used here (L-BFGS-B not (yet) implemented
        for TF 2.x, e.g. using scipy.optimize.minimize).
        Gaps (char '-' respectively '21') included.
        """
        ##############################################################
        # SETUP COMPUTE GRAPH
        ##############################################################
        # kill any existing tensorflow graph
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        # msa (multiple sequence alignment)
        msa = tf.compat.v1.placeholder(tf.int32, shape=(None, self.n_col), name="msa")

        # one-hot encode msa
        oh_msa = tf.one_hot(msa, self.states)

        # msa weights
        msa_weights = tf.compat.v1.placeholder(tf.float32, shape=(None,), name="msa_weights")

        # 1-body-term of the MRF
        v = tf.compat.v1.get_variable(name="v",
                                      shape=[self.n_col, self.states],
                                      initializer=tf.compat.v1.zeros_initializer)

        # 2-body-term of the MRF
        w = tf.compat.v1.get_variable(name="w",
                                      shape=[self.n_col, self.states, self.n_col, self.states],
                                      initializer=tf.compat.v1.zeros_initializer)

        # symmetrize w
        w = self.sym_w(w)

        ########################################
        # Pseudo-Log-Likelihood
        ########################################
        # v + w
        vw = v + tf.tensordot(oh_msa, w, 2)

        # Hamiltonian
        h = tf.reduce_sum(tf.multiply(oh_msa, vw), axis=(1, 2))
        # partition function Z
        z = tf.reduce_sum(tf.reduce_logsumexp(vw, axis=2), axis=1)

        # Pseudo-Log-Likelihood
        pll = h - z

        ########################################
        # Regularization
        ########################################
        l2_v = 0.01 * self.l2_tf(v)
        lw_w = 0.01 * self.l2_tf(w) * 0.5 * (self.n_col - 1) * (self.states - 1)

        # loss function to minimize
        loss = -tf.reduce_sum(pll * msa_weights) / tf.reduce_sum(msa_weights)
        loss = loss + (l2_v + lw_w) / self.n_eff

        ##############################################################
        # MINIMIZE LOSS FUNCTION
        ##############################################################
        opt = self.opt_adam(loss, "adam", lr=opt_rate)
        # initialize V (local fields)
        msa_cat = tf.keras.utils.to_categorical(self.msa_trimmed, self.states)
        pseudo_count = 0.01 * np.log(self.n_eff)
        v_ini = np.log(np.sum(msa_cat.T * self.msa_weights, -1).T + pseudo_count)
        v_ini = v_ini - np.mean(v_ini, -1, keepdims=True)

        # generate input/feed
        def feed(feed_all=False):
            if batch_size is None or feed_all:
                return {msa: self.msa_trimmed, msa_weights: self.msa_weights}
            else:
                idx = np.random.randint(0, self.n_row, size=batch_size)
                return {msa: self.msa_trimmed[idx], msa_weights: self.msa_weights[idx]}

        with tf.compat.v1.Session() as sess:
            # initialize variables V (local fields) and W (couplings)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(v.assign(v_ini))
            # compute loss across all data
            get_loss = lambda: round(sess.run(loss, feed(feed_all=True)) * self.n_eff, 2)
            print(f"Initial loss: {get_loss()}. Starting parameter optimization...")
            for i in range(self.opt_iter):
                sess.run(opt, feed())
                try:
                    if (i + 1) % int(self.opt_iter / 10) == 0:
                        print(f"Iteration {(i + 1)} {get_loss()}")
                except ZeroDivisionError:
                    print(f"Iteration {(i + 1)} {get_loss()}")
            # save the v and w parameters of the MRF
            v_opt = sess.run(v)
            w_opt = sess.run(w)
        return v_opt, w_opt


    def initialize_v_w(self, remove_gap_entries=True):
        """
        For optimization of v and w ADAM is used here (L-BFGS-B not (yet)
        implemented for TF 2.x, e.g. using scipy.optimize.minimize).
        Gaps (char '-' respectively '21') included.
        """
        w_ini = np.zeros((self.n_col, self.states, self.n_col, self.states))
        onehot_cat_msa = np.eye(self.states)[self.msa_trimmed]
        aa_counts = np.sum(onehot_cat_msa, axis=0)
        pseudo_count = 0.01 * np.log(self.n_eff)
        v_ini = np.log(np.sum(onehot_cat_msa.T * self.msa_weights, -1).T + pseudo_count)
        v_ini = v_ini - np.mean(v_ini, -1, keepdims=True)
        # loss_score_ini = self.objective(v_ini, w_ini, flattened=False)

        if remove_gap_entries:
            no_gap_states = self.states - 1
            v_ini = v_ini[:, :no_gap_states]
            w_ini = w_ini[:, :no_gap_states, :, :no_gap_states]
            aa_counts = aa_counts[:, :no_gap_states]

        return v_ini, w_ini, aa_counts

    @property
    def get_v_w_opt(self):
        try:
            return self.v_opt, self.w_opt
        except AttributeError:
            raise SystemError(
                "No v_opt and w_opt available, this means GREMLIN "
                "has not been initialized setting optimize to True, "
                "e.g., try GREMLIN('Alignment.fasta', optimize=True)."
            )

    def get_score(self, seqs, v=None, w=None, v_idx=None, encode=False, h_wt_seq=0.0, recompute_z=False):
        """
        Computes the GREMLIN score for a given sequence or list of sequences.
        """
        if v is None or w is None:
            if self.optimize:
                v, w = self.v_opt, self.w_opt
            else:
                v, w, _ = self.initialize_v_w(remove_gap_entries=True)
        if v_idx is None:
            v_idx = self.v_idx
        seqs_int = self.seq2int(seqs)
        # if length of sequence != length of model use only
        # valid positions (v_idx) from the trimmed alignment
        try:
            if seqs_int.shape[-1] != len(v_idx):
                seqs_int = seqs_int[..., v_idx]
        except IndexError:
            raise SystemError(
                "The loaded GREMLIN parameter model does not match the input model "
                "in terms of sequence encoding shape or is a gap-substituted sequence. "
                "E.g., when providing two different DCA models/parameters provided by: "
                "\"-m DCA\" and \"--params GREMLIN\", where -m DCA represents a ml input "
                "model potentially generated using plmc parameters and --params GREMLIN "
                "provides differently encoded sequences generated using GREMLIN."
            )

        # one hot encode
        x = np.eye(self.states)[seqs_int]
        # aa_pos_counts = np.sum(x, axis=0)

        # get non-gap positions
        # no_gap = 1.0 - x[..., -1]

        # remove gap from one-hot-encoding
        x = x[..., :-1]

        # compute score
        vw = v + np.tensordot(x, w, 2)

        # ============================================================================================
        # Note, Z (the partition function) is a constant. In GREMLIN, V, W & Z are estimated using all
        # the original weighted input sequence(s). It is NOT recommended to recalculate z with a
        # different set of sequences. Given the common ERROR of recomputing Z, we include the option
        # to do so, for comparison.
        # ============================================================================================
        h = np.sum(np.multiply(x, vw), axis=-1)

        if encode:
            return h

        if recompute_z:
            z = logsumexp(vw, axis=-1)
            return np.sum((h - z), axis=-1) - h_wt_seq
        else:
            return np.sum(h, axis=-1) - h_wt_seq

    def get_wt_score(self, wt_seq=None, v=None, w=None, encode=False):
        if wt_seq is None:
            wt_seq = self.wt_seq
        if v is None or w is None:
            if self.optimize:
                v, w = self.v_opt, self.w_opt
            else:
                v, w = self.v_ini, self.w_ini
        wt_seq = np.array(wt_seq, dtype=str)
        return self.get_score(wt_seq, v, w, encode=encode)

    def collect_encoded_sequences(self, seqs, v=None, w=None, v_idx=None):
        """
        Wrapper function for encoding input sequences using the self.get_score
        function with encode set to True.
        """
        xs = self.get_score(seqs, v, w, v_idx, encode=True)
        return xs

    @staticmethod
    def normalize(apc_mat):
        """
        Normalization of APC matrix for getting z-Score matrix
        """
        dim = apc_mat.shape[0]
        apc_mat_flat = apc_mat.flatten()
        x, _ = boxcox(apc_mat_flat - np.amin(apc_mat_flat) + 1.0)
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / x_std
        x = x.reshape(dim, dim)
        return x

    def mtx_gaps_as_zeros(self, gap_reduced_mtx, insert_gap_zeros=True):
        """
        Inserts zeros at gap positions of the (L,L) matrices,
        i.e., raw/apc/zscore matrices.
        """
        if insert_gap_zeros:
            gap_reduced_mtx = list(gap_reduced_mtx)
            mtx_zeroed = []
            c = 0
            for i in range(self.n_col_ori):
                mtx_i = []
                c_i = 0
                if i in self.gaps:
                    mtx_zeroed.append(list(np.zeros(self.n_col_ori)))
                else:
                    for j in range(self.n_col_ori):
                        if j in self.gaps:
                            mtx_i.append(0.0)
                        else:
                            mtx_i.append(gap_reduced_mtx[c][c_i])
                            c_i += 1
                    mtx_zeroed.append(mtx_i)
                    c += 1
            return np.array(mtx_zeroed)

        else:
            return gap_reduced_mtx

    def get_correlation_matrix(self, matrix_type: str = 'apc', insert_gap_zeros=False):
        """
        Requires optimized w matrix (of shape (L, 20, L, 20))
        inputs
        ------------------------------------------------------
        w           : coevolution       shape=(L,A,L,A)
        ------------------------------------------------------
        outputs
        ------------------------------------------------------
        raw         : l2norm(w)         shape=(L,L)
        apc         : apc(raw)          shape=(L,L)
        zscore      : normalize(apc)    shape=(L,L)
        """
        # l2norm of 20x20 matrices (note: gaps already excluded)
        raw = np.sqrt(np.sum(np.square(self.w_opt), (1, 3)))

        # apc (average product correction)
        ap = np.sum(raw, 0, keepdims=True) * np.sum(raw, 1, keepdims=True) / np.sum(raw)
        apc = raw - ap

        if matrix_type == 'apc':
            return self.mtx_gaps_as_zeros(apc, insert_gap_zeros=insert_gap_zeros)
        elif matrix_type == 'raw':
            return self.mtx_gaps_as_zeros(raw, insert_gap_zeros=insert_gap_zeros)
        elif matrix_type == 'zscore' or matrix_type == 'z_score':
            return self.mtx_gaps_as_zeros(self.normalize(apc), insert_gap_zeros=insert_gap_zeros)
        else:
            raise SystemError("Unknown matrix type. Choose between 'apc', 'raw', or 'zscore'.")

    def plot_correlation_matrix(self, matrix_type: str = 'apc', set_diag_zero=True):
        matrix = self.get_correlation_matrix(matrix_type, insert_gap_zeros=True)
        if set_diag_zero:
            np.fill_diagonal(matrix, 0.0)

        fig, ax = plt.subplots(figsize=(10, 10))

        if matrix_type == 'zscore' or matrix_type == 'z_score':
            ax.imshow(matrix, cmap='Blues', interpolation='none', vmin=1, vmax=3)
        else:
            ax.imshow(matrix, cmap='Blues')
        tick_pos = ax.get_xticks()
        tick_pos = np.array([int(t) for t in tick_pos])
        tick_pos[-1] = matrix.shape[0]
        if tick_pos[2] > 1:
            tick_pos[2:] -= 1
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        try:
            labels = [labels[0]] + [str(int(label) + 1) for label in labels[1:]]
        except ValueError:
            pass
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlim(-1, matrix.shape[0])
        ax.set_ylim(-1, matrix.shape[0])
        plt.title(matrix_type.upper())
        plt.savefig(f'{matrix_type}.png', dpi=500)
        plt.close('all')

    def get_top_coevolving_residues(self, wt_seq=None, min_distance=0, sort_by="apc"):
        if wt_seq is None:
            wt_seq = self.wt_seq
        if wt_seq is None:
            raise SystemError("Getting top co-evolving residues requires "
                              "the wild type sequence as input.")
        raw = self.get_correlation_matrix(matrix_type='raw')
        apc = self.get_correlation_matrix(matrix_type='apc')
        zscore = self.get_correlation_matrix(matrix_type='zscore')

        # Explore top co-evolving residue pairs
        i_rel_idx = self.w_rel_idx[:, 0]
        j_rel_idx = self.w_rel_idx[:, 1]

        apc_flat = []
        zscore_flat = []
        raw_flat = []
        for i, _ in enumerate(i_rel_idx):
            raw_flat.append(raw[i_rel_idx[i]][j_rel_idx[i]])
            apc_flat.append(apc[i_rel_idx[i]][j_rel_idx[i]])
            zscore_flat.append(zscore[i_rel_idx[i]][j_rel_idx[i]])

        i_idx = self.w_idx[:, 0]
        j_idx = self.w_idx[:, 1]

        i_aa = [f"{wt_seq[i]}_{i + 1}" for i in i_idx]
        j_aa = [f"{wt_seq[j]}_{j + 1}" for j in j_idx]

        # load mtx into pandas dataframe
        mtx = {
            "i": i_idx, "j": j_idx, "apc": apc_flat, "zscore": zscore_flat,
            "raw": raw_flat, "i_aa": i_aa, "j_aa": j_aa
        }
        df_mtx = pd.DataFrame(mtx, columns=["i", "j", "apc", "zscore", "raw", "i_aa", "j_aa"])
        df_mtx_sorted = df_mtx.sort_values(sort_by, ascending=False)

        # get contacts with sequence separation > min_distance
        df_mtx_sorted_mindist = df_mtx_sorted.loc[df_mtx['j'] - df_mtx['i'] > min_distance]

        return df_mtx_sorted_mindist


"""
GREMLIN class helper functions below.
"""


def save_gremlin_as_pickle(alignment: str, wt_seq: str, opt_iter: int = 100):
    """
    Function for getting and/or saving (optimized or unoptimized) GREMLIN model
    """
    logger.info(f'Inferring GREMLIN DCA parameters based on the provided MSA...')
    gremlin = GREMLIN(alignment, wt_seq=wt_seq, optimize=True, opt_iter=opt_iter)
    try:
        mkdir('Pickles')
    except FileExistsError:
        pass

    logger.info(f'Saving GREMLIN model as Pickle file...')
    pickle.dump(
        {
            'model': gremlin,
            'model_type': 'GREMLINpureDCA',
            'beta_1': None,
            'beta_2': None,
            'spearman_rho': None,
            'regressor': None
        },
        open('Pickles/GREMLIN', 'wb')
    )
    return gremlin


def plot_all_corr_mtx(gremlin: GREMLIN):
    gremlin.plot_correlation_matrix(matrix_type='raw')
    gremlin.plot_correlation_matrix(matrix_type='apc')
    gremlin.plot_correlation_matrix(matrix_type='zscore')


def save_corr_csv(gremlin: GREMLIN, min_distance: int = 0, sort_by: str = 'apc'):
    df_mtx_sorted_mindist = gremlin.get_top_coevolving_residues(
        min_distance=min_distance, sort_by=sort_by
    )
    df_mtx_sorted_mindist.to_csv(f"coevolution_{sort_by}_sorted.csv")