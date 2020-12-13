#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

import matplotlib
matplotlib.use('Agg')  # no plt.show(), just save plot
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # default: refit=True

# import regression models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# import adjust_text  # only locally imported for labeled validation plots and in silico directed evolution

import sys
from tqdm import tqdm  # progress bars
import warnings
import random
# ignoring warnings of PLS regression using n_components
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def read_models(number):
    """
    reads the models found in the file Model_Results.txt.
    If no model was trained, the .txt file does not exist.
    """
    try:
        ls = ""
        with open('Model_Results.txt', 'r') as file:
            for i, lines in enumerate(file):
                if i == 0:
                    if lines[:6] == 'No FFT':
                        number += 2
                if i <= number + 1:
                    ls += lines
        return ls
    except FileNotFoundError:
        return "No Model_Results.txt found."


def Full_Path(Filename):
    """
    returns the path of an index inside the folder /AAindex/,
    e.g. path/to/AAindex/FAUJ880109.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return (os.path.join(modules_path, 'AAindex/'+Filename))


def Path_AAindex_Dir():
    """
    returns the path to the /AAindex folder, e.g. path/to/AAindex/.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AAindex')


class AAindex:
    """
    gets all the information that are given in each AAindex file.
    For the program routine it provides the library to enable translation
    of the alphabetical amino acid sequence to an array of numericals.
    """
    def __init__(self, Filename):
        self.File = Filename
        self.Accession_Number = None
        self.Data_Description = None
        self.PMID = None
        self.Authors = None
        self.Title_Of_Article = None
        self.Journal_Reference = None

    def General_Information(self):
        """
        Gets and allocates general information based on the AAindex file
        format defined by file sections 'H', 'D', 'E', 'A', 'T', 'J'
        """
        with open(self.File, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines.
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract some general information about AAindex file.
                if id_letter == 'H':
                    self.Accession_Number = words[1]
                elif id_letter == 'D':
                    self.Data_Description = words[1]
                elif id_letter == 'E':
                    self.PMID = words[1:]
                elif id_letter == 'A':
                    self.Authors = ' '.join(words[1:])
                elif id_letter == 'T':
                    self.Title_Of_Article = ' '.join(words[1:])
                elif id_letter == 'J':
                    self.Journal_Reference = ' '.join(words[1:])

    def Encoding_Dictionary(self):
        """
        Get numerical values of AAindex for each amino acid
        """
        with open(self.File, 'r') as f:
            for line in f:
                # try/ except "removes" empty lines
                try:
                    words = line.split()
                    id_letter = words[0]
                except IndexError:
                    break

                # Extract numerical values of AAindex.
                if id_letter == 'I':

                    Keys = []
                    for word in words[1:]:
                        Keys.append(word[0])
                        Keys.append(word[-1])

                    Values = []
                    for row in range(2):
                        line = f.readline()
                        strings = line.split()
                        for idx, string in enumerate(strings):
                            #Some Aminoacids may have no value.
                            try:
                                strings[idx] = float(string)
                            except ValueError:
                                strings[idx] = None
                        Values.append(strings)
                    Values = np.reshape(np.array(Values).T, len(Keys))

                    return dict(zip(Keys, Values))


def Get_Sequences(Fasta, Mult_Path=None, Prediction=False):
    """
    "Get_Sequences" reads (learning and validation).fasta format files and extracts the name,
    the target value and the sequence of the peptide. See example directory for required fasta file format.
    Make sure every marker (> and ;) is seperated by an space ' ' from the value/ name.
    """
    if Mult_Path is not None:
        os.chdir(Mult_Path)

    Sequences = []
    Values = []
    Names_Of_Mutations = []

    with open(Fasta, 'r') as f:
        for line in f:
            if '>' in line:
                words = line.split()
                Names_Of_Mutations.append(words[1])
                # words[1] is appended so make sure there is a space in between > and the name!

            elif '#' in line:
                pass  # are Comments

            elif ';' in line:
                words = line.split()
                Values.append(float(words[1]))
                # words[1] is appended so make sure there is a space in between ; and the value!

            else:
                try:
                    words = line.split()
                    Sequences.append(words[0])
                except IndexError:
                    raise IndexError("Learning or Validation sets (.fasta) likely have emtpy lines at end of file")

    # Check consistency
    if Prediction == False:
        if len(Sequences) != len(Values):
            print('Error: Number of sequences does not fit with number of target values!')
            print('Number of sequences: {}, Number of target values: {}.'.format(str(len(Sequences)), str(len(Values))))
            sys.exit()

    return Sequences, Names_Of_Mutations, Values


class XY:
    """
    converts the string sequence into a list of numericals using the AAindex translation library,
    Fourier transforming the numerical array that was translated by Get_Numerical_Sequence (Do_Fourier),
    computing the input matrices X and Y for the PLS regressor (Get_X_And_Y)
    """
    def __init__(self, AAindex_File, Fasta_File, Mult_Path=None, Prediction=False):
        aaidx = AAindex(AAindex_File)
        self.dictionary = aaidx.Encoding_Dictionary()
        self.sequences, self.names, self.values = Get_Sequences(Fasta_File, Mult_Path, Prediction)

    def Get_Numerical_Sequence(self, Sequence):
        return (np.array([self.dictionary[aminoacid] for aminoacid in Sequence]))

    def Do_Fourier(self, Sequence):
        """
        This function does the Fast Fourier Transform. Since the condition

                    len(Array) = 2^k -> k = log_2(len(Array))
                    k in N

        must be satisfied, the array must be reshaped (zero padding) if k is no integer value.
        The verbose parameter prints also the real and imaginary part separately.
        """
        threshold = 1e-8  # errors due to computer uncertainties
        k = np.log2(Sequence.size)  # get exponent k
        Mean = np.mean(Sequence, axis=0)  # calculate mean of numerical array
        Sequence = np.subtract(Sequence, Mean)  # subtract mean to avoid artificial effects of FT

        if (abs(int(k) - k) > threshold):  # check if length of array fulfills previous equation
            Numerical_Sequence_Reshaped = np.zeros(pow(2, (int(k) + 1)))  # reshape array
            for index, value in enumerate(Sequence):
                Numerical_Sequence_Reshaped[index] = value
            Sequence = Numerical_Sequence_Reshaped

        Fourier_Transformed = np.fft.fft(Sequence)  # FFT
        FT_real = np.real(Fourier_Transformed)
        FT_imag = np.imag(Fourier_Transformed)

        x = np.linspace(1, Sequence.size, Sequence.size)  # frequencies
        x = x / max(x)  # normalization of frequency

        Amplitude = FT_real * FT_real + FT_imag * FT_imag

        if (max(Amplitude) != 0):
            Amplitude = np.true_divide(Amplitude, max(Amplitude))  # normalization of amplitude

        return Amplitude, x

    def Get_X_And_Y(self):
        """
        getting the input matrices X (FFT amplitudes) and Y (variant labels)
        """
        Frequencies = []
        Amplitudes = []
        raw_numerical_seq = []

        for sequence in self.sequences:
            num = self.Get_Numerical_Sequence(sequence)

            # There may be amino acids without a value in AAindex.
            # Skip these Indices.
            if None in num:
                break

            # Numerical sequence gets expended by zeros so that also different lengths of sequences
            # can be processed using '--nofft' option
            k = np.log2(len(num))
            if abs(int(k) - k) > 1e-8:  # check if length of array fulfills previous equation
                num = np.append(num, np.zeros(pow(2, (int(k) + 1)) - len(num)))  # reshape array

            amplitudes, frequencies = self.Do_Fourier(num)

            # Fourier spectra are mirrored at frequency = 0.5. No more information at higher frequencies.
            half = len(frequencies) // 2  # // for integer division
            Frequencies.append(frequencies[:half])
            Amplitudes.append(amplitudes[:half])    # FFT-ed encoded amino acid sequences
            raw_numerical_seq.append(num)           # Raw encoded amino acid sequences

        Amplitudes = np.array(Amplitudes)
        Frequencies = np.array(Frequencies)
        raw_numerical_seq = np.array(raw_numerical_seq)

        X = Amplitudes
        Y = self.values                             # Fitness values (sequence labels)

        return X, Y, raw_numerical_seq


def Get_R2(X_learn, X_valid, Y_learn, Y_valid, regressor='pls'):
    """
    The function Get_R2 takes features and labels from the learning and validation set.

    When using 'pls' as regressor, the MSE is calculated for all LOOCV sets for predicted vs true labels
    (mse = mean_squared_error(y_test_loo, y_pred_loo) for a fixed number of components for PLS regression.
    In the next iteration, the number of components is increased by 1 (number_of_components += 1)
    and the MSE is calculated for this regressor. The loop breaks if i > 9.
    Finally, the model of the single AAindex model with the lowest MSE is chosen.

    When using other regressors the parameters are tuned using GridSearchCV.

    This function returnes performance (R2, (N)RMSE, Pearson's r) and model parameters.
    """
    regressor = regressor.lower()
    Mean_Squared_Error = []

    if regressor == 'pls':
        # PLS regression as used by Cadet et al.
        # https://doi.org/10.1186/s12859-018-2407-8
        # https://doi.org/10.1038/s41598-018-35033-y
        # Hyperparameter (N component) tuning of PLS regressor
        for n_comp in range(1, 10):
            pls = PLSRegression(n_components=n_comp)
            loo = LeaveOneOut()

            y_pred_loo = []
            y_test_loo = []

            for train, test in loo.split(X_learn):
                x_learn_loo = []
                y_learn_loo = []
                x_test_loo = []

                for j in train:
                    x_learn_loo.append(X_learn[j])
                    y_learn_loo.append(Y_learn[j])

                for k in test:
                    x_test_loo.append(X_learn[k])
                    y_test_loo.append(Y_learn[k])

                pls.fit(x_learn_loo, y_learn_loo)
                y_pred_loo.append(pls.predict(x_test_loo)[0][0])

            mse = mean_squared_error(y_test_loo, y_pred_loo)

            Mean_Squared_Error.append(mse)

        Mean_Squared_Error = np.array(Mean_Squared_Error)
        idx = np.where(Mean_Squared_Error == np.min(Mean_Squared_Error))[0][0] + 1  # finds best number of components

        # Model is fitted with best n_components (lowest MSE)
        best_params = {'n_components': idx}
        regressor_ = PLSRegression(n_components=best_params.get('n_components'))

    # other regression options (CV tuning)
    elif regressor == 'pls_cv':
        params = {'n_components': list(np.arange(1, 10))}
        regressor_ = GridSearchCV(PLSRegression(), param_grid=params, iid=False, cv=5)

    elif regressor == 'rf':
        params = {                      # quite similar tu Xu et al., https://doi.org/10.1021/acs.jcim.0c00073
            'random_state': [42],
            'n_estimators': [100, 250, 500, 1000],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        regressor_ = GridSearchCV(RandomForestRegressor(), param_grid=params, iid=False, cv=5)

    elif regressor == 'svr':
        params = {                      # quite similar tu Xu et al.
            'C': [2 ** 0, 2 ** 2, 2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12],
            'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]
        }
        regressor_ = GridSearchCV(SVR(), param_grid=params, iid=False, cv=5)

    else:
        raise SystemError("Did not find specified regression model as valid option. See '--help' for valid "
                          "regression model options.")

    regressor_.fit(X_learn, Y_learn)  # fit model

    if regressor != 'pls':      # take best parameters for the regressor and the AAindex
        best_params = regressor_.best_params_

    Y_pred = []
    for y_p in regressor_.predict(X_valid):  # predict validation entries with fitted model
        Y_pred.append(float(y_p))

    R2 = r2_score(Y_valid, Y_pred)
    RMSE = np.sqrt(mean_squared_error(Y_valid, Y_pred))
    NRMSE = RMSE / np.std(Y_valid, ddof=1)
    with warnings.catch_warnings():  # catching RunTime warning when there's no variance in an array, e.g. [2, 2, 2, 2]
        warnings.simplefilter("ignore")  # which would mean divide by zero
        pearson_r = np.corrcoef(Y_valid, Y_pred)[0][1]

    return R2, RMSE, NRMSE, pearson_r, regressor, best_params


def R2_List(Learning_Set, Validation_Set, regressor='pls', noFFT=False):
    """
    returns the sorted list of all the model parameters and
    the performance values (R2 etc.) from function Get_R2.
    """
    AAindices = [file for file in os.listdir(Path_AAindex_Dir()) if file.endswith('.txt')]

    AAindex_R2_List = []
    for index, aaindex in enumerate(tqdm(AAindices)):
        xy_learn = XY(Full_Path(aaindex), Learning_Set)
        if noFFT == False:  # X is FFT-ed of encoded alphabetical sequence
            x_learn, y_learn, _ = xy_learn.Get_X_And_Y()
        else:               # X is raw encoded of alphabetical sequence
            _, y_learn, x_learn = xy_learn.Get_X_And_Y()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip
        if len(x_learn) != 0:
            xy_test = XY(Full_Path(aaindex), Validation_Set)
            if noFFT == False:  # X is FFT-ed of the encoded alphabetical sequence
                x_test, y_test, _ = xy_test.Get_X_And_Y()
            else:               # X is the raw encoded of alphabetical sequence
                _, y_test, x_test = xy_test.Get_X_And_Y()
            r2, rmse, nrmse, pearson_r, regression_model, params = Get_R2(x_learn, x_test, y_learn, y_test, regressor)
            AAindex_R2_List.append([aaindex, r2, rmse, nrmse, pearson_r, regression_model, params])

    AAindex_R2_List.sort(key=lambda x: x[1], reverse=True)

    return AAindex_R2_List


def Formatted_Output(AAindex_R2_List, noFFT=False, Minimum_R2=0.0):
    """
    takes the sorted list from function R2_List and writes the model names with an R2 ≥ 0
    as well as the corresponding parameters for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """

    index, value, value2, value3, value4, regression_model, params = [], [], [], [], [], [], []

    for (idx, val, val2, val3, val4, r_m, pam) in AAindex_R2_List:
        if val >= Minimum_R2:
            index.append(idx[:-4])
            value.append('{:f}'.format(val))
            value2.append('{:f}'.format(val2))
            value3.append('{:f}'.format(val3))
            value4.append('{:f}'.format(val4))
            regression_model.append(r_m.upper())
            params.append(pam)

    if len(value) == 0:
        raise ValueError('No model with positive R2.')

    data = np.array([index, value, value2, value3, value4, regression_model, params]).T
    col_width = max(len(str(value)) for row in data for value in row[:-1]) + 5

    head = ['Index', 'R2', 'RMSE', 'NRMSE', 'Pearson\'s r', 'Regression', 'Model parameters']
    with open('Model_Results.txt', 'w') as f:
        if noFFT is not False:
            f.write("No FFT used in this model construction, performance"
                    " represents model accuracies on raw encoded sequence data.\n\n")

        heading = "".join(caption.ljust(col_width) for caption in head) + '\n'
        f.write(heading)

        row_length = []
        for row in data:
            row_ = "".join(str(value).ljust(col_width) for value in row) + '\n'
            row_length.append(len(row_))
        row_length_max = max(row_length)
        f.write(row_length_max * '-' + '\n')

        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')

    return ()


def cross_validation(X, Y, regressor_, n_samples=5):
    # perform k-fold cross-validation on all data
    # k = Number of splits, change for changing k in k-fold split-up, default=5
    Y_test_total = []
    Y_predicted_total = []

    kf = KFold(n_splits=n_samples, shuffle=True)
    for train_index, test_index in kf.split(Y):
        Y = np.array(Y)
        try:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            for numbers in Y_test:
                Y_test_total.append(numbers)
            regressor_.fit(X_train, Y_train)  # Fitting on a random subset for Final_Model
            # (and on a subset subset for Learning_Model)
            # Predictions for samples in the test_set during that iteration
            pred_for_test_set_samples = regressor_.predict(X[test_index])
            for values in pred_for_test_set_samples:
                Y_predicted_total.append(float(values))
        except UserWarning:
            continue

    return Y_test_total, Y_predicted_total


def Save_Model(Path, Fasta_File, AAindex_R2_List, Learning_Set, Validation_Set, Threshold=5, regressor='pls',
               noFFT=False):
    """
    Function Save_Model saves the best -s THRESHOLD models as 'Pickle' files (pickle.dump),
    which can be loaded again for doing predictions. Also, in Save_Model included is the def cross_validation
    -based computing of the k-fold CV performance of the n component-optimized model on all data
    (learning + validation set); by default  k  is 5 (n_samples = 5).
    Plots of the CV performance for the t best models are stored inside the folder CV_performance.
    """
    regressor = regressor.lower()
    try:
        os.mkdir('CV_performance')
    except FileExistsError:
        pass
    try:
        os.mkdir('Pickles')
    except FileExistsError:
        pass

    for t in range(Threshold):
        try:
            idx = AAindex_R2_List[t][0]
            parameter = AAindex_R2_List[t][6]

            # Estimating the CV performance of the n_component-fitted model on all data
            xy_learn = XY(Full_Path(idx), Learning_Set)
            xy_test = XY(Full_Path(idx), Validation_Set)
            if noFFT is False:
                x_test, y_test, _ = xy_test.Get_X_And_Y()
                x_learn, y_learn, _ = xy_learn.Get_X_And_Y()
            else:
                _, y_test, x_test = xy_test.Get_X_And_Y()
                _, y_learn, x_learn = xy_learn.Get_X_And_Y()

            X = np.concatenate([x_learn, x_test])
            Y = np.concatenate([y_learn, y_test])

            if regressor == 'pls' or regressor == 'pls_cv':
                # n_components according to lowest MSE for validation set
                regressor_ = PLSRegression(n_components=parameter.get('n_components'))

            elif regressor == 'rf':
                regressor_ = RandomForestRegressor(random_state=parameter.get('random_state'),
                                                   n_estimators=parameter.get('n_estimators'),
                                                   max_features=parameter.get('max_features'))

            elif regressor == 'svr':
                regressor_ = SVR(C=parameter.get('C'), gamma=parameter.get('gamma'))

            else:
                raise SystemError("Did not find specified regression model as valid option. See '--help' for valid "
                         "regression model options.")

            # perform 5-fold cross-validation on all data
            n_samples = 5
            Y_test_total, Y_predicted_total = cross_validation(X, Y, regressor_, n_samples)

            r_squared = r2_score(Y_test_total, Y_predicted_total)
            rmse = np.sqrt(mean_squared_error(Y_test_total, Y_predicted_total))
            stddev = np.std(Y_test_total, ddof=1)
            nrmse = rmse / stddev
            pearson_r = np.corrcoef(Y_test_total, Y_predicted_total)[0][1]
            figure, ax = plt.subplots()
            ax.scatter(Y_test_total, Y_predicted_total, marker='o', s=20, linewidths=0.5, edgecolor='black')
            ax.plot([min(Y_test_total) - 1, max(Y_test_total) + 1],
                    [min(Y_predicted_total) - 1, max(Y_predicted_total) + 1], 'k', lw=2)
            ax.legend(['R$^2$ = ' + str(round(r_squared, 3)) + '\nRMSE = ' + str(round(rmse, 3)) +
                       '\nNRMSE = ' + str(round(nrmse, 3)) + '\nPearson\'s $r$ = ' + str(round(pearson_r, 3))])
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            plt.savefig('CV_performance/' + idx[:-4] + '_' + str(n_samples) + '-fold-CV.png', dpi=250)
            plt.close('all')

            # fit on full learning set
            xy = XY(Full_Path(idx), Fasta_File)
            X, Y, X_raw = xy.Get_X_And_Y()

            if noFFT is False:
                regressor_.fit(X, Y)
            else:
                regressor_.fit(X_raw, Y)

            file = open(os.path.join(Path, 'Pickles/'+idx[:-4]), 'wb')
            pickle.dump(regressor_, file)
            file.close()

        except IndexError:
            break

    return ()


def Predict(Path, Prediction_Set, Model, Mult_Path=None, noFFT=False, print_matrix=False):
    """
    The function Predict is used to perform predictions.
    Saved pickle files of models will be loaded again (mod = pickle.load(file))
    and used for predicting the label Y (Y = mod.predict(X)) of sequences given in the Prediction_Set.fasta.
    """
    aaidx = Full_Path(str(Model) + '.txt')
    xy = XY(aaidx, Prediction_Set, Mult_Path, Prediction=True)
    X, _, X_raw = xy.Get_X_And_Y()

    file = open(os.path.join(Path, 'Pickles/'+str(Model)), 'rb')
    mod = pickle.load(file)
    file.close()

    try:
        Y_ = []
        if noFFT is False:
            Y = mod.predict(X)
            for y in Y:
                Y_.append([float(y)])   # just make sure predicted Y is nested list of list [[Y_1], [Y_2], ..., [Y_N]]
        else:
            Y = mod.predict(X_raw)
            for y in Y:
                Y_.append([float(y)])   # just make sure predicted Y is nested list of list [[Y_1], [Y_2], ..., [Y_N]]

    except ValueError:
        raise ValueError("You likely tried to predict using a model with (or without) FFT featurization ('--nofft')"
                         " while the model was trained without (or with) FFT featurization. Check the Model_Results.txt"
                         " line 1, if the models were trained using FFT.")

    _ , Names_Of_Mutations, _ = Get_Sequences(Prediction_Set, Mult_Path, Prediction=True)

    predictions = [(Y_[i][0], Names_Of_Mutations[i]) for i in range(len(Y_))]

    # Pay attention if more negative values would define a better variant --> --use negative flag
    predictions.sort()
    predictions.reverse()
    # if predictions array too large?  if Mult_Path is not None: predictions = predictions[:100000]

    # Print FFT-ed and raw sequence vectors for directed evolution if desired
    if print_matrix == True:
        print('X (FFT):\n{} len(X_raw): {}\nX_raw (noFFT):\n{} len(X): {}\n(Predicted value, Variant): {}\n\n'
              .format(X, len(X[0]), X_raw, len(X_raw[0]), predictions))

    return predictions


def Predictions_Out(Predictions, Model, Prediction_Set):
    """
    Writes predictions (of the new sequence space) to text file(s).
    """
    name, value = [], []
    for (val, nam) in Predictions:
        name.append(nam)
        value.append('{:f}'.format(val))

    data = np.array([name, value]).T
    col_width = max(len(str(value)) for row in data for value in row) + 5

    head = ['Name', 'Prediction']
    with open('Predictions_' + str(Model) + '_' + str(Prediction_Set)[:-6] + '.txt', 'w') as f:
        f.write("".join(caption.ljust(col_width) for caption in head) + '\n')
        f.write(len(head)*col_width*'-' + '\n')
        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')


def Plot(Path, Fasta_File, Model, Label, Color, y_WT, noFFT=False):
    """
    Function Plot is used to make plots of the validation process and
    shows predicted (Y_pred) vs. measured/"true" (Y_true) protein fitness and
    calculates the corresponding model performance (R2, (N)RMSE, Pearson's r).
    Also allows colored version plotting to classify predictions in true or
    false positive or negative predictions.
    """
    aaidx = Full_Path(str(Model) + '.txt')
    xy = XY(aaidx, Fasta_File, Prediction=False)
    X, Y_true, X_raw = xy.Get_X_And_Y()

    try:
        file = open(os.path.join(Path, 'Pickles/'+str(Model)), 'rb')
        mod = pickle.load(file)
        file.close()

        Y_pred = []

        try:
            if noFFT is False:
                    Y_pred_ = mod.predict(X)
            else:
                Y_pred_ = mod.predict(X_raw)
        except ValueError:
            raise ValueError("You likely tried to plot a validation set with (or without) FFT featurization ('--nofft')"
                             " while the model was the model was trained without (or with) FFT featurization. Check the"
                             " Model_Results.txt line 1, if the models were trained using FFT.")

        for y_p in Y_pred_:
            Y_pred.append(float(y_p))

        _, Names_Of_Mutations, _ = Get_Sequences(Fasta_File)

        R2 = r2_score(Y_true, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        stddev = np.std(Y_true, ddof=1)
        nrmse = rmse / stddev
        pearson_r = np.corrcoef(Y_true, Y_pred)[0][1]
        legend = '$R^2$ = ' + str(round(R2, 3)) + '\nRMSE = ' + str(round(rmse, 3)) +\
                 '\nNRMSE = ' + str(round(nrmse, 3)) + '\nPearson\'s $r$ = ' + str(round(pearson_r, 3))
        x = np.linspace(min(Y_pred) - 1, max(Y_pred) + 1, 100)

        fig, ax = plt.subplots()
        ax.scatter(Y_true, Y_pred, label=legend, marker='o', s=20, linewidths=0.5, edgecolor='black')
        ax.plot(x, x, color='black', linewidth=0.5)  # plot diagonal line

        if Label is not False:
            from adjustText import adjust_text
            texts = [ax.text(Y_true[i], Y_pred[i], txt, fontsize=4) for i, txt in enumerate(Names_Of_Mutations)]
            adjust_text(texts, only_move={'points': 'y', 'text': 'y'}, force_points=0.5)

        if Color is not False:
            try:
                y_WT = float(y_WT)
            except TypeError:
                raise TypeError('Needs label value of WT (y_WT) when making color plot (e.g. --color --ywt 1.0)')
            if y_WT == 0:
                y_WT = 1E-9  # choose a value close to zero
            true_v, true_p, false_v, false_p = [], [], [], []
            for i, v in enumerate(Y_true):
                if Y_true[i] / y_WT >= 1 and Y_pred[i] / y_WT >= 1:
                    true_v.append(Y_true[i]), true_p.append(float(Y_pred[i]))
                elif Y_true[i] / y_WT < 1 and Y_pred[i] / y_WT < 1:
                    true_v.append(Y_true[i]), true_p.append(float(Y_pred[i]))
                else:
                    false_v.append(Y_true[i]), false_p.append(float(Y_pred[i]))
            try:
                ax.scatter(true_v, true_p, color='tab:blue', marker='o', s=20, linewidths=0.5, edgecolor='black')
            except IndexError:
                pass
            try:
                ax.scatter(false_v, false_p, color='tab:red', marker='o', s=20, linewidths=0.5, edgecolor='black')
            except IndexError:
                pass

            if (y_WT - min(Y_true)) < (max(Y_true) - y_WT):
                limit_Y_true = float(max(Y_true) - y_WT)
            else:
                limit_Y_true = float(y_WT - min(Y_true))
            limit_Y_true = limit_Y_true * 1.1

            if (y_WT - min(Y_pred)) < (max(Y_pred) - y_WT):
                limit_Y_pred = float(max(Y_pred) - y_WT)
            else:
                limit_Y_pred = float(y_WT - min(Y_pred))
            limit_Y_pred = limit_Y_pred * 1.1

            plt.vlines(x=(y_WT + limit_Y_true) - (((y_WT + limit_Y_true) - (y_WT - limit_Y_true)) / 2),
                       ymin=y_WT - limit_Y_pred, ymax=y_WT + limit_Y_pred, color='grey', linewidth=0.5)

            plt.hlines(y=(y_WT + limit_Y_pred) - (((y_WT + limit_Y_pred) - (y_WT - limit_Y_pred)) / 2),
                       xmin=y_WT - limit_Y_true, xmax=y_WT + limit_Y_true, color='grey', linewidth=0.5)

            crossline = np.linspace(y_WT - limit_Y_true, y_WT + limit_Y_true)
            plt.plot(crossline, crossline, color='black', linewidth=0.5)

            steps = float(abs(max(Y_pred)))
            gradient = []
            for x in np.linspace(0, steps, 100):
                # arr = np.linspace(x/steps, 1-x/steps, steps)
                arr = 1 - np.linspace(x / steps, 1 - x / steps, 100)
                gradient.append(arr)
            gradient = np.array(gradient)

            plt.imshow(gradient, extent=[y_WT - limit_Y_true, y_WT + limit_Y_true,
                                         y_WT - limit_Y_pred, y_WT + limit_Y_pred],
                       aspect='auto', alpha=0.8, cmap='coolwarm')  # RdYlGn
            plt.xlim([y_WT - limit_Y_true, y_WT + limit_Y_true])
            plt.ylim([y_WT - limit_Y_pred, y_WT + limit_Y_pred])

        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.legend(prop={'size': 8})

        plt.savefig(str(Model) + '_' + str(Fasta_File[:-6]) + '.png', dpi=500)
    except FileNotFoundError:
        raise FileNotFoundError("Did not find specified model: {}. You can define the threshold of models to be saved;"
                                " e.g. with pypef.py run -l LS.fasta -v VS.fasta -t 10.".format(str(Model)))

    return ()


def mutate_sequence(seq, m, Model, prev_mut_loc, AAs, Sub_LS, iteration, counter, usecsv, csvaa):
    """
    produces a mutant sequence (integer representation), given an initial sequence
    and the number of mutations to introduce ("m") for in silico directed evolution
    """
    try:
        os.mkdir('EvoTraj')
    except FileExistsError:
        pass
    myfile = 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iteration+1) + '.fasta'
    var_seq_list = []
    with open(myfile, 'w') as mf:
        for i in range(m):  # iterate through number of mutations to add
            rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)  # find random position to mutate
            while (rand_loc <= 0) or (rand_loc >= len(seq)):
                rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)

            if usecsv is True:  # Only perform directed evolution on positional csv variant data
                pos_list = []
                aa_list = []
                for aa_positions in Sub_LS:
                    for pos in aa_positions:
                        pos_int = int(pos[1:-1])
                        if pos_int not in pos_list:
                            pos_list.append(pos_int)
                        if csvaa is True:
                            new_aa = str(pos[-1:])
                            if new_aa not in aa_list:
                                aa_list.append(new_aa)
                            AAs = aa_list
                # Select closest Position to single AA positions
                absolute_difference_function = lambda list_value: abs(list_value - rand_loc)
                closest_loc = min(pos_list, key=absolute_difference_function)
                rand_loc = closest_loc - 1   # - 1 as Position 17 is 16 when starting with 0 index
            rand_aa = random.choice(AAs)  # find random amino acid to mutate to
            sequence = seq
            seq_ = list(sequence)
            seq_[rand_loc] = rand_aa  # update sequence to have new amino acid at randomly chosen position
            seq_ = ''.join(seq_)
            var = str(rand_loc+1)+str(rand_aa)
            var_seq_list.append([var, seq_])

            print('> {}{}'.format(rand_loc+1, rand_aa), file=mf)
            print(''.join(seq_), file=mf)

    return dict(var_seq_list)   # chose dict as one can easily change to more sequences to predict per iteration


def restructure_dict(prediction_dict):
    """
    Exchange key and value of a dictionary
    """
    restructured = []
    for pred in prediction_dict:
        restruct = []
        for p in pred:
            restruct.insert(0, p)
        restructured.append(restruct)
    structured_dict = dict(restructured)
    return structured_dict


def write_MCMC_predictions(Model, iter, predictions, counter):
    """
    write predictions to EvoTraj folder to .fasta files for each iteration of evolution
    """
    with open('EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iter+1)
              + '.fasta', 'r') as f_in:
        with open('EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iter+1)
                  + '_prediction.fasta', 'w') as f_out:
            for line in f_in:
                f_out.write(line)
                if '>' in line:
                    key = line[2:].strip()
                    f_out.writelines('; ' + str(predictions.get(key)) + '\n')
    return ()


def in_silico_de(s_WT, num_iterations, Model, amino_acids, T, Path, Sub_LS, counter,
                 noFFT=False, negative=False, usecsv=False, csvaa=False, print_matrix=False):
    """
    Perform directed evolution by randomly selecting a sequence position for substitution and randomly choose the
    amino acid to substitute to. New sequence gets accepted if meeting the Metropolis criterion and will be
    taken for new substitution iteration.
    Metropolis-Hastings-driven directed evolution, similar to Biswas et al.:
    Low-N protein engineering with data-efficient deep learning,
    see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
    """
    v_traj = []  # initialize an array to keep records of the variant names for this trajectory
    y_traj = []  # initialize an array to keep records of the fitness scores for this trajectory
    s_traj = []  # initialize an array to keep records of the protein sequences for this trajectory

    # iterate through the trial mutation steps for the directed evolution trajectory
    for i in range(num_iterations):  # num_iterations

        if i == 0:
            # randomly choose the location of the first mutation in the trajectory
            mut_loc_seed = random.randint(0, len(s_WT))
            # m = 1 instead of (np.random.poisson(2) + 1)
            var_seq_dict = mutate_sequence(s_WT, 1, Model, mut_loc_seed, amino_acids, Sub_LS, 0, counter, usecsv, csvaa)

            predictions = Predict(Path, 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_'
                                  + str(i+1) + '.fasta', Model, None, noFFT, print_matrix)
            predictions = restructure_dict(predictions)

            write_MCMC_predictions(Model, i, predictions, counter)

            ys, variants = [], []
            for var in predictions:
                variants.append(var)
                ys.append(predictions.get(var))


            y, var = ys[0], variants[0]  # only one entry anyway
            new_mut_loc = int(var[:-1]) - 1
            sequence = var_seq_dict.get(var)

            v_traj.append(var)
            y_traj.append(y)
            s_traj.append(sequence)

        else:
            # only chose 1 mutation to introduce and not:
            # mu = np.random.uniform(1, 2.5) --> Number of Mutations = m = np.random.poisson(mu - 1) + 1
            new_var_seq_dict = mutate_sequence(sequence, 1, Model, new_mut_loc, amino_acids,
                                               Sub_LS, i, counter, usecsv, csvaa)

            predictions = Predict(Path, 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_'
                                  + str(i+1) + '.fasta', Model, None, noFFT, print_matrix)
            predictions = restructure_dict(predictions)

            write_MCMC_predictions(Model, i, predictions, counter)

            new_ys, new_variants = [], []
            for var in predictions:
                new_variants.append(var)
                new_ys.append(predictions.get(var))

            new_y, new_y_var = new_ys[0], new_variants[0]
            new_mut_loc = int(new_y_var[:-1]) - 1
            new_sequence = new_var_seq_dict.get(new_y_var)

            # probability function for trial sequence
            # The lower the fitness (y) of the new variant, the higher are the chances to get excluded
            with warnings.catch_warnings():  # catching Overflow warning
                warnings.simplefilter("ignore")
                try:
                    boltz = np.exp(((new_y - y) / T), dtype=np.longfloat)
                    if negative is True:
                        boltz = np.exp(((-new_y - -y) / T), dtype=np.longfloat)
                except OverflowError:
                        boltz = 1
            p = min(1, boltz)
            rand_var = random.random()  # random float between 0 and 1
            if rand_var < p:  # Metropolis-Hastings update selection criterion
                # print('Updated sequence as: Rand ({}) < Boltz ({})'.format(str(rand_var), str(boltz)))
                # print(str(new_mut_loc + 1) + " " + sequence[new_mut_loc] + "->" + new_sequence[new_mut_loc])
                var, y, sequence = new_y_var, new_y, new_sequence  # if criteria is met, update sequence and
                                                                   # corresponding fitness
                v_traj.append(var)  # update the variant naming trajectory records for this iteration of mutagenesis
                y_traj.append(y)  # update the fitness trajectory records for this iteration of mutagenesis
                s_traj.append(sequence)  # update the sequence trajectory records for this iteration of mutagenesis

    return v_traj, s_traj, y_traj


def run_DE_trajectories(s_wt, Model, y_WT, num_iterations, num_trajectories, DE_record_folder, amino_acids, T, Path,
                        Sub_LS, noFFT=False, negative=False, save=False, usecsv=False, csvaa=False, print_matrix=False):
    """
    Runs the directed evolution by adressing the in_silico_de function and plots the evolution trajectories.
    """
    v_records = []  # initialize list of sequence variant names
    s_records = []  # initialize list of sequence records
    y_records = []  # initialize list of fitness score records


    for i in range(num_trajectories):  #iterate through however many mutation trajectories we want to sample
        # call the directed evolution function, outputting the trajectory sequence and fitness score records
        v_traj, s_traj, y_traj = in_silico_de(s_wt, num_iterations, Model, amino_acids, T, Path, Sub_LS, i,
                                              noFFT, negative, usecsv, csvaa, print_matrix)

        v_records.append(v_traj)    # update the variant naming trajectory records for this full mutagenesis trajectory
        s_records.append(s_traj)  # update the sequence trajectory records for this full mutagenesis trajectory
        y_records.append(y_traj)  # update the fitness trajectory records for this full mutagenesis trajectory

        if save==True:
            try:
                os.mkdir(DE_record_folder)
            except FileExistsError:
                pass
            # save sequence records for trajectory i
            np.savetxt(DE_record_folder + "/" + str(Model) + "_trajectory" + str(i+1)
                       + "_seqs.txt", np.array(s_traj), fmt="%s")
            # save fitness records for trajectory i
            np.savetxt(DE_record_folder + "/" + str(Model) + "_trajectory" + str(i+1)
                       + "_fitness.txt", np.array(y_traj))

    # numpy warning filter needed for arraying ragged nested sequences
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    v_records = np.array(v_records)
    s_records = np.array(s_records)
    y_records = np.array(y_records)

    fig, ax = plt.subplots()  # figsize=(10, 6)
    ax.locator_params(integer=True)
    f_len_max = 0
    for j, f in enumerate(y_records):
        if y_WT is not None:
            f_len = len(f)
            if f_len > f_len_max:
                f_len_max = f_len
            ax.plot(np.arange(1, len(f)+2, 1), np.insert(f, 0, y_WT),
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(j+1))
        else:
            ax.plot(np.arange(1, len(f)+1, 1), f,
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(j+1))

    label_x_y_name = []
    for k, l in enumerate(v_records):
        for kk, ll in enumerate(l):  # kk = 1, 2, 3, ...  (=x); ll = variant name; y_records[k][kk] = fitness (=y)
            if y_WT is not None:     # kk+2 as enumerate starts with 0 and WT is 1 --> start labeling with 2
                label_x_y_name.append(ax.text(kk+2, y_records[k][kk], ll))
            else:
                label_x_y_name.append(ax.text(kk+1, y_records[k][kk], ll))

    from adjustText import adjust_text
    adjust_text(label_x_y_name, only_move={'points': 'y', 'text': 'y'}, force_points=0.5)
    leg = ax.legend()
    if y_WT is not None:
        wt_tick = (['', 'WT'] + ((np.arange(1, f_len_max+1, 1)).tolist()))
        with warnings.catch_warnings():  # catching matplotlib 3.3.1 UserWarning
            warnings.simplefilter("ignore")
            ax.set_xticklabels(wt_tick)

    plt.ylabel('Predicted Fitness')
    plt.xlabel('Mutation Trial Steps')
    plt.savefig(str(Model) + '_DE_trajectories.png')

    return s_records, y_records
