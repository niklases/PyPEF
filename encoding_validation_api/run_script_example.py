#!/usr/bin/env python3
from exemplary_dataset_B import all_sequences, all_labels
from pypef_api import AaIndexPerformance, SequenceToNum
from pypef_api import pls_cv_regressor, svr_cv_regressor, rf_cv_regressor, mlp_cv_regressor
import pickle
# or just use a BaseEstimator, e.g. PLSRegression()
from sklearn.cross_decomposition import PLSRegression  # needed for model=eval(best_model[-2]), import also for other
                                                       # regression options if needed

# split all data (sequences and corresponding fitness labels), e.g. in 120 entries
# for learning and 30 entries for validation (20% validation set size)
learn_sequences, y_learn = all_sequences[:120], all_labels[:120]
valid_sequences, y_valid = all_sequences[120:], all_labels[120:]

performances, _cv_performances = AaIndexPerformance(        # _cv_performances not used herein
    learn_sequences, y_learn, valid_sequences, y_valid,
    fft=True, tqdm_=True, kfold_cv_on_all=0, regressor=pls_cv_regressor(),  # next to GridSearchCV-based Regression,
    finally_train_on_all=False, save_model=5, sort='1'                       # you can use just a sklearn BaseEstimator
).get_performance()                                                          # , e.g. PLSRegression()

# Sequence to predict
seq_to_predict = [
    'MSAPFAKFPSSASISPNPFTVSIPDEQLDDLKTLVRLSKIAPPTYESLQADGRFGITSEWLTTMREKWLSEFDWRPFEARLNSFPQFTTEIEGLTIHFAALFSEREDAVPIALL'
    'HGWPGSFVEFYPILQLFREEYTPETLPFHLVVPSLPGYTFSSGPPLDKDFGLMDNARVVDQLMKDLGFGSGYIIQGGDIGSFVGRLLGVGFDACKAVHLNFCAMDAPPEGPSIE'
    'SLSAAEKEGIARMEKVMTDGIAYAMEHSTRPSTIGHVLSSSPIALLAWIGEKYLQWVDKPLPSETILEMVSLYWLTESFPRAIHTYRECFPTASAPNGATMLQKELYIHKPFGF'
    'SFFPKDVHPVPRSWIATTGNLVFFRDHAEGGHFAALERPRELKTDLTAFVEQVWQK'
]

# printing top 10 models: used indices for aa encoding, achieved performance values, and model parameters
for i, p in enumerate(performances[:10]):
    print(i+1, p)

# the best model regarding R2 (if AAIndexPerformance(sort='1')) is performances[0]
best_model = performances[0]
print('\nBest model according to R2 on validation set: {}\n'.format(best_model))

# the best model was encoded by amino acid index stored in the last entry of the list: best_model[-1]
use_aaindex = best_model[-1]


######## EXAMPLE 1: Reconstruct best model from performance list ########

# reconstruct model parameters in list entry best_model[-2]
model = eval(best_model[-2])

# get fft-ed or raw-encoded sequences for fitting and prediction, use raw_encoded_num_seq_to_learn if no fft was used
fft_encoded_num_seq_to_learn, raw_encoded_num_seq_to_learn = SequenceToNum(use_aaindex, learn_sequences).get_x_and_y()
fft_encoded_num_seq_to_pred, raw_encoded_num_seq_to_pred = SequenceToNum(use_aaindex, seq_to_predict).get_x_and_y()
# fft_encoded_num_seq_to_valid, raw_encoded_num_seq_to_valid = SequenceToNum(use_aaindex, valid_sequences).get_x_and_y()

# refitting model reconstructed from string in performance list on learning data
model.fit(fft_encoded_num_seq_to_learn, y_learn)

# predict (list of) (unknown) sequence(s) to estimate the fitness
print('Predicted fitness of sequence with reconstructed model: {}\n'.format(model.predict(fft_encoded_num_seq_to_pred)))


######## EXAMPLE 2: Reload saved top model stored in folder Models/ (if not AaIndexPerformance(save_model=0) ########

# get name of AAindex from best model of performance list
aaindex_name = best_model[0]

# load model stored in folder /Models
model = pickle.load(open('Models/' + aaindex_name + '.sav', 'rb'))

# get fft-ed (and raw for AaIndexPerformance(fft=False)) encoded sequence to predict
fft_encoded_num_seq_to_pred, raw_encoded_num_seq_to_pred = SequenceToNum(use_aaindex, seq_to_predict).get_x_and_y()

# Predict (list of) (unknown) sequence(s) to estimate the fitness
print('Predicted fitness of sequence with loaded model (should be the same result, '
      'except when finally trained model on all data): {}\n'.format(model.predict(fft_encoded_num_seq_to_pred)))

print('Done!')
