#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode


# Docstring essential for docopt arguments
"""
PyPEF - Pythonic Protein Engineering Framework.

Creation of Learning and Validation sets: To split .CSV data in Learning and Validation sets run
    pypef.py mklsvs [...]
Creation of Prediction sets: To create Prediction sets from .CSV data single point mutational variants run
    pypef.py mkps [...]
Running:
 1. To train and validate models run
        pypef.py run -l Learning_Set.fasta -v Validation_Set.fasta [-s 5] [--parallel] [-c 4]
    ! Attention using ray for parallel computing ('--parallel') in Windows: ray is not yet fully supported for Windows !
 2. To plot the validation creating a figure (.png) run
        pypef.py run -m MODEL12345 -f Validation_Set.fasta
 3. To predict variants run
        pypef.py run -m MODEL12345 -p Prediction_Set.fasta
    or for predicting variants in created prediction set folders exemplary run
        pypef.py run -m MODEL12345 --pmult [--drecomb] [...] [--qdiverse]
    or for performing in silico directed evolution run:
        pypef.py directevo -m MODEL12345 [...]


Usage:
    pypef.py mklsvs [--wtseq WT_SEQ] [--input CSV_FILE] [--drop THRESHOLD] [--nornd NUMBER]
    pypef.py mkps [--wtseq WT_SEQ] [--input CSV_FILE] [--drop THRESHOLD]
                                 [--drecomb] [--trecomb] [--qrecomb]
                                 [--ddiverse] [--tdiverse] [--qdiverse]
    pypef.py run --ls LEARNING_SET --vs VALIDATION_SET [--save NUMBER] [--regressor TYPE] [--nofft] [--all]
                                                       [--sort METRIC] [--parallel] [--cores NUMCORES]
    pypef.py --show [MODELS]
    pypef.py run --model MODEL12345 --figure VS_FOR_PLOTTING  [--label] [--color] [--ywt WT_FITNESS] [--nofft]
    pypef.py run --model MODEL12345 --ps PREDICTION_SET [--nofft] [--negative] [--print]
    pypef.py run --model MODEL12345 --pmult [--drecomb] [--trecomb] [--qrecomb]
                                          [--ddiverse] [--tdiverse] [--qdiverse] [--nofft] [--negative]
    pypef.py directevo --model MODEL12345 [--ywt WT_FITNESS] [--wtseq WT_SEQ]
                                        [--numiter NUM_ITER] [--numtraj NUM_TRAJ]
                                        [--temp TEMPERATURE] [--nofft] [--negative] [--print]
                                        [--usecsv] [--csvaa] [--input CSV_FILE] [--drop THRESHOLD]


Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --show                       Show saved models in Model_Results.txt.
  MODELS                       Number of saved models to show [default: 5].
  -w --wtseq WT_SEQ            Input file (in .fa format) for Wild-Type sequence [default: None].
  -i --input CSV_FILE          Input data file in .csv format [default: None].
  -d --drop THRESHOLD          Below threshold variants will be discarded from the data [default: -9E09].
  -n --nornd NUMBER            Number of randomly created Learning and Validation datasets [default: 0].
  -s --save NUMBER             Number of Models to be saved as pickle files [default: 5].
  --parallel                   Parallel computing of training and validation of models [default: False].
  -c --cores NUMCORES          Number of cores used in parallel computing.
  --drecomb                    Create/predict double recombinants [default: False].
  --trecomb                    Create/predict triple recombinants [default: False].
  --qrecomb                    Create/predict quadruple recombinants [default: False].
  --ddiverse                   Create/predict double natural diverse variants [default: False].
  --tdiverse                   Create/predict triple natural diverse variants [default: False].
  --qdiverse                   Create quadruple natural diverse variants [default: False].
  -u --pmult                   Predict for all Prediction files in folder for recomb
                               or diverse variants [default: False].
  --negative                   Set if more negative values define better variants [default: False].
  -l --ls LEARNING_SET         Input Learning set in .fasta format.
  -v --vs VALIDATION_SET       Input Validation set in .fasta format.
  --regressor TYPE             Type of regression (R.) to use, options: PLS R.: pls, PLS CV R.: pls_cv,
                               Random Forest R.: rf, SVM R.: svr, MLP R.: mlp [default: pls].
  --nofft                      Raw sequence input, i.e. no FFT for establishing protein spectra
                               as vector inputs [default: False].
  --all                        Finally training on all data [default: False]
  --sort METRIC                Rank models based on metric {1: R^2, 2: RMSE, 3: NRMSE, 4: Pearson's r,
                               5: Spearman's rank} [default: 1].
  -m --model MODEL12345        Model (pickle file) for plotting of Validation or for performing predictions.
  -f --figure VS_FOR_PLOTTING  Validation set for plotting using a trained Model.
  --label                      Label the plot instances [default: False].
  --color                      Color the plot for "true" and "false" predictions quarters [default: False].
  -p --ps PREDICTION_SET       Prediction set for performing predictions using a trained Model.
  --print                      Print raw encoded and FFT-ed sequence matrices and predictions in Shell [default: False].
  -y --ywt WT_FITNESS          Fitness value (y) of wild-type.
  --numiter NUM_ITER           Number of mutation iterations per evolution trajectory [default: 5].
  --numtraj NUM_TRAJ           Number of trajectories, i.e. evolution pathways [default: 5].
  --temp TEMPERATURE           "Temperature" of Metropolis-Hastings criterion [default: 0.01]
  --usecsv                     Perform directed evolution on single variant csv position data [default: False].
  --csvaa                      Directed evolution csv amino acid substitutions,
                               requires flag "--usecsv" [default: False].
"""

# standard import, for all required modules see requirements.txt file(s)
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
from docopt import docopt
import multiprocessing
# import Modules_Parallelization.R2_List_Parallel locally to avoid error
# when not running in parallel, thus commented out:
# from Modules_Parallelization import R2_List_Parallel
# ray imported later locally as is is only used for parallelized running, thus commented out:
# import ray

# importing own modules
from Modules_Regression import read_models, Formatted_Output, R2_List, Save_Model, Predict, Predictions_Out, Plot
from Modules_Directed_Evolution import run_DE_trajectories
from Modules_Learning_Validation import (get_wt_sequence, csv_input, drop_rows, get_variants, make_sub_LS_VS,
                                         make_sub_LS_VS_randomly, make_fasta_LS_VS)
from Modules_Prediction import (Make_Combinations_Double, Make_Combinations_Triple, Make_Combinations_Quadruple,
                                create_split_files, Make_Combinations_Double_All_Diverse,
                                Make_Combinations_Triple_All_Diverse, Make_Combinations_Quadruple_All_Diverse)

def run():
    """
    Running the program, importing all required self-made modules and
    running them dependent on user-passed input arguments using docopt
    for argument parsing.
    """
    arguments = docopt(__doc__, version='PyPEF 0.1.2 (December 2020)')
    # print(arguments)  # uncomment for printing parsed arguments
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    if arguments['--show']:
        if arguments['MODELS'] != str(5):
            try:
                print(read_models(int(arguments['MODELS'])))
            except ValueError:
                print(read_models(5))
            except TypeError:
                print(read_models(5))
        else:
            print(read_models(5))

    if arguments['mklsvs']:
        WT_Sequence = get_wt_sequence(arguments['--wtseq'])
        csv_file = csv_input(arguments['--input'])
        t_drop = float(arguments['--drop'])

        print('Length of provided sequence: {} amino acids.'.format(len(WT_Sequence)))
        df = drop_rows(csv_file, amino_acids, t_drop)
        no_rnd = arguments['--nornd']

        single_variants, single_values, higher_variants, higher_values = get_variants(df, amino_acids, WT_Sequence)
        print('Number of single variants: {}.'.format(len(single_variants)))
        if len(single_variants) == 0:
            print('Found NO single substitution variants for possible recombination!')
        Sub_LS, Val_LS, Sub_VS, Val_VS = make_sub_LS_VS(single_variants, single_values, higher_variants, higher_values)
        print('Tip: You can edit your LS and VS datasets just by cutting/pasting between the LS and VS fasta datasets.')

        print('Creating LS dataset...', end='\r')
        make_fasta_LS_VS('LS.fasta', WT_Sequence, Sub_LS, Val_LS)
        print('Creating VS dataset...', end='\r')
        make_fasta_LS_VS('VS.fasta', WT_Sequence, Sub_VS, Val_VS)

        try:
            no_rnd = int(no_rnd)
        except ValueError:
            no_rnd = 0
        if no_rnd != 0:
            random_set_counter = 1
            no_rnd = int(no_rnd)
            while random_set_counter <= no_rnd:
                print('Creating random LV and VS No. {}...'.format(random_set_counter), end='\r')
                Sub_LS, Val_LS, Sub_VS, Val_VS = make_sub_LS_VS_randomly(single_variants, single_values,
                                                                         higher_variants, higher_values
                                                                         )
                make_fasta_LS_VS('LS_random_' + str(random_set_counter) + '.fasta', WT_Sequence, Sub_LS, Val_LS)
                make_fasta_LS_VS('VS_random_' + str(random_set_counter) + '.fasta', WT_Sequence, Sub_VS, Val_VS)
                random_set_counter += 1
        print('\n\nDone!\n')

    elif arguments['mkps']:
        WT_Sequence = get_wt_sequence(arguments['--wtseq'])
        csv_file = csv_input(arguments['--input'])
        t_drop = float(arguments['--drop'])

        df = drop_rows(csv_file, amino_acids, t_drop)
        print('Length of provided sequence: {} amino acids.'.format(len(WT_Sequence)))
        single_variants, _, higher_variants, _ = get_variants(df, amino_acids, WT_Sequence)
        print('Number of single variants: {}.'.format(len(single_variants)))
        no_done = False
        if len(single_variants) == 0:
            print('Found NO single substitution variants for possible recombination! '
                  'No prediction files can be created!')
            no_done = True

        if arguments['--drecomb']:
            print('Creating Recomb_Double_Split...')
            for no, files in enumerate(Make_Combinations_Double(single_variants)):
                Double_Mutants = np.array(files)
                create_split_files(Double_Mutants, single_variants, WT_Sequence, 'Recomb_Double', no)

        if arguments['--trecomb']:
            print('Creating Recomb_Triple_Split...')
            for no, files in enumerate(Make_Combinations_Triple(single_variants)):
                Triple_Mutants = np.array(files)
                create_split_files(Triple_Mutants, single_variants, WT_Sequence, 'Recomb_Triple', no)

        if arguments['--qrecomb']:
            print('Creating Recomb_Quadruple_Split...')
            for no, files in enumerate(Make_Combinations_Quadruple(single_variants)):
                Quadruple_Mutants = np.array(files)
                create_split_files(Quadruple_Mutants, single_variants, WT_Sequence, 'Recomb_Quadruple', no)

        if arguments['--ddiverse']:
            print('Creating Diverse_Double_Split...')
            for no, files in enumerate(Make_Combinations_Double_All_Diverse(single_variants, amino_acids)):
                Doubles = np.array(files)
                create_split_files(Doubles, single_variants, WT_Sequence, 'Diverse_Double', no + 1)

        if arguments['--tdiverse']:
            print('Creating Diverse_Triple_Split...')
            for no, files in enumerate(Make_Combinations_Triple_All_Diverse(single_variants, amino_acids)):
                Triples = np.array(files)
                create_split_files(Triples, single_variants, WT_Sequence, 'Diverse_Triple', no + 1)

        if arguments['--qdiverse']:
            print('Creating Diverse_Quadruple_Split...')
            for no, files in enumerate(Make_Combinations_Quadruple_All_Diverse(single_variants, amino_acids)):
                Quadruples = np.array(files)
                create_split_files(Quadruples, single_variants, WT_Sequence, 'Diverse_Quadruple', no + 1)

        if arguments['--drecomb'] is False and arguments['--trecomb'] is False \
                and arguments['--qrecomb'] is False and arguments['--ddiverse'] is False \
                and arguments['--tdiverse'] is False and arguments['--qdiverse'] is False:
            print('\nInput Error:\nAt least one specification needed: Specify recombinations for mkps ; '
                  'e.g. try: "pypef.py mkps --drecomb" for performing double recombinant Prediction set.\n')
            no_done = True

        if no_done is False:
            print('\nDone!\n')

    elif arguments['run']:
        if arguments['--ls'] is not None and arguments['--vs'] is not None:
            if arguments['--model'] is None and arguments['--figure'] is None:
                Path = os.getcwd()
                try:
                    t_save = int(arguments['--save'])
                except ValueError:
                    t_save = 5
                if arguments['--parallel']:
                    # Parallelization of AAindex iteration
                    # import parallel modules here as ray is yet not supported for Windows
                    import ray
                    ray.init()
                    from Modules_Parallelization import R2_List_Parallel
                    Cores = arguments['--cores']
                    try:
                        Cores = int(Cores)
                    except (ValueError, TypeError):
                        try:
                            Cores = multiprocessing.cpu_count() // 2
                        except NotImplementedError:
                            Cores = 4
                    print('Using {} cores for parallel computing. Running...'.format(Cores))
                    AAindex_R2_List = R2_List_Parallel(arguments['--ls'], arguments['--vs'], Cores,
                                                       arguments['--regressor'], arguments['--nofft'],
                                                       arguments['--sort']
                                                       )
                    Formatted_Output(AAindex_R2_List, arguments['--nofft'])
                    Save_Model(Path, AAindex_R2_List, arguments['--ls'], arguments['--vs'], t_save,
                               arguments['--regressor'], arguments['--nofft'], arguments['--all']
                               )

                else:
                    AAindex_R2_List = R2_List(arguments['--ls'], arguments['--vs'], arguments['--regressor'],
                                              arguments['--nofft'], arguments['--sort']
                                              )
                    Formatted_Output(AAindex_R2_List, arguments['--nofft'])
                    Save_Model(Path, AAindex_R2_List, arguments['--ls'], arguments['--vs'], t_save,
                               arguments['--regressor'], arguments['--nofft'], arguments['--all']
                               )
                print('\nDone!\n')

        elif arguments['--figure'] is not None and arguments['--model'] is not None:
            Path = os.getcwd()
            Plot(Path, arguments['--figure'], arguments['--model'], arguments['--label'], arguments['--color'],
                 arguments['--ywt'], arguments['--nofft']
                 )
            print('\nCreated plot!\n')

        # Prediction of single .fasta file
        elif arguments['--ps'] is not None and arguments['--model'] is not None:
            Path = os.getcwd()
            print(arguments['--nofft'])
            predictions = Predict(Path, arguments['--ps'], arguments['--model'], None, arguments['--nofft'])
            if arguments['--negative']:
                predictions = sorted(predictions, key=lambda x: x[0], reverse=False)
            Predictions_Out(predictions, arguments['--model'], arguments['--ps'])
            print('\nDone!\n')

        # Prediction on recombinant/diverse variant folder data
        elif arguments['--pmult'] and arguments['--model'] is not None:
            Path = os.getcwd()
            recombs_total = []
            recomb_d, recomb_t, recomb_q = '/Recomb_Double_Split/', '/Recomb_Triple_Split/', '/Recomb_Quadruple_Split/'
            diverse_d, diverse_t, diverse_q = '/Diverse_Double_Split/', '/Diverse_Triple_Split/', \
                                              '/Diverse_Quadruple_Split/'
            if arguments['--drecomb']:
                recombs_total.append(recomb_d)
            if arguments['--trecomb']:
                recombs_total.append(recomb_t)
            if arguments['--qrecomb']:
                recombs_total.append(recomb_q)
            if arguments['--ddiverse']:
                recombs_total.append(diverse_d)
            if arguments['--tdiverse']:
                recombs_total.append(diverse_t)
            if arguments['--qdiverse']:
                recombs_total.append(diverse_q)
            if arguments['--drecomb'] is False and arguments['--trecomb'] is False and arguments['--qrecomb'] is False \
                    and arguments['--ddiverse'] is False and arguments['--tdiverse'] is False \
                    and arguments['--qdiverse'] is False:
                print('Define prediction target for --pmult, e.g. --pmult --drecomb.')

            for args in recombs_total:
                predictions_total = []
                print('Running predictions for files in {}...'.format(args[1:-1]))
                Path_recomb = Path + args
                os.chdir(Path)
                files = [f for f in listdir(Path_recomb) if isfile(join(Path_recomb, f)) if f.endswith('.fasta')]
                for f in tqdm(files):
                    predictions = Predict(Path, f, arguments['--model'], Path_recomb, arguments['--nofft'])
                    for pred in predictions:
                        predictions_total.append(pred)  # perhaps implement numpy.save if array gets too large byte size
                predictions_total = list(dict.fromkeys(predictions_total))  # removing duplicates from list
                if arguments['--negative']:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=False)

                else:
                    predictions_total = sorted(predictions_total, key=lambda x: x[0], reverse=True)

                Predictions_Out(predictions_total, arguments['--model'], 'Top' + args[1:-1])
                os.chdir(Path)
            print('\nDone!\n')

    # Metropolis-Hastings-driven directed evolution, similar to Biswas et al.:
    # Low-N protein engineering with data-efficient deep learning,
    # see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
    elif arguments['directevo']:
        if arguments['--model'] is not None:
            Path = os.getcwd()
            try:
                # "temperature" parameter: determines sensitivity of Metropolis-Hastings acceptance criteria
                T = float(arguments['--temp'])
                # how many subsequent mutation trials per simulated evolution trajectory
                num_iterations = int(arguments['--numiter'])
                # how many separate evolution trajectories to run
                num_trajectories = int(arguments['--numtraj'])
            except ValueError:
                raise ValueError("Define 'numiter' and 'numtraj' as integer and 'temp' as float.")

            args_model = arguments['--model']
            s_WT = get_wt_sequence(arguments['--wtseq'])
            y_WT = arguments['--ywt']
            negative = arguments['--negative']

            # Metropolis-Hastings-driven directed evolution on single mutant position csv data
            usecsv = arguments['--usecsv']
            if usecsv is True:
                csv_file = csv_input(arguments['--input'])
                t_drop = float(arguments['--drop'])

                print('Length of provided sequence: {} amino acids.'.format(len(s_WT)))
                df = drop_rows(csv_file, amino_acids, t_drop)

                single_variants, single_values, higher_variants, higher_values = get_variants(df, amino_acids, s_WT)
                print('Number of single variants: {}.'.format(len(single_variants)))
                if len(single_variants) == 0:
                    print('Found NO single substitution variants for possible recombination!')
                Sub_LS, Val_LS, _, _ = make_sub_LS_VS(single_variants, single_values, higher_variants,
                                                      higher_values, directed_evolution=True
                                                      )
                print('Creating single variant dataset...')

                make_fasta_LS_VS('Single_variants.fasta', s_WT, Sub_LS, Val_LS)

            else:
                Sub_LS = None

            # Metropolis-Hastings-driven directed evolution on single mutant .csv amino acid substitution data
            csvaa = arguments['--csvaa']
            traj_records_folder = 'DE_record'

            print('Running evolution trajectories and plotting...')

            run_DE_trajectories(s_WT, args_model, y_WT, num_iterations, num_trajectories,
                                traj_records_folder, amino_acids, T, Path, Sub_LS, arguments['--nofft'],
                                negative=negative, save=True, usecsv=usecsv, csvaa=csvaa,
                                print_matrix=arguments['--print']
                                )
            print('\nDone!')


if __name__ == "__main__":
    run()
