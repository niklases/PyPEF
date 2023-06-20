### based on the two datasets provided (ANEH and avGFP)
Write-Host "For successful running, following files are required:`n`nin test_dataset_aneh/`n`tSequence_WT_ANEH.fasta`n`t37_ANEH_variants.csv`n`tANEH_jhmmer.a2m`n`tANEH_72.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/workflow/test_dataset_aneh/ANEH_72.6.params)`n`nin test_dataset_avgfp/`n`tP42212_F64L.fasta`n`tavGFP.csv`n`turef100_avgfp_jhmmer_119.a2m`n`turef100_avgfp_jhmmer_119_plmc_42.6.params (generated using PLMC or dowloaded from https://github.com/niklases/PyPEF/blob/main/workflow/test_dataset_avgfp/uref100_avgfp_jhmmer_119_plmc_42.6.params)`n`n"

Set-PSDebug -Trace 1  # Write-Host on
$ErrorActionPreference = "Stop"  # exit on (PyPEF) errors
$PSDefaultParameterValues = @{
    'Write-Debug:Separator' = " (Line $($MyInvocation.ScriptLineNumber)): "
}

### RUN ME WITH
### $ ./run_cli_tests.sh                      # printing STDOUT and STDERR to terminal
### $ ./run_cli_tests.sh &> test_cli_run.log  # writing STDOUT and STDERR to log file

### if using downloaded/locally stored pypef .py files:
############### CHANGE THIS PATHS AND USED THREADS, REQUIRES PYTHON ENVIRONMENT WITH PRE-INSTALLED MODULES ###############
$env:PYTHONPATH="C:\path\to\pypef-main"                                                                                  #
function pypef { python C:\path\to\pypef-main\pypef\main.py @args }                                                      #
##########################################################################################################################
### else just use pip-installed pypef version (uncomment):                                                               #
#function pypef { pypef @args }                                                                                          #
##########################################################################################################################
$threads = 1                                                                                                             #
##########################################################################################################################

### threads=1 shows progress bar where possible
### CV-based mlp and rf regression option optimization take a long time and related testing commands are commented out/not included herein

### Pure ML (and some hybrid model) tests on ANEH dataset
cd 'test_dataset_aneh'
#######################################################################
Write-Host

pypef --version
Write-Host
pypef -h
Write-Host
pypef mklsts -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
Write-Host

pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls
Write-Host
pypef ml --show
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor pls_loocv
Write-Host
pypef ml --show
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor ridge
Write-Host
pypef ml --show
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor lasso
Write-Host
pypef ml --show
Write-Host
pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor elasticnet
Write-Host
pypef ml --show
Write-Host
#pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor mlp
#pypef ml --show
#pypef ml -e onehot -l LS.fasl -t TS.fasl --regressor rf
#pypef ml --show

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --threads $threads
Write-Host
pypef ml --show
Write-Host

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls_loocv --nofft --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor ridge --nofft --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor lasso --nofft --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor elasticnet --nofft --threads $threads
Write-Host
pypef ml --show
Write-Host

pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml --show
Write-Host

pypef param_inference --msa ANEH_jhmmer.a2m --opt_iter 100
Write-Host
pypef save_msa_info --msa ANEH_jhmmer.a2m -w Sequence_WT_ANEH.fasta --opt_iter 100
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls_loocv --params GREMLIN
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor ridge --params GREMLIN
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor lasso --params GREMLIN
Write-Host
pypef ml --show
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor elasticnet --params GREMLIN
Write-Host
pypef ml --show
Write-Host

pypef ml -e aaidx -m FAUJ880104 -t TS.fasl
Write-Host
pypef ml -e onehot -m ONEHOT -t TS.fasl
Write-Host
pypef ml -e dca -m MLplmc -t TS.fasl --params ANEH_72.6.params --threads $threads 
Write-Host
pypef ml -e aaidx -m FAUJ880104 -t TS.fasl --label
Write-Host
pypef ml -e onehot -m ONEHOT -t TS.fasl --label
Write-Host
pypef ml -e dca -m MLplmc -t TS.fasl --label --params ANEH_72.6.params --threads $threads
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
Write-Host
pypef ml -e aaidx -m FAUJ880104 -p 37_ANEH_variants_prediction_set.fasta
Write-Host
pypef ml -e onehot -m ONEHOT -p 37_ANEH_variants_prediction_set.fasta
Write-Host
pypef ml -e dca -m MLplmc -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
Write-Host
pypef ml -e dca -m MLgremlin -p 37_ANEH_variants_prediction_set.fasta --params GREMLIN
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --drecomb --trecomb --qarecomb --qirecomb --ddiverse
Write-Host
pypef ml -e aaidx -m FAUJ880104 --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
Write-Host
pypef ml -e onehot -m ONEHOT --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
Write-Host
pypef ml -e dca -m MLplmc --params ANEH_72.6.params --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse --threads $threads
Write-Host
pypef ml -e dca -m MLgremlin --params GREMLIN --pmult --drecomb --trecomb --qarecomb --qirecomb --ddiverse
Write-Host

pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative
Write-Host
pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative
Write-Host
pypef ml -e dca directevo -m MLplmc -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative --params ANEH_72.6.params
Write-Host
pypef ml -e dca directevo -m MLgremlin -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative --params GREMLIN
Write-Host
pypef ml -e aaidx directevo -m FAUJ880104 -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative
Write-Host
pypef ml -e onehot directevo -m ONEHOT -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative
Write-Host
pypef ml -e dca directevo -m MLplmc -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
Write-Host
pypef ml -e dca directevo -m MLgremlin -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative --params GREMLIN
Write-Host
pypef ml -e aaidx directevo -m FAUJ880104 -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative
Write-Host
pypef ml -e onehot directevo -m ONEHOT -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative
Write-Host
pypef ml -e dca directevo -m MLplmc -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
Write-Host
pypef ml -e dca directevo -m MLgremlin -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative --params GREMLIN
Write-Host

pypef ml -e aaidx -l LS.fasl -t TS.fasl --regressor pls --nofft
Write-Host
pypef ml --show
Write-Host
pypef ml -e aaidx directevo -m WEBA780101 -w Sequence_WT_ANEH.fasta -y -1.5 --negative --nofft
Write-Host

pypef encode -i 37_ANEH_variants.csv -e aaidx -m FAUJ880104 -w Sequence_WT_ANEH.fasta
Write-Host
pypef encode -i 37_ANEH_variants.csv -e onehot -w Sequence_WT_ANEH.fasta
Write-Host
pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
Write-Host
mv 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_plmc_dca_encoded.csv
Write-Host
pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params GREMLIN
Write-Host
mv 37_ANEH_variants_dca_encoded.csv 37_ANEH_variants_gremlin_dca_encoded.csv
Write-Host

pypef ml low_n -i 37_ANEH_variants_aaidx_encoded.csv
Write-Host
pypef ml low_n -i 37_ANEH_variants_onehot_encoded.csv
Write-Host
pypef ml low_n -i 37_ANEH_variants_plmc_dca_encoded.csv
Write-Host
pypef ml low_n -i 37_ANEH_variants_gremlin_dca_encoded.csv
Write-Host

pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv
Write-Host

pypef ml extrapolation -i 37_ANEH_variants_aaidx_encoded.csv --conc
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_onehot_encoded.csv --conc
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_plmc_dca_encoded.csv --conc
Write-Host
pypef ml extrapolation -i 37_ANEH_variants_gremlin_dca_encoded.csv --conc
Write-Host

pypef hybrid train_and_save -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --fit_size 0.66 --threads $threads
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params ANEH_72.6.params --threads $threads
Write-Host
pypef hybrid -m HYBRIDplmc -t TS.fasl --params ANEH_72.6.params --threads $threads
Write-Host

pypef hybrid train_and_save -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta --params GREMLIN --fit_size 0.66
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -m HYBRIDgremlin -t TS.fasl --params GREMLIN
Write-Host

pypef mkps -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta
Write-Host
pypef hybrid -m HYBRIDplmc -p 37_ANEH_variants_prediction_set.fasta --params ANEH_72.6.params --threads $threads
Write-Host
pypef hybrid -m HYBRIDplmc --params ANEH_72.6.params --pmult --drecomb --threads $threads
Write-Host

pypef hybrid directevo -m HYBRIDplmc -w Sequence_WT_ANEH.fasta --y_wt -1.5 --negative --params ANEH_72.6.params
Write-Host
pypef hybrid directevo -m HYBRIDplmc -w Sequence_WT_ANEH.fasta -y -1.5 --numiter 10 --numtraj 8 --negative --params ANEH_72.6.params
Write-Host
pypef hybrid directevo -m HYBRIDplmc -i 37_ANEH_variants.csv -w Sequence_WT_ANEH.fasta -y -1.5 --temp 0.1 --usecsv --csvaa --negative --params ANEH_72.6.params
Write-Host

pypef encode -i 37_ANEH_variants.csv -e dca -w Sequence_WT_ANEH.fasta --params ANEH_72.6.params --threads $threads
Write-Host
pypef hybrid low_n -i 37_ANEH_variants_dca_encoded.csv
Write-Host
pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv
Write-Host
pypef hybrid extrapolation -i 37_ANEH_variants_dca_encoded.csv --conc
Write-Host


### Hybrid model (and some pure ML and pure DCA) tests on avGFP dataset 
cd '../test_dataset_avgfp'
#######################################################################
Write-Host

pypef mklsts -i avGFP.csv -w P42212_F64L.fasta
Write-Host
pypef param_inference --msa uref100_avgfp_jhmmer_119.a2m --opt_iter 100
Write-Host
# Check MSA coevolution info
pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
###
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
Write-Host
# Similar to line above
pypef hybrid -t TS.fasl --params GREMLIN
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
Write-Host
# pure statistical
pypef hybrid -t TS.fasl --params GREMLIN
Write-Host


# using .params file
pypef ml -e dca -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
# ML LS/TS
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params GREMLIN
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params uref100_avgfp_jhmmer_119_plmc_42.6.params
Write-Host
# Transforming .params file to DCAEncoding and using DCAEncoding Pickle; output file: Pickles/MLplmc.
# That means using uref100_avgfp_jhmmer_119_plmc_42.6.params or PLMC as params file is identical.
pypef param_inference --params uref100_avgfp_jhmmer_119_plmc_42.6.params
Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --regressor pls --params PLMC --threads $threads
Write-Host
# ml only TS
pypef ml -e dca -m MLplmc -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
pypef ml -e dca -m MLgremlin -t TS.fasl --params GREMLIN --threads $threads
Write-Host


Write-Host
pypef ml -e dca -l LS.fasl -t TS.fasl --params PLMC --threads $threads
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params PLMC --threads $threads
Write-Host
pypef hybrid -m PLMC -t TS.fasl --params PLMC --threads $threads
Write-Host

# pure statistical
pypef hybrid -t TS.fasl --params PLMC --threads $threads
Write-Host
pypef hybrid -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -m GREMLIN -t TS.fasl --params GREMLIN
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host
pypef save_msa_info --msa uref100_avgfp_jhmmer_119.a2m -w P42212_F64L.fasta --opt_iter 100
# train and save only for hybrid
pypef hybrid train_and_save -i avGFP.csv --params GREMLIN --wt P42212_F64L.fasta
Write-Host
# Encode CSV
pypef encode -e dca -i avGFP.csv --wt P42212_F64L.fasta --params GREMLIN
Write-Host
pypef encode --encoding dca -i avGFP.csv --wt P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads 12

#Extrapolation
Write-Host
pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
Write-Host

# Direct Evo
pypef ml -e dca directevo -m MLgremlin --wt P42212_F64L.fasta --params GREMLIN
Write-Host
pypef ml -e dca directevo -m MLplmc --wt P42212_F64L.fasta --params PLMC
Write-Host
pypef hybrid directevo -m GREMLIN --wt P42212_F64L.fasta --params GREMLIN
Write-Host
pypef hybrid directevo -m PLMC --wt P42212_F64L.fasta --params PLMC
Write-Host
pypef hybrid directevo --wt P42212_F64L.fasta --params GREMLIN
Write-Host
pypef hybrid directevo --wt P42212_F64L.fasta --params PLMC
Write-Host

pypef hybrid -l LS.fasl -t TS.fasl --params GREMLIN
Write-Host 
pypef hybrid -m HYBRIDgremlin -t TS.fasl --params GREMLIN
Write-Host 

### Similar to old CLI run test from here

pypef encode -i avGFP.csv -e dca -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
pypef encode -i avGFP.csv -e onehot -w P42212_F64L.fasta
Write-Host
pypef ml -e aaidx -l LS.fasl -t TS.fasl --threads $threads
Write-Host
pypef ml --show
Write-Host
pypef encode -i avGFP.csv -e aaidx -m GEIM800103 -w P42212_F64L.fasta 
Write-Host

pypef hybrid train_and_save -i avGFP.csv --params uref100_avgfp_jhmmer_119_plmc_42.6.params --fit_size 0.66 -w P42212_F64L.fasta --threads $threads
Write-Host
pypef hybrid -l LS.fasl -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
pypef hybrid -m HYBRIDplmc -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host

# No training set given
pypef hybrid -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
pypef ml -e dca -m MLplmc -t TS.fasl --params uref100_avgfp_jhmmer_119_plmc_42.6.params --label --threads $threads
Write-Host

pypef mkps -i avGFP.csv -w P42212_F64L.fasta
Write-Host
pypef hybrid -m HYBRIDplmc -p avGFP_prediction_set.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params --threads $threads
Write-Host
pypef mkps -i avGFP.csv -w P42212_F64L.fasta --drecomb
#pypef hybrid -m HYBRID --params uref100_avgfp_jhmmer_119_plmc_42.6.params --pmult --drecomb --threads $threads  # many single variants for recombination, takes too long
Write-Host

pypef hybrid directevo -m HYBRIDplmc -w P42212_F64L.fasta --params uref100_avgfp_jhmmer_119_plmc_42.6.params
Write-Host
pypef hybrid directevo -m HYBRIDplmc -w P42212_F64L.fasta --numiter 10 --numtraj 8 --params uref100_avgfp_jhmmer_119_plmc_42.6.params
Write-Host
pypef hybrid directevo -m HYBRIDplmc -i avGFP.csv -w P42212_F64L.fasta --temp 0.1 --usecsv --csvaa --params uref100_avgfp_jhmmer_119_plmc_42.6.params

pypef hybrid low_n -i avGFP_dca_encoded.csv
Write-Host
pypef hybrid extrapolation -i avGFP_dca_encoded.csv
Write-Host
pypef hybrid extrapolation -i avGFP_dca_encoded.csv --conc
Write-Host

pypef ml low_n -i avGFP_dca_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_dca_encoded.csv --conc --regressor ridge
Write-Host

pypef ml low_n -i avGFP_onehot_encoded.csv --regressor pls
Write-Host
pypef ml extrapolation -i avGFP_onehot_encoded.csv --regressor pls
Write-Host
pypef ml extrapolation -i avGFP_onehot_encoded.csv --conc --regressor pls
Write-Host

pypef ml low_n -i avGFP_aaidx_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_aaidx_encoded.csv --regressor ridge
Write-Host
pypef ml extrapolation -i avGFP_aaidx_encoded.csv --conc --regressor ridge
Write-Host

Write-Host 'All tests finished without error!'