"""
Evaluate PyPEF supervised (5-CV) ProteinGym results with the official ProteinGym
summary script (proteingym/performance_DMS_supervised_benchmarks.py) and rank them
against the published supervised baselines.

Steps:
  1. Per DMS and CV scheme, compute the Spearman correlation (and MSE) over all pooled
     out-of-fold predictions, as done in ProteinGym's merge_supervised.py.
  2. Add the published per-DMS baseline scores (benchmarks/DMS_supervised/substitutions/
     <metric>/DMS_substitutions_<metric>_DMS_level_<cv_scheme>.csv).
  3. Run the official summary script (aggregation over UniProt ID -> function category ->
     CV scheme) on this long-format table.

The official script needs results for all three CV schemes per DMS, so only DMS with
results for all schemes are evaluated (all models, incl. baselines, on the same subset).
Note: MSE of PyPEF is computed on raw DMS_score values, while ProteinGym uses
normalized targets, so only Spearman is comparable.
"""

import argparse
import os
import subprocess
import sys

import pandas as pd


CV_SCHEMES = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
METRICS = ["Spearman", "MSE"]
FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_pypef_scores(llm, model_name):
    rows = []
    for cv_scheme in CV_SCHEMES:
        folder = os.path.join(
            FILE_PATH, 'model_scores', 'supervised_substitutions',
            cv_scheme, 'pypef_hybrid', llm
        )
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(folder, f))
            rows.append({
                'DMS_id': f[:-len('.csv')],
                'model_name': model_name,
                'fold_variable_name': cv_scheme,
                # Pooled over all folds, like ProteinGym's merge_supervised.py
                'Spearman': df['y'].corr(df['y_pred'], method='spearman'),
                'MSE': ((df['y'] - df['y_pred']) ** 2).mean(),
            })
    return pd.DataFrame(rows)


def get_baseline_scores(proteingym_dir):
    per_metric = []
    for metric in METRICS:
        dfs = []
        for cv_scheme in CV_SCHEMES:
            path = os.path.join(
                proteingym_dir, 'benchmarks', 'DMS_supervised', 'substitutions', metric,
                f'DMS_substitutions_{metric}_DMS_level_{cv_scheme}.csv'
            )
            df = pd.read_csv(path).melt(id_vars='DMS_id', var_name='model_name', value_name=metric)
            df['fold_variable_name'] = cv_scheme
            dfs.append(df)
        per_metric.append(pd.concat(dfs))
    return per_metric[0].merge(per_metric[1], on=['DMS_id', 'model_name', 'fold_variable_name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--proteingym_dir', default=os.path.expanduser('~/dev/ProteinGym'))
    parser.add_argument('--llm', default='prosst+esm+pmpnn')
    parser.add_argument('--model_name', default=None, help='Default: "PyPEF hybrid (<llm>)"')
    parser.add_argument('--output_dir', default=None, help='Default: ./official_performance/<llm>')
    parser.add_argument('--number_assay_reshuffle', type=int, default=1000,
                        help='Bootstrap resamples for standard errors (official default: 10000, slow)')
    parser.add_argument('--baselines_only', action='store_true',
                        help='Sanity check: evaluate only the published baselines on all DMS '
                             '(should reproduce the published summary)')
    args = parser.parse_args()
    model_name = args.model_name or f'PyPEF hybrid ({args.llm})'
    output_dir = os.path.abspath(args.output_dir or os.path.join(
        FILE_PATH, 'official_performance', 'baselines_only' if args.baselines_only else args.llm))
    os.makedirs(output_dir, exist_ok=True)

    ref_df = pd.read_csv(os.path.join(args.proteingym_dir, 'reference_files', 'DMS_substitutions.csv'))
    baselines = get_baseline_scores(args.proteingym_dir)

    if args.baselines_only:
        scores = baselines
        dms_ids = list(ref_df['DMS_id'])
    else:
        pypef = get_pypef_scores(args.llm, model_name)
        unknown = set(pypef['DMS_id']) - set(ref_df['DMS_id'])
        if unknown:
            print(f'WARNING: {len(unknown)} DMS not in reference file, ignored: {sorted(unknown)}')
        n_schemes = pypef.groupby('DMS_id')['fold_variable_name'].nunique()
        dms_ids = [d for d in ref_df['DMS_id'] if n_schemes.get(d, 0) == len(CV_SCHEMES)]
        print(f'{model_name}: DMS per CV scheme: '
              f'{pypef.groupby("fold_variable_name")["DMS_id"].nunique().reindex(CV_SCHEMES).to_dict()}')
        print(f'Evaluating all models on {len(dms_ids)}/{len(ref_df)} DMS with results for all CV schemes.')
        scores = pd.concat([baselines, pypef])
    scores = scores[scores['DMS_id'].isin(dms_ids)]

    ref_path = os.path.join(output_dir, 'DMS_substitutions_subset.csv')
    scores_path = os.path.join(output_dir, 'merged_scores_substitutions_DMS.csv')
    ref_df[ref_df['DMS_id'].isin(dms_ids)].to_csv(ref_path, index=False)
    scores.to_csv(scores_path, index=False)

    subprocess.run([
        sys.executable,
        os.path.join(args.proteingym_dir, 'proteingym', 'performance_DMS_supervised_benchmarks.py'),
        '--input_scoring_file', scores_path,
        '--output_performance_file_folder', output_dir,
        '--DMS_reference_file_path', ref_path,
        '--top_model', 'ProteinNPT',
        '--number_assay_reshuffle', str(args.number_assay_reshuffle),
    ], check=True)

    summary = pd.read_csv(os.path.join(output_dir, 'Spearman', 'Summary_performance_DMS_substitutions_Spearman.csv'))
    cols = ['Model_rank', 'Model_name', 'Average_Spearman', 'Bootstrap_standard_error_Spearman',
            *[f'Average_Spearman_{cv}' for cv in CV_SCHEMES]]
    print(f'\nOfficial ProteinGym aggregation on {len(dms_ids)} DMS:')
    print(summary[cols].to_string(index=False))
    if not args.baselines_only:
        own = scores[scores['model_name'] == model_name]
        print(f'\nPlain means over these {len(dms_ids)} DMS for {model_name} (no UniProt/function aggregation):')
        print(own.groupby('fold_variable_name')['Spearman'].mean().reindex(CV_SCHEMES).round(3).to_string(),
              f'\n-> {own["Spearman"].mean():.3f}')
    print(f'\nOutputs written to {output_dir}')


if __name__ == '__main__':
    main()
