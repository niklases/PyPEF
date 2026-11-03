"""
End-to-end GUI runs: click the real buttons and let the real PyPEF jobs run in
the GUI's worker thread on the small ANEH dataset (datasets/ANEH), then check
that each job succeeded and wrote its output files.

Two tiers:
  * pip_specific (default, CPU, ~1 min): set creation, purely supervised
    (one-hot and DCA encoding) and hybrid DCA train/test/predict as well as DCA
    zero-shot test/predict. The GREMLIN parameters used by these steps are
    inferred once with a reduced number of iterations for speed.
  * gui_slow (opt-in, long-running, PLM models are loaded/downloaded; a GPU
    is recommended): GREMLIN inference and SSM buttons as well as every
    PLM button (zero-shot, SSM, DCA+PLM hybrid incl. LoRA/GP options):

        python -m pytest tests/gui/test_gui_end_to_end.py -m gui_slow -v
        python -m pytest tests/gui/test_gui_end_to_end.py -m gui_slow -k "PMPNN"
"""
import os
import shutil
from pathlib import Path

import pytest

ANEH = Path(__file__).resolve().parents[2] / 'datasets' / 'ANEH'
WT = ANEH / 'Sequence_WT_ANEH.fasta'
CSV = ANEH / '37_ANEH_variants.csv'
MSA = ANEH / 'ANEH_jhmmer.a2m'
PDB = ANEH / 'AF-Q9UR30-F1-model_v4.pdb'

# step -> (button, prerequisite steps, files the step must create)
STEPS = {
    'mklsts': ('button_mklsts', (), ('LS.fasl', 'TS.fasl')),
    'mkps': ('button_mkps', (), ('37_ANEH_variants_prediction_set.fasta',)),
    'onehot_train_test': ('button_supervised_train_test_onehot', ('mklsts',), ('Pickles/ONEHOT',)),
    'onehot_train': ('button_supervised_train_onehot', ('mklsts',), ('Pickles/ONEHOT',)),
    'onehot_test': ('button_supervised_test_onehot', ('onehot_train_test',), ()),
    'onehot_predict': ('button_supervised_predict_onehot', ('onehot_train_test', 'mkps'), ()),
    'dca_ml_train_test': ('button_supervised_train_test_dca', ('mklsts',), ('Pickles/MLGREMLIN',)),
    'dca_ml_train': ('button_supervised_train_dca', ('mklsts',), ('Pickles/MLGREMLIN',)),
    'dca_ml_test': ('button_supervised_test_dca', ('dca_ml_train_test',), ()),
    'dca_ml_predict': ('button_supervised_predict_dca', ('dca_ml_train_test', 'mkps'), ()),
    'hybrid_train_test': ('button_hybrid_train_test_dca', ('mklsts',), ('Pickles/HYBRIDGREMLIN',)),
    'hybrid_train': ('button_hybrid_train_dca', ('mklsts',), ('Pickles/HYBRIDGREMLIN',)),
    'hybrid_test': ('button_hybrid_test_dca', ('hybrid_train_test',), ()),
    'hybrid_predict': ('button_hybrid_predict_dca', ('hybrid_train_test', 'mkps'), ()),
    'dca_zero_shot_test': ('button_dca_test_dca', ('mklsts',), ()),
    'dca_zero_shot_predict': ('button_dca_predict_dca', ('mkps',), ()),
}


class E2E:
    """A working directory with the GREMLIN parameters and finished steps."""
    def __init__(self, workdir, qapp, gui):
        self.dir = workdir
        self.qapp = qapp
        self.gui = gui
        self.done = set()
        self.files = {
            'wt': WT, 'csv': CSV, 'msa': MSA, 'pdb': PDB,
            'ls': workdir / 'LS.fasl', 'ts': workdir / 'TS.fasl',
            'ps': workdir / '37_ANEH_variants_prediction_set.fasta',
            'params': workdir / 'Pickles' / 'GREMLIN',
            'hybrid_model': workdir / 'Pickles' / 'HYBRIDGREMLIN',
            'ml_model': workdir / 'Pickles' / 'MLGREMLIN',
            'onehot_model': workdir / 'Pickles' / 'ONEHOT',
        }

    def run(self, window, file_dialogs, button_name, timeout=900):
        file_dialogs(self.files)
        log, failed = self.gui.run_job(window, getattr(window, button_name), timeout)
        assert not failed, f'{button_name} failed:\n{log[-3000:]}'
        assert 'Done!' in log, f'{button_name} did not finish:\n{log[-3000:]}'
        return log

    def step(self, window, file_dialogs, name):
        button_name, requires, outputs = STEPS[name]
        for requirement in requires:
            if requirement not in self.done:
                self.step(window, file_dialogs, requirement)
        for output in outputs:  # make sure the files are really created by this step
            path = self.dir / output
            if path.exists():
                path.unlink()
        log = self.run(window, file_dialogs, button_name)
        for output in outputs:
            assert (self.dir / output).is_file(), f'{name} did not create {output}'
        self.done.add(name)
        return log


@pytest.fixture(scope='module')
def e2e_module(tmp_path_factory, qapp, request):
    from pypef.main import run_main
    workdir = tmp_path_factory.mktemp('gui e2e')  # space: paths must stay one argument
    old_cwd = os.getcwd()
    failed_before = request.session.testsfailed
    os.chdir(workdir)  # the GUI writes all outputs to the current directory
    try:
        # Few GREMLIN iterations: the parameters only need to be valid, not good.
        run_main(argv=['param_inference', '--wt', str(WT), '--msa', str(MSA), '--opt_iter', '2'])
        yield workdir
    finally:
        os.chdir(old_cwd)
        remove_unless_failed(request, workdir, failed_before)


def remove_unless_failed(request, path, failed_before):
    """
    Delete a test working directory (GREMLIN pickles alone take ~1.5 GB, and
    /tmp is often a size-limited tmpfs), but keep it for inspection if a test
    failed meanwhile.
    """
    if request.session.testsfailed == failed_before:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def e2e(e2e_module, qapp, gui, request):
    state = request.module.__dict__.setdefault('_e2e_state', E2E(e2e_module, qapp, gui))
    os.chdir(e2e_module)
    return state


@pytest.mark.pip_specific
@pytest.mark.parametrize('step', list(STEPS))
def test_gui_step(gui_window, file_dialogs, e2e, step):
    log = e2e.step(gui_window, file_dialogs, step)
    if step.endswith('_predict'):
        assert 'Predictions' in log or 'prediction' in log.lower()


@pytest.mark.pip_specific
@pytest.mark.parametrize('regressor', ['Ridge', 'Lasso', 'ElasticNet', 'PLS_LOOCV'])
def test_gui_regressor_choice(gui_window, file_dialogs, e2e, regressor):
    gui_window.box_regression_model.setCurrentText(regressor)
    e2e.step(gui_window, file_dialogs, 'onehot_train_test')
    assert f'ONEHOT_{regressor.upper()}' in ' '.join(os.listdir(e2e.dir))


@pytest.mark.pip_specific
@pytest.mark.parametrize('split, outputs', [
    ('Random split', [f'{s}_random_{k}.fasl' for s in ('LS', 'TS') for k in range(1, 6)]),
    ('Modulo split', [f'{s}_modulo_{k}.fasl' for s in ('LS', 'TS') for k in range(1, 6)]),
    ('Continuous split', [f'{s}_continuous_{k}.fasl' for s in ('LS', 'TS') for k in range(1, 6)]),
    ('Plot distributions', ['CV_split_pos_aa_distr.png']),
])
def test_gui_mklsts_splits(gui_window, file_dialogs, e2e, split, outputs):
    for output in outputs:
        (e2e.dir / output).unlink(missing_ok=True)
    gui_window.box_mklsts_cv.setCurrentText(split)
    e2e.run(gui_window, file_dialogs, 'button_mklsts')
    missing = [o for o in outputs if not (e2e.dir / o).is_file()]
    assert not missing, f'{split} did not create {missing}'


# Long-running tier ##########################################################
SLOW_DCA_PLM = [  # (PLM drop-down, LoRA, GP optimization, combined GP)
    ('ESM', False, False, False),
    ('ESM', True, False, False),
    ('ESM', False, True, False),
    ('ProSST', False, False, False),
    ('ProSST', True, False, False),
    ('PMPNN', False, False, False),
    ('PMPNN', False, True, False),
    ('ESM+ProSST', False, True, True),
    ('ESM+PMPNN', False, False, False),
    ('ProSST+PMPNN', False, False, False),
    ('ProSST+PMPNN', False, True, False),
    ('ESM+ProSST+PMPNN', False, False, False),
]


@pytest.fixture
def slow_e2e(e2e, gui_window, file_dialogs):
    e2e.step(gui_window, file_dialogs, 'mklsts')
    e2e.step(gui_window, file_dialogs, 'mkps')
    return e2e


@pytest.mark.gui_slow
def test_slow_gremlin_inference_and_ssm(gui_window, file_dialogs, slow_e2e, tmp_path):
    # Own directory: the full GREMLIN run must not replace the shared parameters.
    shutil.copy(slow_e2e.files['ls'], tmp_path)
    os.chdir(tmp_path)
    files = dict(slow_e2e.files, params=tmp_path / 'Pickles' / 'GREMLIN')
    file_dialogs(files)
    for name in ('button_dca_inference_gremlin', 'button_gremlin_ssm'):
        log, failed = slow_e2e.gui.run_job(gui_window, getattr(gui_window, name), 3600)
        assert not failed and 'Done!' in log, log[-3000:]
    assert (tmp_path / 'SSM_landscape.csv').is_file()
    os.chdir(slow_e2e.dir)
    shutil.rmtree(tmp_path, ignore_errors=True)  # only reached if the test passed


@pytest.mark.gui_slow
@pytest.mark.parametrize('plm', ['ESM', 'ProSST', 'PMPNN'])
def test_slow_plm_zero_shot(gui_window, file_dialogs, slow_e2e, plm):
    gui_window.box_llm.setCurrentText(plm)
    slow_e2e.run(gui_window, file_dialogs, 'button_llm_test_zs', 3600)
    slow_e2e.run(gui_window, file_dialogs, 'button_llm_predict_zs', 3600)


@pytest.mark.gui_slow
@pytest.mark.parametrize('plm', ['ESM', 'ProSST', 'PMPNN'])
def test_slow_plm_ssm(gui_window, file_dialogs, slow_e2e, plm):
    gui_window.box_llm.setCurrentText(plm)
    slow_e2e.run(gui_window, file_dialogs, 'button_llm_ssm', 3600)


@pytest.mark.gui_slow
@pytest.mark.parametrize('plm, lora, gauss_opt, gauss_comb', SLOW_DCA_PLM,
                         ids=['-'.join([p] + [n for n, on in zip(('LoRA', 'GP', 'CombGP'), o) if on])
                              for p, *o in SLOW_DCA_PLM])
def test_slow_dca_plm_hybrid(gui_window, file_dialogs, slow_e2e, plm, lora, gauss_opt, gauss_comb):
    gui_window.box_llm.setCurrentText(plm)
    gui_window.check_lora.setChecked(lora)
    gui_window.check_gauss_opt.setChecked(gauss_opt)
    gui_window.check_gauss_comb.setChecked(gauss_comb)
    pickles = slow_e2e.dir / 'Pickles'
    for old in pickles.glob('HYBRID*'):
        old.unlink()
    slow_e2e.run(gui_window, file_dialogs, 'button_hybrid_train_test_dca_llm', 7200)
    # The model file name carries the PLM keys, e.g. Pickles/HYBRIDGREMLINESM
    models = list(pickles.glob('HYBRID*'))
    assert len(models) == 1, f'Expected one trained hybrid model, found {models}'
    files = dict(slow_e2e.files, hybrid_model=models[0])
    file_dialogs(files)
    for name in ('button_hybrid_test_dca_llm', 'button_hybrid_predict_dca_llm'):
        log, failed = slow_e2e.gui.run_job(gui_window, getattr(gui_window, name), 3600)
        assert not failed and 'Done!' in log, f'{name} failed:\n{log[-3000:]}'
