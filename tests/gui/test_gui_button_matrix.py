"""
Click every button of the real GUI window under every relevant combination of
GUI options (PLM drop-down, LoRA/GP check boxes, regression model, split method,
train set proportion) without running any job.

For each click the test checks
  * which files are asked for (no missing and no superfluous file prompts),
  * that the command handed to the job thread is accepted by the real CLI parser
    (docopt usage + schema validation of pypef.main), i.e. it would not fail on
    start-up with a usage error,
  * that the parsed arguments carry the selected files and options,
  * the button state (all disabled while a job is "running", all enabled again
    after a refused or cancelled action),
  * and that cancelling any of the file prompts aborts the action.
"""
import itertools
import os

import pytest

pytestmark = pytest.mark.pip_specific

PLM_LABELS = (
    'None', 'ESM', 'ProSST', 'PMPNN', 'ESM+ProSST',
    'ESM+PMPNN', 'ProSST+PMPNN', 'ESM+ProSST+PMPNN',
)
REGRESSORS = ('PLS', 'PLS_LOOCV', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'RF', 'MLP')
SPLITS = {  # box_mklsts_cv item -> CLI flag
    'None': None, 'Random split': '--random', 'Modulo split': '--modulo',
    'Continuous split': '--cont', 'Plot distributions': '--plot',
}
FILES = {
    'wt': '/data/wt.fasta', 'csv': '/data/variants.csv', 'pdb': '/data/wt.pdb',
    'msa': '/data/msa.a2m', 'ls': '/data/LS.fasl', 'ts': '/data/TS.fasl',
    'ps': '/data/PS.fasta', 'params': '/data/Pickles/GREMLIN',
    'hybrid_model': '/data/Pickles/HYBRIDGREMLIN', 'ml_model': '/data/Pickles/MLGREMLIN',
    'onehot_model': '/data/Pickles/ONEHOT',
}

# Buttons independent of the PLM selection: button -> (CLI command, {CLI option: file role}).
DCA_ML_BUTTONS = {
    'button_mklsts': ('mklsts', {'--wt': 'wt', '--input': 'csv'}),
    'button_mkps': ('mkps', {'--wt': 'wt', '--input': 'csv'}),
    'button_gremlin_ssm': ('predict_ssm', {'--wt': 'wt', '--params': 'params'}),
    'button_dca_inference_gremlin': ('param_inference', {'--wt': 'wt', '--msa': 'msa'}),
    'button_dca_test_dca': ('hybrid', {'--ts': 'ts', '--model': 'params', '--params': 'params'}),
    'button_dca_predict_dca': ('hybrid', {'--ps': 'ps', '--model': 'params', '--params': 'params'}),
    'button_hybrid_train_dca': (
        'hybrid', {'--ls': 'ls', '--ts': 'ls', '--model': 'params', '--params': 'params'}),
    'button_hybrid_train_test_dca': (
        'hybrid', {'--ls': 'ls', '--ts': 'ts', '--model': 'params', '--params': 'params'}),
    'button_hybrid_test_dca': (
        'hybrid', {'--ts': 'ts', '--model': 'hybrid_model', '--params': 'params'}),
    'button_hybrid_predict_dca': (
        'hybrid', {'--ps': 'ps', '--model': 'hybrid_model', '--params': 'params'}),
    'button_supervised_train_dca': ('ml', {'--ls': 'ls', '--ts': 'ls', '--params': 'params'}),
    'button_supervised_train_test_dca': ('ml', {'--ls': 'ls', '--ts': 'ts', '--params': 'params'}),
    'button_supervised_test_dca': (
        'ml', {'--ts': 'ts', '--model': 'ml_model', '--params': 'params'}),
    'button_supervised_predict_dca': (
        'ml', {'--ps': 'ps', '--model': 'ml_model', '--params': 'params'}),
    'button_supervised_train_onehot': ('ml', {'--ls': 'ls', '--ts': 'ls'}),
    'button_supervised_train_test_onehot': ('ml', {'--ls': 'ls', '--ts': 'ts'}),
    'button_supervised_test_onehot': ('ml', {'--ts': 'ts', '--model': 'onehot_model'}),
    'button_supervised_predict_onehot': ('ml', {'--ps': 'ps', '--model': 'onehot_model'}),
}
ML_TRAIN_BUTTONS = [b for b in DCA_ML_BUTTONS if b.startswith('button_supervised_train')]

# PLM-dependent buttons: button -> (CLI command, {CLI option: file role},
#                                   PLM selections that start a job, is training)
_ALL_PLMS = {p.lower() for p in PLM_LABELS[1:]}
PLM_BUTTONS = {
    'button_llm_ssm': ('predict_ssm', {'--wt': 'wt'}, {'esm', 'prosst'}, False),
    'button_llm_test_zs': ('hybrid', {'--ts': 'ts', '--wt': 'wt'},
                           {'esm', 'prosst', 'pmpnn'}, False),
    'button_llm_predict_zs': ('hybrid', {'--ps': 'ps', '--wt': 'wt'},
                              {'esm', 'prosst', 'pmpnn'}, False),
    'button_hybrid_train_dca_llm': (
        'hybrid', {'--ls': 'ls', '--ts': 'ls', '--params': 'params'}, _ALL_PLMS, True),
    'button_hybrid_train_test_dca_llm': (
        'hybrid', {'--ls': 'ls', '--ts': 'ts', '--params': 'params'}, _ALL_PLMS, True),
    'button_hybrid_test_dca_llm': (
        'hybrid', {'--ts': 'ts', '--model': 'hybrid_model', '--params': 'params'}, _ALL_PLMS, False),
    'button_hybrid_predict_dca_llm': (
        'hybrid', {'--ps': 'ps', '--model': 'hybrid_model', '--params': 'params'}, _ALL_PLMS, False),
}
TRAINING_OPTIONS = list(itertools.product([False, True], repeat=3))  # lora, gauss_opt, gauss_comb


def parse_cli(cmd):
    """Parse a GUI command exactly like run_main() does before starting any work."""
    from docopt import docopt
    from pypef.main import __doc__ as usage, validate
    try:
        return validate(docopt(usage, argv=cmd))
    except SystemExit as e:  # DocoptExit / schema exit
        pytest.fail(f'GUI command is rejected by the CLI parser: {cmd!r}\n{e}')


@pytest.fixture
def launches(gui_window, monkeypatch):
    """Record commands instead of starting job threads."""
    commands = []
    def record():
        # A string would be split on whitespace by docopt (breaking paths with spaces)
        assert isinstance(gui_window.cmd, list), f'Command is not an argument list: {gui_window.cmd!r}'
        assert all(isinstance(a, str) for a in gui_window.cmd), gui_window.cmd
        commands.append(gui_window.cmd)
    monkeypatch.setattr(gui_window, 'start_main_thread', record)
    return commands


def select(window, plm='ESM', regressor='PLS', split='None', proportion=80,
           lora=False, gauss_opt=False, gauss_comb=False):
    window.box_llm.setCurrentText(plm)
    window.box_regression_model.setCurrentText(regressor)
    window.box_mklsts_cv.setCurrentText(split)
    window.slider.setValue(proportion)
    window.check_lora.setChecked(lora)
    window.check_gauss_opt.setChecked(gauss_opt)
    window.check_gauss_comb.setChecked(gauss_comb)
    assert window.box_llm.currentText() == plm
    assert window.box_regression_model.currentText() == regressor
    assert window.box_mklsts_cv.currentText() == split


def click(window, gui, launches, name, dialogs):
    button = getattr(window, name)
    button.click()
    assert window.target_button is button, f'{name} handler sets a wrong target_button'
    if launches:
        assert len(launches) == 1
        assert not any(gui.buttons_enabled(window)), 'Buttons must be disabled while a job runs'
        window.end_process()  # what the finished worker would trigger
        return parse_cli(launches[0])
    assert all(gui.buttons_enabled(window)), 'Buttons must be re-enabled after a refused action'
    return None


def assert_files(args, mapping):
    for option, role in mapping.items():
        assert args[option] == FILES[role], f'{option}: expected the {role} file'


def test_every_button_is_covered(gui_window, gui):
    names = {gui.button_name(gui_window, b) for b in gui_window.all_buttons}
    untested = names - set(DCA_ML_BUTTONS) - set(PLM_BUTTONS) - {'button_help', 'button_work_dir'}
    assert not untested, f'Add the new GUI button(s) to this test module: {untested}'
    assert [gui_window.box_llm.itemText(i) for i in range(gui_window.box_llm.count())] \
        == list(PLM_LABELS)
    assert [gui_window.box_regression_model.itemText(i)
            for i in range(gui_window.box_regression_model.count())] == list(REGRESSORS)
    assert [gui_window.box_mklsts_cv.itemText(i)
            for i in range(gui_window.box_mklsts_cv.count())] == list(SPLITS)


@pytest.mark.parametrize('name', [b for b in DCA_ML_BUTTONS if b not in ML_TRAIN_BUTTONS])
@pytest.mark.parametrize('plm', ['None', 'ESM+ProSST+PMPNN'])
def test_dca_and_util_buttons(gui_window, gui, launches, file_dialogs, name, plm):
    # The PLM selection and training check boxes must not leak into these commands.
    select(gui_window, plm=plm, lora=True, gauss_opt=True, gauss_comb=True)
    dialogs = file_dialogs(FILES)
    command, mapping = DCA_ML_BUTTONS[name]
    args = click(gui_window, gui, launches, name, dialogs)
    assert args is not None, f'{name} did not start a job'
    assert args[command]
    assert set(dialogs.prompted) == set(mapping.values())
    assert_files(args, mapping)
    for flag in ('--plm', '--llm', '--pdb'):
        assert not args.get(flag), f'{name} must not pass {flag}'
    for flag in ('--lora', '--gauss_opt', '--gauss_comb'):
        assert not args.get(flag), f'{name} must not pass {flag}'
    if command == 'ml':
        assert args['--encoding'] == ('onehot' if name.endswith('onehot') else 'dca')
        assert args['--threads'] == gui_window.n_cores


@pytest.mark.parametrize('name', ML_TRAIN_BUTTONS)
@pytest.mark.parametrize('regressor', REGRESSORS)
def test_supervised_training_regressors(gui_window, gui, launches, file_dialogs, name, regressor):
    from pypef.ml.regression import cv_regression_options
    select(gui_window, regressor=regressor)
    dialogs = file_dialogs(FILES)
    command, mapping = DCA_ML_BUTTONS[name]
    args = click(gui_window, gui, launches, name, dialogs)
    assert set(dialogs.prompted) == set(mapping.values())
    assert_files(args, mapping)
    assert args['--regressor'] == regressor.lower()
    if args['--regressor'] != 'pls_loocv':  # handled separately in regression.py
        assert cv_regression_options(args['--regressor']) is not None


def plm_cases():
    for name, (_cmd, _mapping, _supported, training) in PLM_BUTTONS.items():
        options = TRAINING_OPTIONS if training else [(True, True, True)]
        for plm, opts in itertools.product(PLM_LABELS, options):
            yield pytest.param(name, plm, opts, id=f'{name}-{plm}-lora{opts[0]:d}'
                               f'-gp{opts[1]:d}-comb{opts[2]:d}')


@pytest.mark.parametrize('name, plm, options', list(plm_cases()))
def test_plm_buttons(gui_window, gui, launches, file_dialogs, name, plm, options):
    lora, gauss_opt, gauss_comb = options
    select(gui_window, plm=plm, lora=lora, gauss_opt=gauss_opt, gauss_comb=gauss_comb)
    dialogs = file_dialogs(FILES)
    command, mapping, supported, training = PLM_BUTTONS[name]
    args = click(gui_window, gui, launches, name, dialogs)
    if plm.lower() not in supported:
        assert args is None, f'{name} should refuse PLM option {plm}'
        assert 'Provide a PLM option' in gui_window.logTextBox.widget.toPlainText()
        return
    assert args is not None, f'{name} did not start a job for {plm}'
    assert args[command]
    assert args['--plm'] == plm.lower()
    assert_files(args, mapping)
    needs_structure = 'prosst' in plm.lower() or 'pmpnn' in plm.lower() or (training and gauss_opt)
    expected_roles = set(mapping.values()) | ({'wt', 'pdb'} if needs_structure else set())
    assert set(dialogs.prompted) == expected_roles
    if needs_structure:
        assert args['--wt'] == FILES['wt'] and args['--pdb'] == FILES['pdb']
    else:
        assert not args['--pdb']
    # LoRA/GP options only apply to training and must follow the check boxes there.
    assert args['--lora'] == (training and lora)
    assert args['--gauss_opt'] == (training and gauss_opt)
    assert args['--gauss_comb'] == (training and gauss_comb)


@pytest.mark.parametrize('split', list(SPLITS))
@pytest.mark.parametrize('proportion', [0, 50, 80, 100])
@pytest.mark.parametrize('columns', [False, True])
def test_mklsts_options(gui_window, gui, launches, file_dialogs, split, proportion, columns):
    select(gui_window, split=split, proportion=proportion)
    keys = {'--variant_key': 'mutant', '--sequence_key': 'mutated_sequence',
            '--fitness_key': 'DMS_score'}
    fields = (gui_window.input_variant_key, gui_window.input_sequence_key,
              gui_window.input_fitness_key)
    for field, value in zip(fields, keys.values()):
        field.setText(f'  {value} ' if columns else '')
    assert gui_window.slider_text.text() == f'Train set proportion: {proportion / 100}'
    args = click(gui_window, gui, launches, 'button_mklsts', file_dialogs(FILES))
    assert args['--ls_proportion'] == proportion / 100
    for flag in SPLITS.values():
        if flag:
            assert args[flag] == (flag == SPLITS[split])
    for flag, value in keys.items():
        assert args[flag] == (value if columns else None)


def cancel_cases():
    for name in DCA_ML_BUTTONS:
        yield pytest.param(name, 'ESM', id=name)
    for name, (_cmd, _mapping, supported, _training) in PLM_BUTTONS.items():
        for plm in PLM_LABELS:
            if plm.lower() in supported:
                yield pytest.param(name, plm, id=f'{name}-{plm}')


@pytest.mark.parametrize('name, plm', list(cancel_cases()))
def test_cancel_each_file_prompt(gui_window, gui, launches, file_dialogs, name, plm):
    select(gui_window, plm=plm, gauss_opt=True)
    # Normal path first, to count the prompts; then cancel each one in turn.
    dialogs = file_dialogs(FILES)
    click(gui_window, gui, launches, name, dialogs)
    n_prompts = len(dialogs.prompted)
    assert n_prompts > 0
    for cancel_at in range(n_prompts):
        launches.clear()
        gui_window.logTextBox.widget.clear()
        dialogs = file_dialogs(FILES, cancel_at=cancel_at)
        assert click(gui_window, gui, launches, name, dialogs) is None
        assert len(dialogs.prompted) == cancel_at + 1, 'No further prompt after a cancel'


def test_help_button(gui_window, gui):
    from pypef.main import __doc__ as usage
    gui_window.button_help.click()
    assert usage.strip() in gui_window.logTextBox.widget.toPlainText()
    assert '--help' in gui_window.textedit_out.toPlainText()
    assert all(gui.buttons_enabled(gui_window))


def test_working_directory_button(gui_window, tmp_path, monkeypatch):
    from pypef.gui import qt_window
    monkeypatch.chdir(tmp_path)  # restores the original cwd afterwards
    target = tmp_path / 'out dir'
    target.mkdir()
    monkeypatch.setattr(qt_window.QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(target))
    gui_window.button_work_dir.click()
    assert os.path.samefile(os.getcwd(), target)
    assert gui_window.working_directory_text.text() == os.getcwd()


def test_stop_button_without_running_job(gui_window):
    gui_window.button_abort.click()
    gui_window.abort_workers()  # direct call: exceptions in slots are only printed
    assert 'No running job to stop.' in gui_window.logTextBox.widget.toPlainText()


def test_progress_labels_until_last_epoch(gui_window):
    gui_window._train_start_time = None
    for epoch in (1, 2, 3):
        for batch in (1, 2):
            gui_window._last_eta_update = 0
            gui_window.on_train_progress_step({
                'epoch': epoch, 'batch': batch, 'loss': 0.1,
                'epoch_total': 3, 'batch_total': 2,
            })
    assert gui_window.epoch_progress_bar.value() == 100
    assert gui_window.batch_progress_bar.value() == 100
    assert gui_window.epoch_time_label.text().startswith('Epoch 3 / 3  (100%)')
    assert gui_window.batch_time_label.text().startswith('Batch 2 / 2  (100%)')
    gui_window.end_process()
    assert gui_window.epoch_progress_bar.value() == 0


def test_file_paths_with_spaces(gui_window, gui, launches, file_dialogs):
    """Commands are argument lists, so a path with spaces stays one argument."""
    spaced = {role: path.replace('/data/', '/my data/') for role, path in FILES.items()}
    broken = []
    select(gui_window, plm='ProSST')
    for name in list(DCA_ML_BUTTONS) + list(PLM_BUTTONS):
        launches.clear()
        dialogs = file_dialogs(spaced)
        getattr(gui_window, name).click()
        gui_window.end_process()
        assert launches, f'{name} did not start a job'
        cmd = launches[0]
        if isinstance(cmd, str):  # docopt would split a string on whitespace
            broken.append(name)
            continue
        args = parse_cli(cmd)
        if not all(spaced[role] in args.values() for role in set(dialogs.prompted)):
            broken.append(name)
    assert not broken, f'Buttons splitting paths with spaces: {broken}'
