#!/usr/bin/env bash
# Run the test steps of the GitHub Actions "ubuntu" job (.github/workflows/ci.yml)
# locally, for one or more Python versions, each in its own isolated conda env.
#
# Steps (same commands, markers and environment variables as ci.yml):
#   lint   flake8 syntax/undefined-name check of ./pypef
#   main   pytest on the source tree (PYTHONPATH=repo), excluding pip_specific,
#          requires_gpu and wheel_build tests
#   gui    `pip install .[gui]`, then the pip_specific tests from a neutral directory
#          with --import-mode=importlib, i.e. against the *installed* package
#   wheel  wheel_build tests (wheel contents / packaging)
#
# Usage:
#   scripts/CI/run_ci_locally.sh                      # all CI Python versions, all steps
#   scripts/CI/run_ci_locally.sh -p 3.13              # one version
#   scripts/CI/run_ci_locally.sh -p 3.12,3.13 -s gui  # selected versions / steps
#   scripts/CI/run_ci_locally.sh -p 3.13 -s main -- -k hybrid -x  # extra pytest args
#
#   The gui step runs all fast GUI tests (marker pip_specific), as CI does. The
#   long-running GUI end-to-end runs with PLMs (marker gui_slow) never run on CI
#   and are skipped unless selected; a later -m overrides the step's -m, and -g
#   is recommended (hours on CPU):
#   scripts/CI/run_ci_locally.sh -p 3.12,3.13 -s gui -g -- -m "pip_specific or gui_slow"  # all GUI tests
#   scripts/CI/run_ci_locally.sh -p 3.13 -s gui -g -- -m gui_slow -k PMPNN  # some slow tests
#
# Options:
#   -p VERSIONS  comma-separated Python versions (default: the ci.yml matrix)
#   -s STEPS     comma-separated steps: lint,main,gui,wheel (default: all)
#   -f           fresh: recreate the env(s) and reinstall all requirements
#   -g           allow the GPU (CI runners have none, so it is hidden by default)
#   -h           show this help
#
# Environments live in $PYPEF_CI_HOME (default ~/.cache/pypef-ci/envs/py<version>)
# and are reused, as installing requirements.txt (CUDA torch etc.) takes several GB.
# Note: the gui step installs PyPEF non-editable into that env, as CI does.
# Logs: $PYPEF_CI_HOME/logs/<timestamp>/py<version>.log
# HF_TOKEN / HUGGING_FACE_HUB_TOKEN are passed through if set (as CI secrets).
# Requires conda (Miniconda/Anaconda) to provide the Python versions.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_YML="$REPO/.github/workflows/ci.yml"
CI_HOME="${PYPEF_CI_HOME:-$HOME/.cache/pypef-ci}"
ALL_STEPS="lint,main,gui,wheel"

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

default_versions() {
    # Read the python-version matrix of the ubuntu job from ci.yml
    sed -n '/^  ubuntu:/,/^  [a-z]/p' "$CI_YML" | grep -m1 'python-version:' \
        | grep -oE '"[0-9]+\.[0-9]+"' | tr -d '"' | paste -sd, -
}

VERSIONS=""; STEPS="$ALL_STEPS"; FRESH=false; USE_GPU=false
while getopts "p:s:fgh" opt; do
    case $opt in
        p) VERSIONS="$OPTARG" ;;
        s) STEPS="$OPTARG" ;;
        f) FRESH=true ;;
        g) USE_GPU=true ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift
EXTRA_PYTEST_ARGS=("$@")
[[ -z "$VERSIONS" ]] && VERSIONS="$(default_versions)"
[[ -z "$VERSIONS" ]] && { echo "Could not read Python versions from $CI_YML; use -p." >&2; exit 1; }

for step in ${STEPS//,/ }; do
    [[ ",$ALL_STEPS," == *",$step,"* ]] || { echo "Unknown step: $step (use $ALL_STEPS)" >&2; exit 1; }
done
has_step() { [[ ",$STEPS," == *",$1,"* ]]; }

CONDA="$(command -v conda || true)"
[[ -z "$CONDA" && -x "$HOME/miniconda3/bin/conda" ]] && CONDA="$HOME/miniconda3/bin/conda"
[[ -z "$CONDA" ]] && { echo "conda not found (needed to provide the Python versions)." >&2; exit 1; }

# Environment of the ci.yml workflow and ubuntu job
export CI=true PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=":4096:8" TQDM_DISABLE=1
export QT_QPA_PLATFORM=offscreen
if ! $USE_GPU; then
    export CUDA_VISIBLE_DEVICES=""  # GitHub runners have no GPU
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

LOG_DIR="$CI_HOME/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$CI_HOME/envs"
declare -A RESULTS

run_step() {  # run_step <version> <name> <command...>; records PASS/FAIL
    local version=$1 name=$2; shift 2
    echo; echo "=== [py$version] $name: $*"
    local start=$SECONDS
    if "$@"; then
        RESULTS["$version/$name"]="PASS ($((SECONDS - start)) s)"
    else
        RESULTS["$version/$name"]="FAIL ($((SECONDS - start)) s)"
    fi
}

setup_env() {  # create/update the env for one Python version; prints its python path
    local version=$1 prefix="$CI_HOME/envs/py$1"
    if $FRESH && [[ -d "$prefix" ]]; then
        "$CONDA" env remove -y -p "$prefix" >/dev/null
    fi
    if [[ ! -x "$prefix/bin/python" ]]; then
        "$CONDA" create -y -q -p "$prefix" "python=$version" pip >/dev/null || return 1
        rm -f "$prefix/.requirements.sha"
    fi
    local py="$prefix/bin/python"
    local req_sha; req_sha="$(sha256sum "$REPO/requirements.txt" | cut -d' ' -f1)"
    # "Install dependencies" step (skipped if requirements.txt is unchanged since last run)
    if [[ "$(cat "$prefix/.requirements.sha" 2>/dev/null)" != "$req_sha" ]]; then
        "$py" -m pip install -q --upgrade pip &&
        "$py" -m pip install -q flake8 pytest &&
        "$py" -m pip install -q -r "$REPO/requirements.txt" &&
        echo "$req_sha" > "$prefix/.requirements.sha" || return 1
    fi
    echo "$py"
}

run_version() {
    local version=$1 py
    echo "##### Python $version  (env: $CI_HOME/envs/py$version)"
    if ! py="$(setup_env "$version")"; then
        RESULTS["$version/setup"]="FAIL"; return
    fi
    py="$(tail -n1 <<< "$py")"
    "$py" -c "import sys, platform; print(sys.version, platform.system())"
    cd "$REPO" || return

    has_step lint && run_step "$version" lint \
        "$py" -m flake8 ./pypef --count --select=E9,F63,F7,F82 --show-source --statistics

    # The main-script step must use the source tree, not a previously pip-installed
    # copy (left behind by an earlier gui step), exactly like on a fresh CI runner.
    has_step main && "$py" -m pip uninstall -y -q pypef >/dev/null 2>&1
    has_step main && run_step "$version" main env PYTHONPATH="${PYTHONPATH:-}:$REPO" \
        "$py" -m pytest ./tests/ -v -m "not (pip_specific or requires_gpu or wheel_build)" \
        --log-cli-level=INFO --capture=tee-sys "${EXTRA_PYTEST_ARGS[@]}"

    if has_step gui; then
        local neutral src
        neutral="$(mktemp -d)"
        # Install from a clean copy (tracked + new, non-ignored files) like a fresh CI
        # checkout: a stale build/ directory in the repo could otherwise supply files
        # that are missing from package-data and hide packaging bugs.
        src="$(mktemp -d)"
        (cd "$REPO" && git ls-files -z -co --exclude-standard \
            | tar --null -T - --ignore-failed-read -cf - 2>/dev/null) | tar -xf - -C "$src"
        run_step "$version" gui bash -c '
            set -e
            "$1" -m pip install -q "$4[gui]"
            cd "$3"
            PYTHONPATH="" "$1" -m pytest "$2/tests" -v -m "pip_specific" \
                --import-mode=importlib --log-cli-level=INFO --capture=tee-sys "${@:5}"
        ' _ "$py" "$REPO" "$neutral" "$src" "${EXTRA_PYTEST_ARGS[@]}"
        rm -rf "$neutral" "$src"
    fi

    has_step wheel && run_step "$version" wheel \
        "$py" -m pytest ./tests/ -v -m "wheel_build" --log-cli-level=INFO --capture=tee-sys \
        "${EXTRA_PYTEST_ARGS[@]}"
}

echo "PyPEF local CI: Python ${VERSIONS}; steps ${STEPS}; GPU $($USE_GPU && echo on || echo off)"
echo "Logs: $LOG_DIR"
for version in ${VERSIONS//,/ }; do
    # fail-fast: false -> every version runs; output goes to terminal and log file
    run_version "$version" > >(tee "$LOG_DIR/py$version.log") 2>&1
    wait
done

echo; echo "===================== Summary ====================="
status=0
for version in ${VERSIONS//,/ }; do
    for step in setup ${STEPS//,/ }; do
        result="${RESULTS["$version/$step"]:-}"
        [[ -z "$result" ]] && continue
        printf "  py%-6s %-6s %s\n" "$version" "$step" "$result"
        [[ "$result" == FAIL* ]] && status=1
    done
done
echo "Logs: $LOG_DIR"
exit $status
