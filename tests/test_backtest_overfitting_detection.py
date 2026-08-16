from datetime import date

from src.backtest.overfitting_detection import detect
from src.backtest.walk_forward_validator import Fold


def _fold(n, passed, in_sample_exp=10.0, out_sample_exp=8.0):
    return Fold(
        fold_number=n,
        train_window_start=date(2024, 1, 1),
        train_window_end=date(2024, 2, 1),
        test_window_start=date(2024, 2, 1),
        test_window_end=date(2024, 3, 1),
        in_sample_metrics={"expectancy": in_sample_exp},
        out_of_sample_metrics={"expectancy": out_sample_exp},
        p_value=0.01,
        passed=passed,
    )


def test_detect_no_folds_is_marginal():
    result = detect([])
    assert result.verdict == "marginal"
    assert result.n_folds == 0


def test_detect_all_folds_pass_is_robust():
    folds = [_fold(i, True) for i in range(1, 6)]
    result = detect(folds)
    assert result.verdict == "robust"
    assert result.walk_forward_failure_rate == 0.0


def test_detect_majority_fail_is_overfit():
    folds = [_fold(1, True), _fold(2, False), _fold(3, False), _fold(4, False)]
    result = detect(folds)
    assert result.verdict == "overfit"


def test_detect_unstable_parameters_forces_overfit_even_if_folds_pass():
    folds = [_fold(i, True) for i in range(1, 6)]
    result = detect(folds, parameter_sensitivity={"stable": False, "jaggedness_score": 2.0})
    assert result.verdict == "overfit"


def test_detect_computes_in_sample_out_of_sample_gap():
    folds = [_fold(1, True, in_sample_exp=10.0, out_sample_exp=5.0)]
    result = detect(folds)
    assert result.in_sample_out_of_sample_gap_pct == 50.0


def test_detect_marginal_when_moderate_failure_rate():
    folds = [_fold(1, True), _fold(2, True), _fold(3, True), _fold(4, False)]  # 25% failure
    result = detect(folds)
    assert result.verdict == "marginal"
