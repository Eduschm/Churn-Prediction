"""
Integration tests for ChurnTrainer using synthetic data.
"""
import os
import numpy as np
import pandas as pd
import pytest

from src.train import ChurnTrainer


def _make_processed_df(n=300, seed=42):
    """
    Synthetic DataFrame that mirrors DataProcessor output shape:
    log-transformed numeric features + categorical features + target.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "MonthlyCharges": np.log1p(rng.uniform(20, 100, n)),
        "tenure": rng.integers(1, 72, n).astype(float),
        "TotalCharges": np.log1p(rng.uniform(100, 7000, n)),
        "avg_monthly_spend": np.log1p(rng.uniform(10, 100, n)),
        "contract_value": np.log1p(rng.uniform(100, 5000, n)),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "tenure_group": rng.choice(["<6m", "6-12m", "12-24m", "24-48m", "48m+"], n),
        "low_charge": rng.integers(0, 2, n),
        "Churn": rng.choice(["Yes", "No"], n),
    })


@pytest.fixture
def trainer(tmp_path):
    df = _make_processed_df()
    return ChurnTrainer(
        df=df,
        model_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
        config_dir=str(tmp_path / "config"),
        random_state=42,
    )


def test_split_sizes(trainer):
    trainer.split()
    total = len(trainer.df)
    assert len(trainer.X_train) + len(trainer.X_test) == total
    assert len(trainer.y_train) + len(trainer.y_test) == total


def test_split_stratification(trainer):
    trainer.split()
    train_rate = trainer.y_train.mean()
    test_rate = trainer.y_test.mean()
    assert abs(train_rate - test_rate) < 0.05


def test_train_decision_tree(trainer):
    trainer.split()
    trainer.param_grids["dt"] = {
        "classifier__max_depth": [3],
        "classifier__min_samples_split": [2],
    }
    trainer._run_grid_search("dt", verbose=0)
    assert "dt" in trainer.best_models
    preds = trainer.best_models["dt"].predict(trainer.X_test)
    assert len(preds) == len(trainer.X_test)
    assert set(preds).issubset({0, 1})


def test_evaluate_prints(trainer, capsys):
    trainer.split()
    trainer.param_grids["dt"] = {"classifier__max_depth": [3], "classifier__min_samples_split": [2]}
    trainer._run_grid_search("dt", verbose=0)
    trainer.evaluate("dt")
    captured = capsys.readouterr()
    assert "DT" in captured.out
    assert "Classification Report" in captured.out


def test_save_dt_model(trainer):
    trainer.split()
    trainer.param_grids["dt"] = {"classifier__max_depth": [3], "classifier__min_samples_split": [2]}
    trainer._run_grid_search("dt", verbose=0)
    path = trainer.save_dt_model()
    assert os.path.exists(path)


def test_save_all_params(trainer):
    trainer.split()
    trainer.param_grids["dt"] = {"classifier__max_depth": [3], "classifier__min_samples_split": [2]}
    trainer._run_grid_search("dt", verbose=0)
    path = trainer.save_all_params()
    assert os.path.exists(path)


def test_apply_saved_params_and_fit(trainer, tmp_path):
    trainer.split()
    trainer.param_grids["dt"] = {"classifier__max_depth": [3], "classifier__min_samples_split": [2]}
    trainer._run_grid_search("dt", verbose=0)
    params_path = trainer.save_all_params()

    # New trainer loads saved params and fits without grid search
    trainer2 = ChurnTrainer(
        df=_make_processed_df(),
        model_dir=str(tmp_path / "models2"),
        results_dir=str(tmp_path / "results2"),
        config_dir=str(tmp_path / "config"),
        random_state=42,
    )
    trainer2.split()
    trainer2.load_saved_params(params_path)
    trainer2._apply_saved_params_and_fit("dt")
    assert "dt" in trainer2.best_models


def test_evaluate_raises_without_model(trainer):
    trainer.split()
    with pytest.raises(ValueError, match="No trained model"):
        trainer.evaluate("dt")
