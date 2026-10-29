"""Минимальные тесты из раздела 7. Запуск из корня проекта:  python -m pytest -v"""
import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import get_split, get_xy, load_config, load_data
from src.pipelines import make_pipeline


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    df = load_data(cfg)
    train_idx, test_idx = get_split(df, cfg)
    return cfg, df, train_idx, test_idx


@pytest.fixture(scope="module")
def fitted(setup):
    cfg, df, train_idx, _ = setup
    X, y = get_xy(df, train_idx, cfg)
    model = make_pipeline(Ridge(alpha=1.0), cfg["data"]["numeric"],
                          cfg["data"]["categorical"], volume=True)
    return model.fit(X, y)


def test_train_test_do_not_overlap(setup):
    _, df, train_idx, test_idx = setup
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(df)


def test_unknown_category_is_accepted(setup, fitted):
    cfg, df, _, test_idx = setup
    X, _ = get_xy(df, test_idx[:1], cfg)
    X = X.copy()
    X["Species"] = "Shark"  # вида нет в обучающих данных
    pred = fitted.predict(X)
    assert pred.shape == (1,) and np.isfinite(pred).all()


def test_prediction_count_and_finite(setup, fitted):
    cfg, df, _, test_idx = setup
    X, _ = get_xy(df, test_idx, cfg)
    pred = fitted.predict(X)
    assert len(pred) == len(X)
    assert np.isfinite(pred).all()


def test_metrics_match_manual_calculation():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 330.0, 380.0])
    # ошибки: -10, 10, -30, 20
    mse = (100 + 100 + 900 + 400) / 4                    # 375
    mae = (10 + 10 + 30 + 20) / 4                        # 17.5
    r2 = 1 - 1500 / (150**2 + 50**2 + 50**2 + 150**2)    # 1 - 1500/50000 = 0.97
    assert mean_squared_error(y_true, y_pred) == pytest.approx(mse)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(mae)
    assert r2_score(y_true, y_pred) == pytest.approx(r2)


def test_saved_pipeline_reproduces_prediction(setup, fitted, tmp_path):
    cfg, df, _, test_idx = setup
    X, _ = get_xy(df, test_idx, cfg)
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(X), fitted.predict(X), rtol=1e-9)