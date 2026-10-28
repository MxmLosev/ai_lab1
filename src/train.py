"""Обучение и сравнение моделей E0–E4 с записью в MLflow.
Запуск из корня проекта:  python -m src.train
"""
import json
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

from src.data import data_version, get_split, get_xy, load_config, load_data
from src.pipelines import make_pipeline

ALPHAS = [0.01, 0.1, 1, 3, 10, 30, 100]


def experiments(cfg):
    num, cat = cfg["data"]["numeric"], cfg["data"]["categorical"]
    seed = cfg["split"]["seed"]
    return [
        # E0: наивные ориентиры
        dict(name="E0-mean", group="E0", reg=DummyRegressor(strategy="mean"), num=num, cat=cat),
        dict(name="E0-median", group="E0", reg=DummyRegressor(strategy="median"), num=num, cat=cat),
        # E1: исходный проект (OLS, без вида, с Volume) и его варианты
        dict(name="E1-ols-original", group="E1", reg=LinearRegression(), num=num, cat=[], volume=True),
        dict(name="E1-ols-species", group="E1", reg=LinearRegression(), num=num, cat=cat),
        dict(name="E1-ols-species-log", group="E1", reg=LinearRegression(), num=num, cat=cat, log=True),
        # E2: гребневая регрессия, alpha по перекрёстной проверке
        dict(name="E2-ridge", group="E2", reg=Ridge(), num=num, cat=cat,
             grid={"regressor__alpha": ALPHAS}),
        dict(name="E2-ridge-log", group="E2", reg=Ridge(), num=num, cat=cat, log=True,
             grid={"regressor__regressor__alpha": ALPHAS}),
        # E3: ансамбли, маленькая заранее заданная сетка
        dict(name="E3-random-forest", group="E3",
             reg=RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
             num=num, cat=cat,
             grid={"regressor__max_depth": [None, 8], "regressor__min_samples_leaf": [1, 3]}),
        dict(name="E3-hist-gb", group="E3",
             reg=HistGradientBoostingRegressor(random_state=seed),
             num=num, cat=cat,
             grid={"regressor__learning_rate": [0.05, 0.1], "regressor__max_iter": [200, 500],
                   "regressor__min_samples_leaf": [5, 10]}),
        # E4: вопрос «улучшает ли признак Volume гребневую регрессию?» (сравнивать с E2-ridge)
        dict(name="E4-ridge-volume", group="E4", reg=Ridge(), num=num, cat=cat, volume=True,
             grid={"regressor__alpha": ALPHAS}),
        # E4: вариант «заменить группу длин одним признаком Volume»
        dict(name="E4-ridge-volume-only", group="E4", reg=Ridge(), num=[], cat=cat, volume=True,
             grid={"regressor__alpha": ALPHAS}),
    ]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def test_metrics(y_true, y_pred):
    return {
        "test_mse": mean_squared_error(y_true, y_pred),
        "test_mae": mean_absolute_error(y_true, y_pred),
        "test_r2": r2_score(y_true, y_pred),
        "test_n_negative": int((y_pred < 0).sum()),
    }


def plot_residuals(y_true, y_pred, title):
    res = y_true - y_pred
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_pred, res, alpha=0.7)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set(xlabel="Прогноз, г", ylabel="Остаток (y - ŷ), г", title="Остатки")
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax[1].scatter(y_true, y_pred, alpha=0.7)
    ax[1].plot(lim, lim, color="red", lw=1)
    ax[1].set(xlabel="Истинная масса, г", ylabel="Прогноз, г", title="Прогноз vs истина")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def run_experiment(exp, cfg, df, train_idx, test_idx, cv, dver, commit):
    X_train, y_train = get_xy(df, train_idx, cfg)
    X_test, y_test = get_xy(df, test_idx, cfg)
    pipe = make_pipeline(exp["reg"], exp["num"], exp["cat"],
                         volume=exp.get("volume", False), log_target=exp.get("log", False))

    best_params = {}
    if exp.get("grid"):
        search = GridSearchCV(pipe, exp["grid"], cv=cv, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        best_params = search.best_params_

    scores = cross_validate(clone(pipe), X_train, y_train, cv=cv,
                            scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"])
    cv_mse, cv_mae, cv_r2 = (-scores["test_neg_mean_squared_error"],
                             -scores["test_neg_mean_absolute_error"], scores["test_r2"])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    tm = test_metrics(y_test.values, y_pred)

    run_name = f"{exp['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as run:
        features = list(exp["num"]) + (["Volume"] if exp.get("volume") else []) + list(exp["cat"])
        mlflow.log_params({
            "experiment_group": exp["group"],
            "model_class": type(exp["reg"]).__name__,
            "log_target": exp.get("log", False),
            "features": ",".join(features),
            "data_path": cfg["data"]["path"],
            "data_version": dver,
            "split_file": cfg["split"]["split_file"],
            "test_size": cfg["split"]["test_size"],
            "seed": cfg["split"]["seed"],
            "cv": f"KFold(n_splits={cfg['split']['cv_folds']}, shuffle=True)",
            "n_train": len(train_idx), "n_test": len(test_idx),
            "grid": json.dumps(exp.get("grid", {})),
            "grid_size": int(np.prod([len(v) for v in exp.get("grid", {}).values()])) if exp.get("grid") else 0,
            **{f"best_{k.split('__')[-1]}": v for k, v in best_params.items()},
        })
        mlflow.log_metrics({
            "cv_mse_mean": cv_mse.mean(), "cv_mse_std": cv_mse.std(),
            "cv_mae_mean": cv_mae.mean(), "cv_mae_std": cv_mae.std(),
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
            **tm,
        })
        mlflow.set_tags({
            "git_commit": commit,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        })
        # таблица прогнозов и остатков
        table = df.loc[test_idx].copy()
        table["prediction"] = y_pred
        table["residual"] = table[cfg["data"]["target"]] - y_pred
        table["abs_error"] = table["residual"].abs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_predictions.csv"
            table.to_csv(path, index_label="row_id")
            mlflow.log_artifact(str(path))
        fig = plot_residuals(y_test.values, y_pred, exp["name"])
        mlflow.log_figure(fig, "residuals.png")
        plt.close(fig)
        run_id = run.info.run_id

    print(f"{exp['name']:24s} CV MSE={cv_mse.mean():10.0f}±{cv_mse.std():8.0f}  "
          f"test MSE={tm['test_mse']:10.0f}  MAE={tm['test_mae']:7.1f}  R2={tm['test_r2']:.3f}")
    return dict(name=exp["name"], group=exp["group"], run_id=run_id,
                cv_mse_mean=cv_mse.mean(), cv_mse_std=cv_mse.std(),
                cv_mae_mean=cv_mae.mean(), cv_mae_std=cv_mae.std(),
                cv_r2_mean=cv_r2.mean(), cv_r2_std=cv_r2.std(),
                **tm, best_params=json.dumps(best_params), pipeline=pipe)


def main():
    cfg = load_config()
    df = load_data(cfg)
    train_idx, test_idx = get_split(df, cfg)
    cv = KFold(n_splits=cfg["split"]["cv_folds"], shuffle=True, random_state=cfg["split"]["seed"])
    dver, commit = data_version(cfg["data"]["path"]), git_commit()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    results = [run_experiment(e, cfg, df, train_idx, test_idx, cv, dver, commit)
               for e in experiments(cfg)]

    # лучшая модель выбирается по CV MSE на train, а НЕ по тесту
    best = min((r for r in results if r["group"] != "E0"), key=lambda r: r["cv_mse_mean"])
    print(f"\nЛучшая по CV MSE: {best['name']}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], "models/best_model.joblib")
    X_test, _ = get_xy(df, test_idx, cfg)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model", "true")
        try:
            mlflow.sklearn.log_model(best["pipeline"], name="model", input_example=X_test.head(3))
        except TypeError:  # MLflow 2.x
            mlflow.sklearn.log_model(best["pipeline"], artifact_path="model", input_example=X_test.head(3))
        if Path("models/model-card.md").exists():
            mlflow.log_artifact("models/model-card.md")

    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results]) \
        .round(3).to_csv("reports/experiments.csv", index=False)
    Path("models/best_model.json").write_text(json.dumps(
        {"name": best["name"], "run_id": best["run_id"], "data_version": dver,
         "git_commit": commit}, indent=1, ensure_ascii=False))
    print("Сохранено: models/best_model.joblib, reports/experiments.csv")


if __name__ == "__main__":
    main()