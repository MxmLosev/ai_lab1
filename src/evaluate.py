"""Анализ ошибок лучшей модели и диагностика линейной регрессии.
Запуск из корня проекта (после src.train):  python -m src.evaluate
Результаты: reports/error_analysis.md и графики в reports/figures/
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from src.data import get_split, get_xy, load_config, load_data
from src.pipelines import add_volume, make_pipeline

FIG = Path("reports/figures")


def md_table(df, digits=1):
    """DataFrame -> таблица Markdown."""
    df = df.round(digits)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def vif(X):
    """VIF_j = 1 / (1 - R2_j), где R2_j — R2 регрессии признака j на остальные."""
    res = {}
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        res[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.DataFrame({"feature": list(res), "VIF": list(res.values())})


def breusch_pagan(X, resid):
    """Тест Бройша-Пагана: регрессия квадратов остатков на признаки, LM = n * R2 ~ chi2(k)."""
    e2 = resid ** 2
    r2 = LinearRegression().fit(X, e2).score(X, e2)
    return chi2.sf(len(e2) * r2, X.shape[1])


def main():
    cfg = load_config()
    df = load_data(cfg)
    train_idx, test_idx = get_split(df, cfg)
    X_train, y_train = get_xy(df, train_idx, cfg)
    X_test, y_test = get_xy(df, test_idx, cfg)
    target, num = cfg["data"]["target"], cfg["data"]["numeric"]
    FIG.mkdir(parents=True, exist_ok=True)
    out = []

    # ---------- 1. Лучшая модель: остатки ----------
    best_info = json.loads(Path("models/best_model.json").read_text())
    model = joblib.load("models/best_model.joblib")
    pred = model.predict(X_test)
    t = df.loc[test_idx].copy()
    t["prediction"] = pred
    t["residual"] = t[target] - pred
    t["abs_error"] = t["residual"].abs()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for sp, g in t.groupby("Species"):
        ax[0].scatter(g["prediction"], g["residual"], label=sp, alpha=0.8)
        ax[1].scatter(g["Length3"], g["residual"], label=sp, alpha=0.8)
    for a in ax:
        a.axhline(0, color="red", lw=1)
    ax[0].set(xlabel="Прогноз, г", ylabel="Остаток, г", title="Остатки vs прогноз")
    ax[1].set(xlabel="Length3, см", ylabel="Остаток, г", title="Остатки vs длина")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Лучшая модель: {best_info['name']}")
    fig.tight_layout()
    fig.savefig(FIG / "best_residuals.png", dpi=120)
    plt.close(fig)

    out.append(f"# Анализ ошибок: {best_info['name']}\n")
    out.append(f"Тест: {len(t)} объектов, MAE = {t['abs_error'].mean():.1f} г\n")

    # ---------- 2. Пять крупнейших ошибок ----------
    cols = ["Species", "Length1", "Length2", "Length3", "Height", "Width",
            target, "prediction", "residual"]
    top = t.sort_values("abs_error", ascending=False).head(5)[cols]
    out.append("## Пять крупнейших абсолютных ошибок\n")
    out.append(md_table(top.reset_index(names="row_id")) + "\n")

    # ---------- 3. MAE по видам и диапазонам размера ----------
    by_sp = t.groupby("Species").agg(n=(target, "size"), mean_weight=(target, "mean"),
                                     MAE=("abs_error", "mean")).reset_index()
    out.append("## MAE по видам\n")
    out.append(md_table(by_sp.sort_values("MAE", ascending=False)) + "\n")

    t["size_bin"] = pd.cut(t["Length3"], bins=[0, 20, 30, 40, 100],
                           labels=["<20 см", "20-30 см", "30-40 см", ">40 см"])
    by_size = t.groupby("size_bin", observed=True).agg(
        n=(target, "size"), mean_weight=(target, "mean"), MAE=("abs_error", "mean"),
        MAE_pct=("abs_error", lambda s: 100 * (s / t.loc[s.index, target]).mean()),
    ).reset_index()
    out.append("## MAE по диапазонам размера (Length3)\n")
    out.append("`MAE_pct`: средняя относительная ошибка, %.\n")
    out.append(md_table(by_size) + "\n")

    # ---------- 4. Отрицательные и невозможные прогнозы ----------
    out.append("## Отрицательные и физически невозможные прогнозы\n")
    out.append(f"Лучшая модель на тесте: отрицательных прогнозов {int((pred < 0).sum())}, "
               f"минимальный прогноз {pred.min():.1f} г.\n")
    exp = pd.read_csv("reports/experiments.csv")
    out.append(md_table(exp[["name", "test_n_negative"]], 0) + "\n")
    small = df[df["Length3"] < 15]
    if len(small):
        sp = model.predict(small[num + cfg["data"]["categorical"]])
        out.append(f"Лучшая модель на всех рыбах с Length3 < 15 см (n={len(small)}): "
                   f"отрицательных прогнозов {(sp < 0).sum()}, мин. {sp.min():.1f} г.\n")
    ols = make_pipeline(LinearRegression(), num, [], volume=True).fit(X_train, y_train)
    p_ols = ols.predict(df[num + cfg["data"]["categorical"]])
    out.append(f"Исходная модель (E1-ols-original) на всех данных: отрицательных прогнозов "
               f"{(p_ols < 0).sum()}, мин. {p_ols.min():.1f} г.\n")

    # ---------- 5. Диагностика E1: мультиколлинеарность и гетероскедастичность ----------
    Xn = add_volume(X_train[num])
    corr = Xn.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Корреляции числовых признаков (train)")
    fig.tight_layout()
    fig.savefig(FIG / "correlation.png", dpi=120)
    plt.close(fig)
    out.append("## Диагностика линейной модели (E1)\n")
    out.append("### Корреляции числовых признаков\n")
    out.append(md_table(corr.reset_index(names="feature"), 3) + "\n")

    out.append("### VIF (фактор инфляции дисперсии, > 10 = сильная мультиколлинеарность)\n")
    out.append(md_table(vif(Xn)) + "\n")

    fitted = LinearRegression().fit(Xn, y_train).predict(Xn)
    resid = y_train.values - fitted
    bp_p = breusch_pagan(Xn, resid)
    out.append(f"### Гетероскедастичность\nТест Бройша-Пагана для OLS (как в исходном проекте), "
               f"p-value = {bp_p:.2e} "
               f"({'дисперсия остатков неоднородна' if bp_p < 0.05 else 'нет оснований отвергнуть однородность'}).\n")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(fitted, resid, alpha=0.7)
    ax.axhline(0, color="red", lw=1)
    ax.set(xlabel="Прогноз, г", ylabel="Остаток, г", title="E1-ols-original: остатки на train")
    fig.tight_layout()
    fig.savefig(FIG / "e1_residuals.png", dpi=120)
    plt.close(fig)

    # ---------- 6. Устойчивость коэффициентов (OLS vs Ridge) ----------
    cv = KFold(cfg["split"]["cv_folds"], shuffle=True, random_state=cfg["split"]["seed"])
    alpha = json.loads(exp.set_index("name").loc["E4-ridge-volume", "best_params"])["regressor__alpha"]
    rows = []
    for label, reg in [("OLS", LinearRegression()), (f"Ridge(alpha={alpha})", Ridge(alpha=alpha))]:
        pipe = make_pipeline(reg, num, [], volume=True)
        coefs = []
        for tr, _ in cv.split(X_train):
            m = clone(pipe).fit(X_train.iloc[tr], y_train.iloc[tr])
            coefs.append(m.named_steps["regressor"].coef_)
        coefs = np.array(coefs)
        for f, mean, std in zip(num + ["Volume"], coefs.mean(0), coefs.std(0)):
            rows.append({"model": label, "feature": f, "coef_mean": mean, "coef_std": std})
    out.append("### Устойчивость коэффициентов по 5 фолдам (признаки стандартизованы, без вида)\n")
    out.append("Коэффициенты сравнимы между собой только после StandardScaler: "
               "тогда каждый показывает изменение массы при изменении признака на 1 ст. откл.\n")
    out.append(md_table(pd.DataFrame(rows)) + "\n")

    Path("reports/error_analysis.md").write_text("\n".join(out), encoding="utf-8")
    t.drop(columns="size_bin").to_csv("reports/test_predictions_best.csv", index_label="row_id")

    # приложить результаты анализа к запуску лучшей модели в MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=best_info["run_id"]):
        mlflow.log_artifact("reports/error_analysis.md", "analysis")
        mlflow.log_artifacts(str(FIG), "analysis/figures")
    print("Готово: reports/error_analysis.md, reports/figures/")


if __name__ == "__main__":
    main()