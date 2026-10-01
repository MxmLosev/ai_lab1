"""Проверки качества данных FishGrow.

Каждая функция check_* возвращает DataFrame с проблемными строками
(индекс = номер строки исходной таблицы) и столбцом `issue` с описанием.
Пустой результат означает, что проверка пройдена.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]
NUMERIC_COLUMNS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
LENGTH_COLUMNS = ["Length1", "Length2", "Length3"]


def _issues(df: pd.DataFrame, mask: pd.Series, text: str) -> pd.DataFrame:
    """Строки df, где mask истинна, с описанием проблемы."""
    out = df.loc[mask].copy()
    out["issue"] = text
    return out


def check_schema(df: pd.DataFrame) -> List[str]:
    """Сверка столбцов и типов с ожидаемой схемой. Возвращает список проблем."""
    problems = []
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing:
        problems.append(f"нет столбцов: {sorted(missing)}")
    if extra:
        problems.append(f"лишние столбцы: {sorted(extra)}")
    if not df.columns.is_unique:
        problems.append("повторяющиеся имена столбцов")
    for col in NUMERIC_COLUMNS:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col}: ожидался числовой тип, получен {df[col].dtype}")
    return problems


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Полные дубликаты строк (все повторы, включая первый)."""
    return _issues(df, df.duplicated(keep=False), "полный дубликат строки")


def check_missing_and_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Пропуски в любом столбце и бесконечности в числовых."""
    num = df[NUMERIC_COLUMNS]
    mask = df.isna().any(axis=1) | np.isinf(num).any(axis=1)
    return _issues(df, mask, "пропуск или бесконечность")


def check_non_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Значения <= 0 в физических величинах (масса и размеры должны быть > 0)."""
    parts = []
    for col in NUMERIC_COLUMNS:
        mask = df[col] <= 0
        if mask.any():
            parts.append(_issues(df, mask, f"{col} <= 0"))
    return pd.concat(parts) if parts else df.iloc[0:0].assign(issue=[])


def check_length_order(df: pd.DataFrame) -> pd.DataFrame:
    """Нарушение порядка Length1 <= Length2 <= Length3."""
    mask = (df["Length1"] > df["Length2"]) | (df["Length2"] > df["Length3"])
    return _issues(df, mask, "нарушен порядок Length1 <= Length2 <= Length3")


def check_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Невозможные пропорции тела: высота или ширина не меньше полной длины."""
    mask = (df["Height"] >= df["Length3"]) | (df["Width"] >= df["Length3"])
    return _issues(df, mask, "Height или Width >= Length3")


def check_rare_categories(df: pd.DataFrame, col: str = "Species",
                          min_share: float = 0.05) -> pd.DataFrame:
    """Численность и доля категорий; флаг rare для долей ниже min_share."""
    counts = df[col].value_counts()
    table = pd.DataFrame({"count": counts, "share": counts / len(df)})
    table["rare"] = table["share"] < min_share
    return table


def check_near_duplicates(df: pd.DataFrame, cols: Optional[List[str]] = None,
                          group: str = "Species", threshold: float = 0.05) -> pd.DataFrame:
    """Почти одинаковые строки внутри одного вида.

    Признаки стандартизуются (z-оценка по всей таблице), затем внутри каждого
    вида считаются попарные евклидовы расстояния. Пары с расстоянием меньше
    threshold (в единицах стандартного отклонения) считаются почти дубликатами.
    Полные дубликаты (расстояние 0) тоже попадают в результат.
    """
    cols = cols or NUMERIC_COLUMNS
    std = df[cols].std().replace(0, 1.0)   # постоянный столбец не должен давать NaN
    z = (df[cols] - df[cols].mean()) / std
    pairs = []
    for species, idx in df.groupby(group).groups.items():
        idx = list(idx)
        x = z.loc[idx].to_numpy()
        dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
        i, j = np.where(np.triu(dist < threshold, k=1))
        for a, b in zip(i, j):
            pairs.append({"row_a": idx[a], "row_b": idx[b],
                          group: species, "distance": round(float(dist[a, b]), 4)})
    return pd.DataFrame(pairs, columns=["row_a", "row_b", group, "distance"])


def check_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Доля уникальных значений по столбцам; unique_ratio = 1 — кандидат в идентификатор."""
    nunique = df.nunique(dropna=False)
    table = pd.DataFrame({"nunique": nunique, "unique_ratio": nunique / len(df)})
    table["id_candidate"] = table["unique_ratio"] == 1.0
    return table


def run_row_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка построчных проверок: число найденных строк по каждой."""
    checks = {
        "полные дубликаты": check_duplicates,
        "пропуски / бесконечности": check_missing_and_inf,
        "значения <= 0": check_non_positive,
        "порядок длин": check_length_order,
        "пропорции тела": check_proportions,
    }
    rows = []
    for name, func in checks.items():
        found = func(df)
        rows.append({"check": name, "n_rows": found.index.nunique(),
                     "rows": sorted(found.index.unique().tolist())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Выбросы (шаг 5). Границы считаются внутри групп (по умолчанию — по виду),
# т.к. масса и размеры сильно зависят от вида.
# ВАЖНО: при моделировании границы нужно оценивать только на обучающей выборке.
# ---------------------------------------------------------------------------

def iqr_bounds(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Границы Q1 - k*IQR и Q3 + k*IQR для одного ряда."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return pd.Series({"q1": q1, "q3": q3, "iqr": iqr,
                      "low": q1 - k * iqr, "high": q3 + k * iqr})


def iqr_flags(df: pd.DataFrame, col: str, group: str = "Species",
              k: float = 1.5) -> pd.Series:
    """True для строк, где значение col вне границ IQR своей группы."""

    def _flag(s):
        b = iqr_bounds(s, k)
        return (s < b["low"]) | (s > b["high"])

    return df.groupby(group)[col].transform(_flag).astype(bool)


def mad_z(df: pd.DataFrame, col: str, group: str = "Species") -> pd.Series:
    """Устойчивая z-оценка 0.6745 * (x - медиана) / MAD внутри группы.

    Если MAD группы равен 0 (больше половины значений одинаковы),
    z не определена и возвращается NaN.
    """

    def _z(s):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return pd.Series(np.nan, index=s.index)
        return 0.6745 * (s - med) / mad

    return df.groupby(group)[col].transform(_z)


def outlier_table(df: pd.DataFrame, col: str, group: str = "Species",
                  k: float = 1.5, z_thr: float = 3.5) -> pd.DataFrame:
    """Строки, помеченные IQR или MAD, с указанием, какой метод сработал."""
    out = df[[group, col]].copy()
    out["iqr_flag"] = iqr_flags(df, col, group, k)
    out["mad_z"] = mad_z(df, col, group).round(2)
    out["mad_flag"] = out["mad_z"].abs() > z_thr
    out["agreement"] = np.select(
        [out["iqr_flag"] & out["mad_flag"], out["iqr_flag"], out["mad_flag"]],
        ["оба метода", "только IQR", "только MAD"], default="")
    return out[out["iqr_flag"] | out["mad_flag"]]