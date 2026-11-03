"""Разбиение данных FishGrow на обучающую, проверочную и тестовую выборки."""
from typing import Dict, Optional

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def stratified_group_split(df: pd.DataFrame, strat_col: str = "Species",
                           group_col: Optional[str] = None,
                           fractions: Dict[str, float] = None,
                           seed: int = 42) -> pd.Series:
    """Стратифицированное групповое разбиение.

    - Внутри каждого значения strat_col (вида) группы перемешиваются
      генератором с фиксированным seed и делятся в пропорциях fractions.
    - Строки одной группы (group_col) всегда попадают в одну часть.
    - Если group_col не задан, каждая строка — отдельная группа.
    - Если в слое хотя бы 3 группы, в val и test попадает минимум по одной.

    Возвращает Series с меткой части ('train' / 'val' / 'test') для каждой строки.
    """
    fractions = fractions or {"train": 0.70, "val": 0.15, "test": 0.15}
    assert abs(sum(fractions.values()) - 1) < 1e-9, "доли должны давать 1"
    rng = np.random.default_rng(seed)
    groups = df[group_col] if group_col else pd.Series(df.index, index=df.index)

    labels = pd.Series(index=df.index, dtype=object)
    for _, layer in df.groupby(strat_col, sort=True):
        layer_groups = groups.loc[layer.index].unique()
        layer_groups = layer_groups[rng.permutation(len(layer_groups))]
        n = len(layer_groups)
        n_test = int(round(n * fractions["test"]))
        n_val = int(round(n * fractions["val"]))
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        parts = (["test"] * n_test + ["val"] * n_val
                 + ["train"] * (n - n_test - n_val))
        mapping = dict(zip(layer_groups, parts))
        labels.loc[layer.index] = groups.loc[layer.index].map(mapping)
    return labels


def check_split(labels: pd.Series, groups: Optional[pd.Series] = None) -> None:
    """Проверки корректности разбиения; при нарушении — AssertionError."""
    assert labels.notna().all(), "есть строки без метки части"
    assert set(labels.unique()) <= set(SPLITS), "неизвестная метка части"
    parts = {s: set(labels.index[labels == s]) for s in SPLITS}
    assert not (parts["train"] & parts["val"]), "train и val пересекаются"
    assert not (parts["train"] & parts["test"]), "train и test пересекаются"
    assert not (parts["val"] & parts["test"]), "val и test пересекаются"
    if groups is not None:
        per_group = labels.groupby(groups).nunique()
        assert (per_group == 1).all(), f"группы разнесены по частям: {list(per_group[per_group > 1].index)}"