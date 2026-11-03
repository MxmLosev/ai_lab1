"""Быстрые тесты данных и разбиения. Запуск из корня проекта:

    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_checks as dc   # noqa: E402
from src import splitting as sp     # noqa: E402

SEED = 42
KNOWN_BAD_ROWS = {40}               # Weight = 0, задокументировано в отчёте


def _find_data() -> Path:
    """Ищет Fish.csv в проекте, пропуская виртуальные окружения."""
    for p in ROOT.rglob("Fish.csv"):
        if not any(part in {"venv", ".venv", "env"} for part in p.parts):
            return p
    pytest.skip("Fish.csv не найден в проекте")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(_find_data())


# --- Схема и качество -------------------------------------------------------

def test_schema(df):
    """Столбцы и типы совпадают с паспортом данных."""
    assert dc.check_schema(df) == []
    assert len(df) == 159


def test_no_missing_and_no_duplicates(df):
    assert df.isna().sum().sum() == 0
    assert dc.check_duplicates(df).empty


def test_only_known_non_positive_rows(df):
    """Значения <= 0 есть только в известной аномальной строке 40."""
    bad = set(dc.check_non_positive(df).index)
    assert bad == KNOWN_BAD_ROWS


def test_length_order(df):
    """Length1 <= Length2 <= Length3 во всех строках."""
    assert dc.check_length_order(df).empty


def test_units_and_ranges(df):
    """Срабатывает при смене единиц: мм вместо см, кг вместо г."""
    assert df["Length3"].between(5, 100).all(), "Length3 вне 5–100 см — единицы изменились?"
    assert df["Weight"].max() < 5000, "Weight > 5 кг — единицы изменились?"
    assert df["Weight"].max() > 100, "Weight слишком мал — масса в кг?"


# --- Разбиение -------------------------------------------------------------

@pytest.fixture(scope="module")
def split(df):
    data = df[df["Weight"] > 0].copy()
    data["group"] = data.index
    data.loc[[103, 104], "group"] = 103
    data["split"] = sp.stratified_group_split(data, "Species", "group", seed=SEED)
    return data


def test_split_no_overlap_and_groups_intact(split):
    sp.check_split(split["split"], split["group"])
    assert split.loc[103, "split"] == split.loc[104, "split"]
    assert 40 not in split.index


def test_split_every_species_in_every_part(split):
    table = pd.crosstab(split["Species"], split["split"])
    assert (table[list(sp.SPLITS)] > 0).all().all()


def test_split_is_reproducible(split):
    again = sp.stratified_group_split(split, "Species", "group", seed=SEED)
    assert (again == split["split"]).all()


def test_saved_split_matches(split):
    """Сохранённое разбиение совпадает с пересчитанным."""
    path = ROOT / "data" / "split_lab01.csv"
    if not path.exists():
        pytest.skip("data/split_lab01.csv ещё не создан")
    saved = pd.read_csv(path, index_col="row")["split"]
    assert (saved.sort_index() == split["split"].sort_index()).all()
