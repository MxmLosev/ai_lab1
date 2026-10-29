import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path="configs/base.yaml"):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def data_version(path):
    """Версия данных = первые 12 символов md5 файла."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]


def load_data(cfg):
    df = pd.read_csv(cfg["data"]["path"], encoding="utf-8-sig")
    target = cfg["data"]["target"]
    # Масса <= 0 физически невозможна (в данных есть Roach с Weight = 0)
    df = df[df[target] > 0].reset_index(drop=True)
    return df


def get_split(df, cfg):
    """Индексы train/test. Файл создаётся один раз, дальше только читается."""
    split_file = Path(cfg["split"]["split_file"])
    if split_file.exists():
        s = json.loads(split_file.read_text())
        return s["train_idx"], s["test_idx"]

    train_idx, test_idx = train_test_split(
        df.index.to_list(),
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["seed"],
        stratify=df["Species"],  # все виды попадают в обе части
    )
    split_file.parent.mkdir(exist_ok=True)
    split_file.write_text(json.dumps(
        {"train_idx": sorted(train_idx), "test_idx": sorted(test_idx)}, indent=1))
    return sorted(train_idx), sorted(test_idx)


def get_xy(df, idx, cfg):
    part = df.loc[idx]
    features = cfg["data"]["numeric"] + cfg["data"]["categorical"]
    return part[features], part[cfg["data"]["target"]]