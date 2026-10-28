"""Сборка цепочек «предобработка + модель». Вся предобработка внутри Pipeline,
поэтому в перекрёстной проверке она настраивается заново на каждом фолде."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_volume(X):
    """Производный признак Vproxy = Length3 * Height * Width (как в исходном проекте)."""
    X = X.copy()
    X["Volume"] = X["Length3"] * X["Height"] * X["Width"]
    return X


def make_pipeline(regressor, numeric, categorical, volume=False, log_target=False):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_cols = list(numeric) + (["Volume"] if volume else [])
    transformers = [("num", num, num_cols)]
    if categorical:
        transformers.append(("cat", cat, list(categorical)))

    steps = []
    if volume:
        steps.append(("volume", FunctionTransformer(add_volume)))
    steps.append(("preprocess", ColumnTransformer(transformers)))

    if log_target:
        # обучаемся на log(1+y), прогноз автоматически возвращается в граммы через expm1
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    steps.append(("regressor", regressor))
    return Pipeline(steps)