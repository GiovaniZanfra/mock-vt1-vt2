import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
)


def load_model(class_path, params=None):
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    if params:
        return cls(**params)
    return cls()


def extract_sid_from_index(idx):
    if isinstance(idx, pd.MultiIndex):
        lvl0 = idx.get_level_values(0)
    else:
        lvl0 = idx
    return pd.Series(lvl0, index=idx).str.split("-", n=1).str[0]


def dropnas_train_test(X_tr, y_tr, X_te, y_te):
    for df in [X_tr, y_tr, X_te, y_te]:
        if "idx" in df.columns:
            df.set_index("idx")

    combined_train = pd.concat([X_tr, y_tr], axis=1)
    combined_train = combined_train.dropna()
    X_tr = combined_train.drop(columns=y_tr.columns)
    y_tr = combined_train[y_tr.columns].to_numpy().ravel()

    # For test data
    combined_test = pd.concat([X_te, y_te], axis=1)
    combined_test = combined_test.dropna()
    X_te = combined_test.drop(columns=y_te.columns)
    y_te = combined_test[y_te.columns].to_numpy().ravel()

    return X_tr, y_tr, X_te, y_te


def compute_mae_by_group(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Calcula o MAE (Mean Absolute Error) para cada subgrupo definido pelas colunas em `group_cols`.
    Cada coluna deve ser binária (0 ou 1), indicando pertencimento ao subgrupo.

    Args:
        X: DataFrame contendo as colunas dos subgrupos (por ex.: "gender_0.0", "age_bin_20-30", etc.).
        y_true: Array com valores verdadeiros (aligned com X).
        y_pred: Array com predições correspondentes (aligned com X).
        group_cols: Lista de colunas em X que definem cada subgrupo.

    Retorna:
        DataFrame com duas colunas:
            - "subgroup": nome da coluna / subgrupo.
            - "mae": valor do MAE calculado para todas as linhas em que X[col] == 1.
    """
    results = []

    for col in group_cols:
        if col not in X.columns:
            # Se a coluna não existe em X, pula
            continue

        mask = X[col] == 1
        if mask.sum() == 0:
            # Se não há nenhum exemplo desse subgrupo, pula
            continue

        mae_val = mean_absolute_error(y_true[mask], y_pred[mask])
        results.append((col, mae_val))

    return pd.DataFrame(results, columns=["subgroup", "mae"])
