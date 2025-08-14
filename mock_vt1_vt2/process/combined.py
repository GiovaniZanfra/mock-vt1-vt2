import itertools

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Transformer para criar um conjunto completo de features demográficas, de bioimpedância,
    ganho de elevação e suas interações.

    Colunas esperadas no DataFrame X:
      - age, gender, height, weight, bmi, bmi_class
      - smm, bfm, ffm, tbw, ffmi, bmr, pbf, smi, tbwDffm
      - gain_mean, gain_std, percentile_gain

    Funcionalidades:
      1) Bin de age em faixas de 0–20, 20–30, 30–40, 40–50, 50+.
      2) Transformação de `gender` e `bmi_class` em categórico.
      3) Criação de atributos de bioimpedância e ganho.
      4) Interações número × número.
      5) Interações categórico × número.
      6) Interações categórico × categórico.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        new_cols = {}

        # Demográficos: age bins, categorias
        age_bins = [0, 20, 30, 40, 50, 200]
        age_labels = ["0-20", "20-30", "30-40", "40-50", "50+"]
        new_cols["age_bin"] = pd.cut(
            df["age"], bins=age_bins, labels=age_labels, right=False
        ).astype("category")
        df["gender"] = df["gender"].astype("category")
        df["bmi_class"] = df["bmi_class"].astype("category")

        # Bioimpedância
        def categorize_pbf(pbf):
            if pbf < 15:
                return "Low"
            if pbf < 25:
                return "Normal"
            return "High"

        def categorize_smi(smi):
            if smi < 7:
                return "Low"
            if smi < 10:
                return "Normal"
            return "High"

        new_cols["pbf_cat"] = df["pbf"].apply(categorize_pbf).astype("category")
        new_cols["smi_cat"] = df["smi"].apply(categorize_smi).astype("category")
        new_cols["ffm_weight_pct"] = df["ffm"] / df["weight"]
        new_cols["tbw_weight_pct"] = df["tbw"] / df["weight"]
        new_cols["smm_weight_pct"] = df["smm"] / df["weight"]
        new_cols["bmr_per_kg"] = df["bmr"] / df["weight"]

        # Ganho
        def categorize_gain_pct(p):
            if p < 33:
                return "Low"
            if p < 66:
                return "Medium"
            return "High"

        new_cols["gain_pct_cat"] = (
            df["percentile_gain"].apply(categorize_gain_pct).astype("category")
        )
        new_cols["gain_var_ratio"] = df["gain_std"] / (df["gain_mean"] + 1e-6)
        new_cols["gain_mean_per_kg"] = df["gain_mean"] / df["weight"]

        # Combina todas colunas numéricas
        numeric_cols = [
            "age",
            "height",
            "weight",
            "bmi",
            "smm",
            "bfm",
            "ffm",
            "tbw",
            "ffmi",
            "bmr",
            "pbf",
            "smi",
            "tbwDffm",
            "ffm_weight_pct",
            "tbw_weight_pct",
            "smm_weight_pct",
            "bmr_per_kg",
            "gain_var_ratio",
            "gain_mean_per_kg",
        ]
        aux_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Interações numérico × numérico
        for c1, c2 in itertools.combinations(numeric_cols, 2):
            new_cols[f"{c1}_x_{c2}"] = aux_df[c1] * aux_df[c2]

        # Interações categórico × numérico
        cat_cols = [
            "bmi_class",
            "gender",
            "pbf_cat",
            "smi_cat",
            "age_bin",
            "gain_pct_cat",
        ]
        cat_codes = {}
        for cat in cat_cols:
            codes = pd.factorize(aux_df[cat])[0]
            cat_codes[cat] = codes
            new_cols[f"{cat}_code"] = codes
            for num in numeric_cols:
                new_cols[f"{cat}_code_x_{num}"] = codes * aux_df[num]

        # Interações categórico × categórico
        for c1, c2 in itertools.combinations(cat_cols, 2):
            combined = aux_df[c1].astype(str) + "_" + aux_df[c2].astype(str)
            new_cols[f"{c1}_AND_{c2}"] = combined.astype("category")

        # Concatena e retorna
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
