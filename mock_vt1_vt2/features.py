import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

# safe_target_encoder_logo.py
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils.general import extract_sid_from_index


class SafeTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder seguro que suporta LeaveOneGroupOut (LOGO) para gerar OOF
    durante fit (útil quando pipeline roda antes do split ou pra anti-leakage
    entre sujeitos dentro do treino). Também armazena mapeamento final do treino
    para uso no teste.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        self._oof_ = None
        self._mapping_ = {}
        self._global_mean_ = None
        self._train_index_ = None

    def _safe_fill(self, s):
        return s.where(~s.isna(), "__nan__")

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)
        self.cols = X.select_dtypes(include=["category", "object"]).columns.tolist()

        if len(self.cols) == 0:
            self._oof_ = pd.DataFrame(index=X.index)
            self._train_index_ = X.index.copy()
            self._global_mean_ = float(y.mean())
            return self

        self._global_mean_ = float(y.mean())

        # escolher splitter
        splitter = LeaveOneGroupOut()
        groups = extract_sid_from_index(X.index)

        oof_df = pd.DataFrame(
            index=X.index, columns=[f"{c}_enc" for c in self.cols], dtype=float
        )

        for col in self.cols:
            col_series = self._safe_fill(X[col].astype(object))
            col_oof = pd.Series(index=X.index, dtype=float)

            # re-create iterator (LeaveOneGroupOut returns generator-like; we will iterate normally)
            for train_idx, val_idx in splitter.split(X, y, groups=groups):
                # train_idx/val_idx são integer positions -> map para index
                train_idx = np.array(train_idx)
                val_idx = np.array(val_idx)

                train_keys = col_series.iloc[train_idx]
                y_train = y.iloc[train_idx]
                mapping = y_train.groupby(train_keys).mean()
                val_keys = col_series.iloc[val_idx]
                mapped = val_keys.map(mapping).astype(float)
                mapped = mapped.fillna(self._global_mean_)
                col_oof.iloc[val_idx] = mapped.values
            oof_df[f"{col}_enc"] = col_oof
            # mapping final com todo o treino (para transformar o teste)
            full_mapping = y.groupby(col_series).mean().to_dict()
            if "__nan__" not in full_mapping and col_series.isna().any():
                full_mapping["__nan__"] = self._global_mean_
            self._mapping_[col] = full_mapping

        self._oof_ = oof_df
        self._train_index_ = X.index.copy()
        return self

    def transform(self, X):
        X = X.copy()
        if self._train_index_ is not None and X.index.equals(self._train_index_):
            enc_df = self._oof_.reindex(X.index).copy()
        else:
            enc_df = pd.DataFrame(
                index=X.index, columns=[f"{c}_enc" for c in self.cols], dtype=float
            )
            for col in self.cols:
                col_series = self._safe_fill(X[col].astype(object))
                mapping = self._mapping_.get(col, {})
                enc_series = col_series.map(mapping).astype(float)
                enc_series = enc_series.fillna(self._global_mean_)
                enc_df[f"{col}_enc"] = enc_series

        X2 = X.drop(columns=self.cols, errors="ignore")
        return pd.concat([X2, enc_df], axis=1)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects columns based on feature groups from config"""

    def __init__(self, group_defs, enabled_groups):
        self.group_defs = group_defs
        self.enabled_groups = enabled_groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = []
        for group in self.enabled_groups:
            if group in self.group_defs:
                columns.extend(self.group_defs[group])
        # filtrar colunas existentes e preservar ordem/únicas
        selected = [c for c in columns if c in X.columns]
        seen = set()
        ordered = []
        for c in selected:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        return X[ordered]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Base class for feature engineering transformers"""

    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self


class BinningTransformer(FeatureEngineer):
    """Handles binning of numerical features"""

    BIN_RULES = {
        "age": {
            "bins": [0, 30, 50, 200],
            "labels": ["below_30", "between_30_and_49", "50_plus"],
            "right": False,
        },
        "hr_rest": {
            "bins": [0, 60, 80, 200],
            "labels": ["Low", "Normal", "High"],
            "right": False,
        },
        "pbf": {
            "bins": [0, 20, 30, 100],
            "labels": ["Healthy", "Elevated", "HighRisk"],
            "right": False,
        },
        "vo2max_myers": {
            "bins": [0, 35, 45, 100],
            "labels": ["Low", "Moderate", "Good"],
            "right": False,
        },
    }

    def transform(self, X):
        X = X.copy()
        for col, rules in self.BIN_RULES.items():
            if col in X:
                new_col = f"{col}_bin"
                X[new_col] = pd.cut(
                    X[col],
                    bins=rules["bins"],
                    labels=rules["labels"],
                    right=rules["right"],
                )
        return X


class CategoricalTransformer(FeatureEngineer):
    """Handles categorical conversions and custom categorizations"""

    CAT_RULES = {
        "gender": {"dtype": "category"},
        "bmi_class": {"dtype": "category"},
        "pbf": {
            "function": lambda x: "Low" if x < 15 else "Normal" if x < 25 else "High",
            "new_col": "pbf_cat",
        },
        "smi": {
            "function": lambda x: "Low" if x < 7 else "Normal" if x < 10 else "High",
            "new_col": "smi_cat",
        },
    }

    def transform(self, X):
        X = X.copy()
        for col, rules in self.CAT_RULES.items():
            if col in X:
                if "dtype" in rules:
                    X[col] = X[col].astype(rules["dtype"])
                elif "function" in rules:
                    X[rules["new_col"]] = (
                        X[col].apply(rules["function"]).astype("category")
                    )
        return X


class DerivedFeatureTransformer(FeatureEngineer):
    """Creates new features from existing ones"""

    DERIVED_FEATURES = {
        "ffm_weight_pct": lambda df: df["ffm"] / df["weight"],
        "tbw_weight_pct": lambda df: df["tbw"] / df["weight"],
        "smm_weight_pct": lambda df: df["smm"] / df["weight"],
        "bmr_per_kg": lambda df: df["bmr"] / df["weight"],
        "gain_var_ratio": lambda df: df["gain_std"] / (df["gain_mean"] + 1e-6),
        "gain_mean_per_kg": lambda df: df["gain_mean"] / df["weight"],
        "hr_drift": lambda df: (df["hr_mean"] - df["hr_rest"]) / df["hr_rest"],
        "hr_reserve_util": lambda df: (df["hr_mean"] - df["hr_rest"])
        / (df["age_hr_madf"] - df["hr_rest"] + 1e-6),
        "pbf_smi_ratio": lambda df: df["pbf"] / (df["smi"] + 1e-6),
        "body_composite": lambda df: df["ffm"] * df["smi"] / (df["pbf"] + 1e-6),
        "bmr_cunningham": lambda df: 500 + 22 * df["ffm"],
        "bmr_discrepancy": lambda df: df["bmr"] - df["bmr_cunningham"],
        "gain_per_ffm": lambda df: df["gain_mean"] / df["ffm"],
        "gain_per_smm": lambda df: df["gain_mean"] / df["smm"],
        "gain_per_tbw": lambda df: df["gain_mean"] / df["tbw"],
        "hr_cv": lambda df: df["hr_std"] / df["hr_mean"],
        "hr_std_per_rest": lambda df: df["hr_std"] / df["hr_rest"],
        "efficiency_index": lambda df: df["gain_mean"] / df["speed_mean"],
        "speed_per_hr": lambda df: df["speed_mean"] / df["hr_mean"],
        "fatigue_factor": lambda df: df["gain_std"] / df["gain_mean"],
        "ponderal_index": lambda df: df["weight"] / (df["height"] / 100) ** 3,
        "metabolic_balance": lambda df: df["vo2max_myers"]
        / (df["ftp_anthropometric"] + 1e-6),
        # --- novas features de frequência cardíaca ---
        "age_hr_max": lambda df: 208 - 0.7 * df["age"],
        "hr_max_std1": lambda df: df["hr_mean"] + df["hr_std"],
        "hr_max_std2": lambda df: df["hr_mean"] + 2 * df["hr_std"],
        "hr_max_std3": lambda df: df["hr_mean"] + 3 * df["hr_std"],
        "hr_reserve1": lambda df: (208 - 0.7 * df["age"]) - df["hr_rest"],
        "hr_reserve2": lambda df: (df["hr_mean"] + df["hr_std"]) - df["hr_rest"],
        "hr_reserve3": lambda df: (df["hr_mean"] + 2 * df["hr_std"]) - df["hr_rest"],
        "hr_reserve4": lambda df: (df["hr_mean"] + 3 * df["hr_std"]) - df["hr_rest"],
        "theorical_vt11": lambda df: df["hr_rest"]
        + 0.5 * ((208 - 0.7 * df["age"]) - df["hr_rest"]),
        "theorical_vt12": lambda df: df["hr_rest"]
        + 0.5 * ((df["hr_mean"] + df["hr_std"]) - df["hr_rest"]),
        "theorical_vt13": lambda df: df["hr_rest"]
        + 0.5 * ((df["hr_mean"] + 2 * df["hr_std"]) - df["hr_rest"]),
        "theorical_vt14": lambda df: df["hr_rest"]
        + 0.5 * ((df["hr_mean"] + 3 * df["hr_std"]) - df["hr_rest"]),
        "theorical_vt21": lambda df: df["hr_rest"]
        + 0.85 * ((208 - 0.7 * df["age"]) - df["hr_rest"]),
        "theorical_vt22": lambda df: df["hr_rest"]
        + 0.85 * ((df["hr_mean"] + df["hr_std"]) - df["hr_rest"]),
        "theorical_vt23": lambda df: df["hr_rest"]
        + 0.85 * ((df["hr_mean"] + 2 * df["hr_std"]) - df["hr_rest"]),
        "theorical_vt24": lambda df: df["hr_rest"]
        + 0.85 * ((df["hr_mean"] + 3 * df["hr_std"]) - df["hr_rest"]),
    }

    def transform(self, X):
        X = X.copy()
        for name, func in self.DERIVED_FEATURES.items():
            try:
                X[name] = func(X)
            except KeyError as e:
                warnings.warn(f"Skipping {name} - missing column: {str(e)}")
        return X


class OHETransformer(FeatureEngineer):
    """Handles One-Hot Encoding for specified categorical columns"""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = None
        self.categorical_columns_ = None

    def fit(self, X, y=None):
        self.categorical_columns_ = X.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        if self.categorical_columns_:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(X[self.categorical_columns_])
        return self

    def transform(self, X):
        if not self.encoder:
            return X
        ohe_array = self.encoder.transform(X[self.categorical_columns_])
        ohe_columns = self.encoder.get_feature_names_out(self.categorical_columns_)
        ohe_df = pd.DataFrame(ohe_array, columns=ohe_columns, index=X.index)
        X_transformed = X.drop(columns=self.categorical_columns_)
        return pd.concat([X_transformed, ohe_df], axis=1)


class NumericInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Creates pairwise numeric × numeric feature interactions.
    """

    def __init__(self, combine_num: bool = True):
        self.combine_num = combine_num

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.combine_num:
            return X
        num_cols = sorted(X.select_dtypes(include=np.number).columns)
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i + 1 :]:
                X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                X[f"{col1}_/_{col2}"] = X[col1] / (X[col2] + 1e-8)
        return X


class CategoricalInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Creates pairwise categorical × categorical feature interactions.
    """

    def __init__(self, combine_cat: bool = True):
        self.combine_cat = combine_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.combine_cat:
            return X
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        for i, col1 in enumerate(cat_cols):
            for col2 in cat_cols[i + 1 :]:
                new_col = f"{col1}_AND_{col2}"
                X[new_col] = X[col1].astype(str) + "_" + X[col2].astype(str)
                X[new_col] = X[new_col].astype("category")
        return X


class MixedInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Creates categorical_code × numeric feature interactions.
    """

    def __init__(self, combine_num_cat: bool = True):
        self.combine_num_cat = combine_num_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.combine_num_cat:
            return X
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        num_cols = X.select_dtypes(include=np.number).columns

        for cat_col in cat_cols:
            # Create numeric representation
            code_col = f"{cat_col}_code"
            X[code_col] = pd.factorize(X[cat_col])[0]

            # Create interactions
            for num_col in num_cols:
                X[f"{code_col}_x_{num_col}"] = X[code_col] * X[num_col]
        return X


# =====================
# Master Pipeline
# =====================
class FeaturePipeline:
    """Orchestrates feature engineering based on configuration"""

    def __init__(self, config):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Get feature groups from config
        group_defs = self.config["general"]["columns"]
        enabled_groups = [
            group
            for group, enabled in self.config["features"]["use_cols"].items()
            if enabled
        ]

        # Build pipeline steps
        steps = [("column_selector", ColumnSelector(group_defs, enabled_groups))]

        # Add feature engineering steps
        steps_cfg = self.config["features"]["steps"]
        if steps_cfg.get("binned", False):
            steps.append(("binning", BinningTransformer(self.config)))

        steps.append(("categorical", CategoricalTransformer(self.config)))

        steps.append(("derived_features", DerivedFeatureTransformer(self.config)))

        if steps_cfg.get("combine_cat", False):
            steps.append(
                ("cat_interaction", CategoricalInteractionTransformer(self.config))
            )

        if steps_cfg.get("target_encoding", False):
            steps.append(("target_encoding", SafeTargetEncoder()))

        if steps_cfg.get("combine_num", False):
            steps.append(
                ("num_interaction", NumericInteractionTransformer(self.config))
            )

        if steps_cfg.get("combine_num_cat", False):
            steps.append(
                ("mixed_interaction", MixedInteractionTransformer(self.config))
            )

        if steps_cfg.get("ohe", False):
            steps.append(("ohe", OHETransformer(self.config)))

        return Pipeline(steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("conf/project.yaml"))
    gen = cfg["general"]
    ft = cfg["features"]
    input_path = Path(ft["input"])
    output_path = Path(ft["output"])
    splits_path = Path(gen["splits_path"])
    proc = cfg["processing"]
    dl = cfg["data_load"]
    target_col = gen["target_column"]

    stem = f"{'_'.join(dl['protocol_list'])}_{dl['device']}_{dl['exercise']}_{dl['labels']['hr']}_{dl['labels']['speed']}"

    interim_data = (
        pd.read_csv(input_path / f"{stem}.csv")
        .set_index(["idx", "variant_id"])
        .dropna(subset=target_col)
    )
    if ft.get("dropna", False):
        interim_data = interim_data.dropna()

    # --- carregar splits antes do fit ---
    splits = yaml.safe_load(open(splits_path))
    split_idxs = [idx for _, v in splits.items() for idx in v]

    # máscaras baseadas no índice atual (idx, variant_id)
    test_mask = interim_data.index.get_level_values("idx").isin(split_idxs) & (
        interim_data.index.get_level_values("variant_id") == 0
    )
    train_mask = ~interim_data.index.get_level_values("idx").isin(split_idxs)

    # criar dataframes de treino/test mantendo idx como coluna (necessário para LOGO)
    train_raw = interim_data[train_mask]
    test_raw = interim_data[test_mask]

    # separar y
    y_train = train_raw[target_col]
    y_test = test_raw[target_col]

    # construir pipeline (assegure que o SafeTargetEncoder esteja instanciado com group_col='idx' e use_logo=True)
    feature_pipeline = FeaturePipeline(config=cfg)
    # fit apenas no conjunto de treino
    feature_pipeline.fit(train_raw, y_train)

    # transformar train e test (transform NÃO deve ser chamado com fit no test)
    features_train = feature_pipeline.transform(train_raw)
    features_test = feature_pipeline.transform(test_raw)

    # Preserve feature names (do train)
    feature_names = list(features_train.columns)

    # salvar
    output_path = Path(ft["output"])
    output_path.mkdir(parents=True, exist_ok=True)

    features_test.to_csv(output_path / "test.csv", index=True)
    pd.DataFrame(y_test).to_csv(output_path / "y_test.csv", index=True)

    features_train.to_csv(output_path / "train.csv", index=True)
    pd.DataFrame(y_train).to_csv(output_path / "y_train.csv", index=True)

    with open(output_path / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

