import json
import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)


def configure_mlflow(
    experiment_name: str,
    tracking_uri: str = "http://srbr-mlflow.la.corp.samsungelectronics.net:5000",
):
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_ARTIFACT_URI", None)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    # mlflow.sklearn.autolog()


class MLflowLogger:
    """
    Utility class for logging metrics and artifacts to MLflow in a structured and DRY manner,
    agora suportando índices simples e MultiIndex.
    """

    @staticmethod
    def _log_temp_file(
        content: bytes,
        suffix: str,
        artifact_path: str,
        mode: str = "wb",
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode=mode) as tmp:
            tmp.write(content)
            tmp.flush()
            mlflow.log_artifact(tmp.name, artifact_path=artifact_path)
            os.unlink(tmp.name)

    @classmethod
    def log_json_artifact(
        cls,
        data: Any,
        artifact_path: str,
        filename: str = "data.json",
    ) -> None:
        serialized = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        cls._log_temp_file(serialized, suffix=".json", artifact_path=artifact_path)

    @classmethod
    def log_df_artifact(
        cls,
        df: pd.DataFrame,
        artifact_path: str,
        filename: str = "data.csv",
    ) -> None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        cls._log_temp_file(csv_bytes, suffix=".csv", artifact_path=artifact_path)

    @staticmethod
    def log_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str | None = None,
    ) -> None:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        expl_var = explained_variance_score(y_true, y_pred)
        msle = mean_squared_log_error(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        smape = 100 * np.mean(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
        )
        metrics = {
            f"{prefix + '_'}rmse" if prefix else "rmse": rmse,
            f"{prefix + '_'}mae" if prefix else "mae": mae,
            f"{prefix + '_'}r2" if prefix else "r2": r2,
            f"{prefix + '_'}mape" if prefix else "mape": mape,
            f"{prefix + '_'}medae" if prefix else "medae": medae,
            f"{prefix + '_'}max_error" if prefix else "max_error": max_err,
            f"{prefix + '_'}expl_var" if prefix else "expl_var": expl_var,
            f"{prefix + '_'}msle" if prefix else "msle": msle,
            f"{prefix + '_'}pearson" if prefix else "pearson": pearson_corr,
            f"{prefix + '_'}spearman" if prefix else "spearman": spearman_corr,
            f"{prefix + '_'}smape" if prefix else "smape": smape,
        }

        # logando no MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

    @staticmethod
    def _extract_sid(idx: Any) -> Any:
        """
        Extrai o subject ID:
        - se for tuple (MultiIndex), retorna o 1º elemento;
        - senão, retorna diretamente.
        """
        if isinstance(idx, tuple):
            return idx[0]
        return idx

    @staticmethod
    def _extract_variant_id(idx: Any) -> Any:
        """
        Se for tuple (MultiIndex), retorna o 2º elemento (variant_id);
        senão, retorna None.
        """
        if isinstance(idx, tuple) and len(idx) > 1:
            return idx[1]
        return None

    @classmethod
    def log_preds_by_sid(
        cls,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        artifact_path: str,
    ) -> None:
        """
        Constrói um DataFrame com colunas:
          - sid
          - variant_id (pode vir como NaN se índice simples)
          - idx (string repr)
          - actual, predicted
        e loga como CSV.
        """
        records = []
        for idx_val, actual, pred in zip(X.index, y_true, y_pred):
            print(idx_val)
            print(actual)
            print(pred)
            sid = cls._extract_sid(idx_val)
            var_id = cls._extract_variant_id(idx_val)
            records.append(
                {
                    "sid": sid,
                    "variant_id": var_id,
                    "idx": idx_val,
                    "actual": actual,
                    "predicted": pred,
                }
            )

        df = pd.DataFrame.from_records(records)
        cls.log_df_artifact(df, artifact_path)

    @staticmethod
    def log_figure(fig, artifact_path: str = "plots", filename: str = "plot.png"):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, filename)
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(file_path, artifact_path=artifact_path)

    @classmethod
    def log_selected_features(
        cls,
        selected: list[str],
        artifact_path: str = "selected_features",
    ) -> None:
        cls.log_json_artifact(selected, artifact_path)

    @classmethod
    def log_pipeline_outputs(
        cls,
        pipeline: Any,
        X: pd.DataFrame,
        prefix: str = "",
    ) -> None:
        current_X = X.copy()
        for step_name, step in pipeline.named_steps.items():
            if hasattr(step, "transform"):
                current_X = step.transform(current_X)
                arr = (
                    current_X.toarray()
                    if hasattr(current_X, "toarray")
                    else (
                        current_X
                        if isinstance(current_X, np.ndarray)
                        else np.asarray(current_X)
                    )
                )
                df_out = pd.DataFrame(arr)
                cls.log_df_artifact(df_out, artifact_path=f"{prefix}{step_name}_output")

    @classmethod
    def log_data_split(
        cls,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("X_val_shape", X_val.shape)
        mlflow.log_param("y_train_shape", y_train.shape)
        mlflow.log_param("y_val_shape", y_val.shape)

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": pd.DataFrame(y_train, columns=["target"]),
            "y_val": pd.DataFrame(y_val, columns=["target"]),
        }
        for name, df in splits.items():
            cls.log_df_artifact(df, artifact_path=f"data_splits/{name}")
