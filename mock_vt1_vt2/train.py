from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.general import extract_sid_from_index, load_model
from utils.mlflow_logging import MLflowLogger, configure_mlflow
from utils.plots import (
    plot_pred_vs_true,
    plot_residuals_by_category,
    plot_residuals_hist_kde,
    plot_residuals_vs_predicted,
)


def load_configs():
    with open("conf/project.yaml", "r") as f:
        project_cfg = yaml.safe_load(f)
    with open("conf/models.yaml", "r") as f:
        models_cfg = yaml.safe_load(f)
    return project_cfg, models_cfg


def load_data(cfg):
    # Raw data for group stats and binning
    raw_path = Path(cfg["data_load"]["output_path"])
    raw_df = pd.read_csv(list(raw_path.rglob("*.csv"))[0])

    # Train/Test splits
    in_path = Path(cfg["train"]["input"])
    X_tr = pd.read_csv(in_path / "train.csv").set_index(["idx", "variant_id"])
    y_tr = pd.read_csv(in_path / "y_train.csv").set_index(["idx", "variant_id"])
    X_te = pd.read_csv(in_path / "test.csv").set_index(["idx", "variant_id"])
    y_te = pd.read_csv(in_path / "y_test.csv").set_index(["idx", "variant_id"])

    return raw_df, X_tr, y_tr, X_te, y_te


def objective_factory(X, y, groups, estimator, param_space, selected_cols):
    def objective(trial):
        params = {}
        for hp, cfg in param_space.items():
            t = cfg.get("_type")
            if t == "uniform":
                params[hp] = trial.suggest_float(hp, cfg["low"], cfg["high"])
            elif t == "loguniform":
                params[hp] = trial.suggest_loguniform(hp, cfg["low"], cfg["high"])
            elif t == "int":
                params[hp] = trial.suggest_int(hp, cfg["low"], cfg["high"])
            elif t == "categorical":
                params[hp] = trial.suggest_categorical(hp, cfg["choices"])

        est = clone(estimator).set_params(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("est", est)])
        preds = cross_val_predict(
            pipe,
            X[selected_cols],
            y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            method="predict",
            n_jobs=-1,
        )
        return mean_absolute_error(y, preds)

    return objective


def tune_hyperparams(name, estimator, specs, X, y, groups, cols, n_trials=50):
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        load_if_exists=True,
    )
    obj = objective_factory(X, y, groups, estimator, specs["params"], cols)
    study.optimize(obj, n_trials=specs.get("n_trials", n_trials))
    return study.best_params


def log_mlflow_run(
    run_name,
    cfg,
    specs,
    model_type,
    best_params,
    pipeline,
    X_tr,
    y_tr,
    oof_preds,
    X_te,
    y_te,
    test_preds,
    raw_df,
    cols,
):
    # test_preds = np.asarray(test_preds).ravel()
    df_train_feats = X_tr[cols].copy()
    df_train_feats[cfg["general"]["target_column"]] = y_tr
    df_test_feats = X_te[cols].copy()
    df_test_feats[cfg["general"]["target_column"]] = y_te

    # Salva em CSV
    train_csv = "train_selected_features.csv"
    test_csv = "test_selected_features.csv"
    df_train_feats.to_csv(train_csv, index=True)
    df_test_feats.to_csv(test_csv, index=True)

    # Loga como artefatos em MLflow
    mlflow.log_artifact(train_csv, artifact_path="datasets")
    mlflow.log_artifact(test_csv, artifact_path="datasets")

    mlflow.log_param("model_type", model_type)
    mlflow.log_params(best_params)

    mlflow.log_param("augmented", cfg["data_load"]["augmentation"])
    if cfg["data_load"]["augmentation"]:
        mlflow.log_param("augmentation_n", cfg["data_load"]["augmentation_n"])

    # log artifacts
    for fname in ["project.yaml", "models.yaml", "splits.yaml", "droplist.yaml"]:
        mlflow.log_artifact(f"conf/{fname}", artifact_path="pipeline_config")

    mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")
    MLflowLogger.log_metrics(y_tr, oof_preds, prefix="oof")
    MLflowLogger.log_metrics(y_te, test_preds, prefix="test")

    # feature importance and preds
    MLflowLogger.log_selected_features(cols, artifact_path="selected_features")
    MLflowLogger.log_preds_by_sid(
        X_tr[cols], y_tr, oof_preds, artifact_path="predictions/oof"
    )
    MLflowLogger.log_preds_by_sid(
        X_te[cols], y_te, test_preds, artifact_path="predictions/test"
    )

    # group-wise logging
    add_group_metrics_and_plots(X_tr, y_tr, oof_preds, X_tr.index, raw_df)

    # figures
    for fig, fname in [
        (plot_pred_vs_true(y_tr, oof_preds), "oof_scatter.png"),
        (plot_residuals_vs_predicted(y_tr, oof_preds), "oof_resid_vs_pred.png"),
        (plot_residuals_hist_kde(y_tr, oof_preds), "oof_resid_kde.png"),
        (plot_pred_vs_true(y_te, test_preds), "test_scatter.png"),
        (plot_residuals_vs_predicted(y_te, test_preds), "test_resid_vs_pred.png"),
        (plot_residuals_hist_kde(y_te, test_preds), "test_resid_kde.png"),
    ]:
        MLflowLogger.log_figure(fig=fig, filename=fname)


def add_group_metrics_and_plots(X_tr, y_tr, oof_preds, index, raw_df):
    # Define bins and group columns
    sub = raw_df.set_index(["idx", "variant_id"]).loc[index].copy()
    if "age" in sub.columns:
        sub["age_bin"] = pd.cut(
            sub["age"],
            bins=[-np.inf, 20, 30, 40, 50, np.inf],
            labels=["under_20", "20-30", "30-40", "40-50", "50_plus"],
        )
    group_cols = ["gender", "bmi_class", "age_bin"]

    for col in group_cols:
        if col not in sub:
            continue
        for cat in sub[col].dropna().unique():
            mask = sub[col] == cat
            safe = str(cat).replace("+", "plus")
            MLflowLogger.log_metrics(
                y_tr[mask], oof_preds[mask], prefix=f"oof_{col}_{safe}_"
            )
            MLflowLogger.log_preds_by_sid(
                X_tr[mask],
                y_tr[mask],
                oof_preds[mask],
                artifact_path=f"predictions/{col}/{safe}",
            )
        MLflowLogger.log_figure(
            plot_residuals_by_category(y_tr, oof_preds, sub[col]),
            filename=f"resid_box_{col}.png",
        )


def main():
    project_cfg, models_cfg = load_configs()
    raw_df, X_tr, y_tr, X_te, y_te = load_data(project_cfg)
    # y_tr = y_tr.values.ravel()
    # y_te = y_te.values.ravel()
    sid_groups = extract_sid_from_index(X_tr.index)
    configure_mlflow(project_cfg["general"]["experiment_name"])

    for feat_file in Path(project_cfg["selection"]["output"]).glob("*.json"):
        feats = pd.read_json(feat_file)[0].tolist()
        for n_feats in project_cfg["train"]["feature_counts"]:
            cols = feats[:n_feats]
            for model_type in project_cfg["train"]["models"]:
                spec = models_cfg[model_type]
                estimator = load_model(spec["class_path"])
                run_id = (
                    f"{model_type}_({feat_file.stem.split('_')[0]}_SFS)_{n_feats}feats"
                )
                X_tr.to_csv("X_tr.csv")
                # Tuning
                best_params = tune_hyperparams(
                    f"{run_id}_optuna", estimator, spec, X_tr, y_tr, sid_groups, cols
                )
                print(f"Best params for {run_id}: {best_params}")

                # Final train and predict
                final_est = clone(estimator).set_params(**best_params)
                pipeline = Pipeline([("scaler", StandardScaler()), ("est", final_est)])

                oof_preds = cross_val_predict(
                    pipeline,
                    X_tr[cols],
                    y_tr,
                    cv=LeaveOneGroupOut(),
                    groups=sid_groups,
                    method="predict",
                    n_jobs=-1,
                )

                pipeline.fit(X_tr[cols], y_tr.values.ravel())
                test_preds = pipeline.predict(X_te[cols])
                with mlflow.start_run(run_name=run_id):
                    log_mlflow_run(
                        run_id,
                        project_cfg,
                        spec,
                        model_type,
                        best_params,
                        pipeline,
                        X_tr,
                        y_tr,
                        oof_preds,
                        X_te,
                        y_te,
                        test_preds,
                        raw_df,
                        cols,
                    )


if __name__ == "__main__":
    main()
