import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.general import load_model

# Carrega configuração do pipeline
cfg = yaml.safe_load(open("conf/project.yaml"))

sel = cfg["selection"]
gen = cfg["general"]
ft = cfg["features"]
dl = cfg["data_load"]

input_path = Path(sel["input"])
output_path = Path(sel["output"])
n_feats = sel["n_feats"]
models = sel["models"]
target_col = gen["target_column"]
model_path = Path(gen["models_path"])
use_cols = ft["use_cols"]
pc = dl["protocol_list"]

# Parâmetros do Instability Filter
B = sel.get("bootstrap_iters", 50)  # número de iterações bootstrap
stable_M = sel.get("stable_n_feats", 50)  # quantas features manter após estabilidade

# Carrega dados
train_fe = (
    pd.read_csv(input_path / "train.csv").set_index(["idx", "variant_id"]).dropna()
)
test_fe = pd.read_csv(input_path / "test.csv").set_index(["idx", "variant_id"]).dropna()
y_train = (
    pd.read_csv(input_path / "y_train.csv").set_index(["idx", "variant_id"]).dropna()
)
y_test = (
    pd.read_csv(input_path / "y_test.csv").set_index(["idx", "variant_id"]).dropna()
)

# Combina e prepara arrays
combined_train = pd.concat([train_fe, y_train], axis=1).dropna()
train_fe = combined_train.drop(columns=y_train.columns)
y_train = combined_train[y_train.columns].to_numpy().ravel()

combined_test = pd.concat([test_fe, y_test], axis=1).dropna()
test_fe = combined_test.drop(columns=y_test.columns)
y_test = combined_test[y_test.columns].to_numpy().ravel()

# ================================
# Instability Filter: Lasso + Bootstrap
# ================================
print(f"Executando Instability Filter: B={B}, mantendo top {stable_M} features...")

# Padronizador único (usado em cada bootstrap)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_fe)
feature_names = train_fe.columns.tolist()

# contador de seleções
counts = pd.Series(0, index=feature_names)

for b in range(B):
    # amostra bootstrap de índices
    idx_boot = np.random.choice(a=len(X_scaled), size=len(X_scaled), replace=True)
    Xb = X_scaled[idx_boot]
    yb = y_train[idx_boot]

    # LassoCV interno (5-fold) para escolher alpha
    lasso = LassoCV(cv=5, n_jobs=-1, random_state=b, max_iter=10_000, tol=1e-3)
    lasso.fit(Xb, yb)

    # features com coef != 0
    sel_feats = np.array(feature_names)[np.abs(lasso.coef_) > 1e-6]
    counts.loc[sel_feats] += 1

# calcula frequência e ordena
freq = (counts / B).sort_values(ascending=False)
top_stable = freq.iloc[:stable_M].index.tolist()

print("Top features mais estáveis (freq):")
for feat, f in freq.iloc[:stable_M].items():
    print(f"  {feat}: {f:.2f}")

# Reduce os datasets às features estáveis
train_fe = train_fe[top_stable]
test_fe = test_fe[top_stable]

# ================================
# Seleção final com SFS + MLflow Tracking
# ================================
model_cfg = yaml.safe_load(open(model_path))

estimators = {
    name: load_model(spec["class_path"])
    for name, spec in model_cfg.items()
    if name in models
}

for model_name, estimator in estimators.items():
    print(f"\n[{model_name}] Selecionando {n_feats} feats com SFS...")
    pipe = Pipeline([("scaler", StandardScaler()), ("est", estimator)])
    sfs = SFS(
        pipe,
        k_features=n_feats,
        forward=True,
        floating=False,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )
    sfs = sfs.fit(train_fe, y_train)

    # Extrai ordem de seleção
    ordem = []
    sel_prev = set()
    for k in range(1, n_feats + 1):
        names_k = set(sfs.subsets_[k]["feature_names"])
        novo = (names_k - sel_prev).pop()
        ordem.append(novo)
        sel_prev = names_k

    print(f"{model_name} selecionou (em ordem):")
    for i, feat in enumerate(ordem, 1):
        print(f" {i:02d}. {feat}")

    # Salva ordem em JSON
    tags = [k for k, v in use_cols.items() if v]
    out_file = (
        output_path / f"{model_name}_{target_col}_{'_'.join(pc)}_{n_feats}feats_"
        f"stable{stable_M}_sfs_{'_'.join(tags)}.json"
    )
    with open(out_file, "w") as f:
        json.dump(ordem, f)
    print(f"Ordem salva em: {out_file}")
