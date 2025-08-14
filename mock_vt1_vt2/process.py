# src/features/pipeline.py
import sys
from pathlib import Path

sys.path.append("/home/g-brandao/workspace/crf-sandbox/experiments")
import pandas as pd
import yaml
from process.pipeline import Pipeline
from process.stages import (
    AnthroFeatureStage,
    CalculatedBiaStage,
    HRFeatureStage,
    MergeEcgStage,
    ReadRawStage,
    RelativeMetricsStage,
    SaveFeaturesStage,
    SpeedMomentumStage,
    TimeSeriesFeatureStage,
)

cfg = yaml.safe_load(open("conf/project.yaml"))
proc = cfg["processing"]
dl = cfg["data_load"]
gen = cfg["general"]
ecg_path = gen["ecg_path"]
ecg_cols = gen["columns"]["ecg"]
ecg_df = pd.read_csv(ecg_path)
stem = f"{'_'.join(dl['protocol_list'])}_{dl['device']}_{dl['exercise']}_{dl['labels']['hr']}_{dl['labels']['speed']}"

input_csv = Path(proc["input"]) / f"{stem}.csv"
output_csv = Path(proc["output"]) / f"{stem}.csv"
pipeline = Pipeline(
    [
        ReadRawStage(input_csv),
        TimeSeriesFeatureStage(algorithm=proc["algorithm"]),
        AnthroFeatureStage(),
        HRFeatureStage(),
        RelativeMetricsStage(),
        SpeedMomentumStage(),
        CalculatedBiaStage(),
        MergeEcgStage(ecg_df, ecg_cols),
        SaveFeaturesStage(output_csv),
    ]
)
df_final = pipeline.run()
print(f"[✔] Geradas {len(df_final)} linhas em {output_csv}")
