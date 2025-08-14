from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd

ROOT = Path("/home/g-brandao/workspace/crf-data")
V3_SUBFOLDER = "v3"
METADATA_SUBFOLDER = "metadata"
MAXIMAL_FILENAMES = {"maximal", "maximal_bk"}


def list_metadata_parquets(root=ROOT, metadata_subfolder=METADATA_SUBFOLDER):
    p = root / metadata_subfolder
    if not p.exists():
        raise FileNotFoundError(f"Metadata folder not found: {p}")
    return sorted(list(p.glob("*.parquet")))


def load_all_metadata(meta_files):
    maximal_df = None
    maximal_bk_df = None
    metadata = None  # vai ser o resultado do merge sucessivo dos arquivos não-maximal

    for fp in meta_files:
        stem = fp.stem.lower()
        if stem == "maximal":
            maximal_df = pd.read_parquet(fp)
            continue
        if stem == "maximal_bk":
            maximal_bk_df = pd.read_parquet(fp)
            continue

        df = pd.read_parquet(fp)

        # merge incremental
        if metadata is None:
            metadata = df.copy()
        else:
            metadata = metadata.merge(df, on="idx", how="outer", suffixes=("", f"_{stem}"))

    # se nada foi carregado, garante DataFrame com col 'idx'
    if (metadata is None) or (maximal_bk_df is None) or (maximal_df is None):
        raise ValueError("Metadata not found, merged metadata files returned None.")
    maximal = maximal_df.merge(
        maximal_bk_df, how="outer", on="sid", suffixes=("_running", "_cycling")
    )
    metadata["sid"] = metadata["idx"].str.split("-").str[0]
    merged_metadata = metadata.merge(maximal, on="sid", how="left", suffixes=("", "_maximal"))

    # salvar para inspeção e retornar
    merged_metadata.to_csv("merged_metadata.csv", index=False)
    return merged_metadata


def filter_by_protocol(meta, pc_list):
    if not isinstance(pc_list, list):
        pc_list = list(pc_list)
    meta_ = meta.copy()
    meta_["protocol"] = meta_["idx"].str.split("-").str[1]
    meta__filt = meta_.query("protocol in @pc_list").drop("protocol", axis=1)
    if meta__filt.empty:
        raise ValueError(f"Protocols {pc_list} not found on metadata files.")
    return meta__filt


def extract_hr_from_ts(path, hr_cols):
    ts = pd.read_csv(path)
    for col in hr_cols:
        if col in ts:
            return np.array(ts[col])
    logger.info(
        f"Cols {hr_cols} not found on idx {path.stem}, returning empty array. Possible hr cols are {[col for col in ts.columns.tolist() if 'hr' in col]}"
    )
    return np.array([])


def extract_speed_from_ts(path, speed_cols):
    ts = pd.read_csv(path)
    for col in speed_cols:
        if col in ts:
            return np.array(ts[col])
    logger.info(
        f"Cols {speed_cols} not found on idx {path.stem}, returning empty array. Possible speed cols are {[col for col in ts.columns.tolist() if 'speed' in col]}"
    )
    return np.array([])


def merge_metadata_and_series(
    pc_list,
    hr_cols,
    speed_cols,
    root=ROOT,
    v3_subfolder=V3_SUBFOLDER,
    metadata_subfolder=METADATA_SUBFOLDER,
    keep_series_in_memory=False,
):
    meta_files = list_metadata_parquets(root, metadata_subfolder)
    logger.info(f"Found {len(meta_files)} metadata files.")
    meta = load_all_metadata(meta_files)
    if meta.empty:
        raise ValueError("No metadata files loaded. Verify .parquet files on the metadata folder")
    meta_filt = filter_by_protocol(meta, pc_list)
    logger.info(
        f"All metadata was merged and filtered using {pc_list}. {len(meta)} -> {len(meta_filt)} lines."
    )
    v3_root = root / v3_subfolder
    if not v3_root.exists():
        raise FileNotFoundError(f"V3 folder not found: {v3_root}")
    meta_filt["_ts_path"] = None
    meta_filt["_selected_hr_col"] = None
    meta_filt["_selected_speed_col"] = None
    if keep_series_in_memory:
        meta_filt["hr_series"] = None
        meta_filt["speed_series"] = None
    csv_paths = list(
        path for path in v3_root.rglob("*.csv") if path.stem in meta_filt.idx.tolist()
    )
    timeseries_data = []
    for path in csv_paths:
        aux = {}
        aux["idx"] = path.stem
        aux["hr"] = extract_hr_from_ts(path, hr_cols)
        aux["speed"] = extract_speed_from_ts(path, speed_cols)
        timeseries_data.append(aux)
    timeseries_df = pd.DataFrame(timeseries_data)
    raw_data = meta_filt.merge(timeseries_df, how="left")
    logger.info(
        f"Raw data concluded with {len(raw_data)} rows. {len(raw_data.dropna(subset=('hr', 'speed')))} rows have speed and hr series."
    )
    return raw_data


if __name__ == "__main__":
    pc_list = ["obk1"]
    hr_candidates = [
        "cola-gw6_polar_hr",
        "cola-gw6_vo2max_hr",
        "cola-gw5_polar_hr",
        "cola-gw5_vo2max_hr",
    ]
    speed_candidates = [
        "cola-gw6_gps_speed",
        "cola-gw6_vo2max_gps-speed",
        "cola-gw5_gps_speed",
        "cola-gw5_vo2max_gps-speed",
    ]
    raw_data = merge_metadata_and_series(pc_list, hr_candidates, speed_candidates)
    raw_data.to_csv("raw_data.csv")
