# ---------------- Pipeline Integration ----------------
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from data.augmentation import augment_workout
from data.config import DataLoadConfig
from data.protocols import SessionLoader, SignalCollector, TargetCollector
from matplotlib.backends.backend_pdf import PdfPages

# Load drop list
droplist = yaml.safe_load(open("conf/droplist.yaml"))["drop"]


class DataLoadPipeline:
    def __init__(
        self,
        cfg: DataLoadConfig,
        session_loader: SessionLoader,
        signal_collectors: List[SignalCollector],
        target_collector: TargetCollector,
    ):
        self.cfg = cfg
        self.loader = session_loader
        self.signals = signal_collectors
        self.target = target_collector

    def run(self) -> pd.DataFrame:
        records = []
        for session in self.loader.load_sessions():
            idx = session.get_data("model")["identification"]["idx"][0]
            if getattr(self.cfg, "droplist", False) and idx in droplist:
                continue
            sid = session.get_data("model")["identification"]["sid"][0]
            # Collect static metadata
            base_meta: Dict[str, Any] = {"sid": sid, "idx": idx}
            # Scalars
            for coll in self.signals:
                result = coll.collect(session, sid, idx)
                if isinstance(result, dict):
                    base_meta.update(result)
                elif not isinstance(result, pd.Series):
                    base_meta[getattr(coll, "label", "")] = result
            # Targets
            target_data = self.target.collect(sid, session, base_meta)
            # Time series signals
            hr_series = None
            speed_series = None
            for coll in self.signals:
                result = coll.collect(session, sid, idx)
                label = getattr(coll, "label", "")
                if "hr" in label and isinstance(result, pd.Series):
                    hr_series = result.tolist()
                if "speed" in label and isinstance(result, pd.Series):
                    speed_series = result.tolist()
            # Variants
            variants = []
            if hr_series is not None and speed_series is not None:
                variants.append((0, hr_series, speed_series))
                if getattr(self.cfg, "augmentation", False):
                    n_aug = getattr(self.cfg, "augmentation_n", 1)
                    for i in range(n_aug):
                        print(i)
                        seed = (getattr(self.cfg, "augmentation_seed", 0) or 0) + i
                        hr_aug, sp_aug = augment_workout(
                            hr_series,
                            speed_series,
                            seed=seed,
                        )
                        variants.append((i + 1, hr_aug.tolist(), sp_aug.tolist()))
            # Append rows
            for variant_id, hr_s, sp_s in variants:
                rec = {
                    **base_meta,
                    **target_data,
                    "variant_id": variant_id,
                    "hr": hr_s,
                    "speed": sp_s,
                }
                records.append(rec)
        # Create DataFrame
        df = pd.DataFrame(records).reset_index(drop=True)
        # Save CSV
        self.cfg.output_path.mkdir(parents=True, exist_ok=True)
        fname = f"{'_'.join(self.cfg.protocol_list)}_{self.cfg.device}_{self.cfg.exercise}_{self.cfg.labels['hr']}_{self.cfg.labels['speed']}.csv"
        df.to_csv(self.cfg.output_path / fname, index=False)
        # Generate plots PDF grouped by sid/idx
        pdf_path = self.cfg.output_path / "variants_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            for (sid, idx), group in df.groupby(["sid", "idx"]):
                plt.figure(figsize=(8, 6))
                for _, row in group.iterrows():
                    label = "orig" if row.variant_id == 0 else f"aug{row.variant_id}"
                    plt.plot(row.hr, label=f"HR_{label}", alpha=0.8)
                    plt.plot(
                        row.speed,
                        label=f"SPD_{label}",
                        linestyle="--",
                        alpha=0.8,
                    )
                plt.title(f"Session {sid}-{idx} Variants")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        print(f"Saved variants plots to {pdf_path}")
        return df
