import os
from pathlib import Path
import pandas as pd
import yaml
from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_transcript

logger = get_logger("IndexBuilder")


def parse_ljspeech(lj_root: Path) -> list:
    records = []
    meta_file = lj_root / "metadata.csv"
    if not meta_file.exists():
        logger.warning(f"LJSpeech metadata not found at {meta_file}")
        return records

    df = pd.read_csv(
        meta_file,
        sep="|",
        header=None,
        names=["ID", "raw_transcript", "normalized_transcript"],
        on_bad_lines="skip",
    )
    for _, row in df.iterrows():
        wav_path = lj_root / "wavs" / f"{row['ID']}.wav"
        if wav_path.exists():
            records.append(
                {
                    "audio_filepath": str(wav_path.resolve()),
                    "transcript": str(row["normalized_transcript"]),
                    "clean_transcript": clean_transcript(
                        str(row["normalized_transcript"])
                    ),
                    "speaker_id": "LJ01",
                    "dataset": "LJSpeech",
                    "is_dysarthric": False,
                    "block": "train",
                    "severity": "control",
                }
            )
    return records


def parse_uaspeech(ua_root: Path) -> list:
    records = []
    if not ua_root.exists():
        logger.warning(f"UA-Speech root not found at {ua_root}")
        return records

    # Look for the metadata CSVs directly in the UA-Speech directory
    metadata_files = {
        "train": ua_root / "train_metadata.csv",
        "test": ua_root / "test_metadata.csv",
    }

    for split_name, csv_path in metadata_files.items():
        if not csv_path.exists():
            logger.warning(f"Metadata file not found at {csv_path}")
            continue

        logger.info(f"Parsing {split_name} metadata: {csv_path}")
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            speaker = str(row["speaker"])
            audio_filename = str(row["audio_file"])

            # Resolve path whether audio_file is relative to ua_root or speaker folder
            possible_paths = [
                ua_root / audio_filename,
                ua_root / speaker / audio_filename,
                ua_root / speaker / split_name / audio_filename,
            ]

            wav_path = None
            for p in possible_paths:
                if p.exists():
                    wav_path = p
                    break

            if wav_path is None:
                continue

            raw_text = str(row["text"])
            severity = str(row.get("severity", "unknown")).lower()
            is_dys = not speaker.startswith("C") and severity != "control"

            records.append(
                {
                    "audio_filepath": str(wav_path.resolve()),
                    "transcript": raw_text,
                    "clean_transcript": clean_transcript(raw_text),
                    "speaker_id": speaker,
                    "dataset": "UA-Speech",
                    "is_dysarthric": is_dys,
                    "block": split_name,
                    "severity": severity,
                }
            )

    return records


def build_master_index(config_path: str = "configs/data_config.yaml") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["dataset"]

    ljspeech_records = parse_ljspeech(Path(cfg["ljspeech_root"]))
    uaspeech_records = parse_uaspeech(Path(cfg["uaspeech_root"]))

    all_records = ljspeech_records + uaspeech_records
    df_master = pd.DataFrame(all_records)

    # Filter out empty clean transcripts
    df_master = df_master[df_master["clean_transcript"] != ""]

    out_csv = Path(cfg["master_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(out_csv, index=False)
    logger.info(
        f"Master index compiled successfully with {len(df_master)} total samples -> {out_csv}"
    )


if __name__ == "__main__":
    build_master_index()