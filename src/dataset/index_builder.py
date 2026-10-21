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
                }
            )
    return records


def parse_uaspeech(ua_root: Path) -> list:
    records = []
    if not ua_root.exists():
        logger.warning(f"UA-Speech root not found at {ua_root}")
        return records

    # UA-Speech directory layout: audio / <speaker> / <block> / *.wav
    for root, _, files in os.walk(ua_root):
        for file in files:
            if file.endswith(".wav"):
                wav_path = Path(root) / file
                parts = wav_path.parts

                speaker_id = "UNKNOWN"
                block = "B1"
                for part in parts:
                    if part.startswith(("M", "F")) and len(part) in (3, 4):
                        speaker_id = part
                    if part in ("B1", "B2", "B3"):
                        block = part

                # Transcript is derived from word ID embedded in filename (e.g. M04_B1_UW10_M4.wav)
                filename_stem = wav_path.stem
                raw_text = filename_stem.split("_")[2] if "_" in filename_stem else filename_stem

                is_dys = not speaker_id.startswith("C")

                records.append(
                    {
                        "audio_filepath": str(wav_path.resolve()),
                        "transcript": raw_text,
                        "clean_transcript": clean_transcript(raw_text),
                        "speaker_id": speaker_id,
                        "dataset": "UA-Speech",
                        "is_dysarthric": is_dys,
                        "block": block,
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

    df_master = df_master[df_master["clean_transcript"] != ""]

    out_csv = Path(cfg["master_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(out_csv, index=False)
    logger.info(
        f"Master index compiled successfully with {len(df_master)} total samples -> {out_csv}"
    )


if __name__ == "__main__":
    build_master_index()