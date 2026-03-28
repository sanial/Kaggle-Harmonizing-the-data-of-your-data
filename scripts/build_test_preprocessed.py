"""
Build outputs/test_preprocessed.csv with exactly 1,659 rows.

Uses the identical PubText-loading pipeline from 02_preprocessing.ipynb,
applied to data/TestPubText/ and joined to data/SampleSubmission.csv.

Output columns: ID, PXD, pub_text
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR          = ROOT / "data"
TEST_PUBTEXT_DIR  = DATA_DIR / "TestPubText"
SUBMISSION_PATH   = DATA_DIR / "SampleSubmission.csv"
OUTPUT_DIR        = ROOT / "outputs"

# Identical section ordering used in 02_preprocessing.ipynb
PUBTEXT_SECTIONS = ["TITLE", "ABSTRACT", "INTRO", "METHODS", "RESULTS", "DISCUSS", "FIG", "SUPPL"]


def load_pubtext(folder: Path) -> pd.DataFrame:
    records = []
    for jp in sorted(folder.glob("*_PubText.json")):
        pxd = jp.stem.replace("_PubText", "")
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  SKIP {jp.name}: {exc}")
            continue

        if isinstance(payload, dict):
            parts = []
            for sec in PUBTEXT_SECTIONS:
                val = payload.get(sec, "")
                if val and isinstance(val, str):
                    parts.append(val.strip())
            # Fallback: any remaining string values
            if not parts:
                parts = [str(v) for v in payload.values() if isinstance(v, str) and v.strip()]
            text = " ".join(parts)
        elif isinstance(payload, list):
            text = " ".join(str(item) for item in payload)
        else:
            text = str(payload)

        records.append({"PXD": pxd, "pub_text": text.strip()})
    return pd.DataFrame(records)


def main():
    # Step 1: load SampleSubmission for ID + PXD
    sub = pd.read_csv(SUBMISSION_PATH, usecols=["ID", "PXD"])
    print(f"SampleSubmission: {sub.shape[0]} rows, unique PXDs: {sub['PXD'].nunique()}")
    print("Test PXDs:", sorted(sub["PXD"].unique()))

    # Step 2: load TestPubText JSONs with identical pipeline
    pubtext_df = load_pubtext(TEST_PUBTEXT_DIR)
    print(f"\nLoaded {len(pubtext_df)} PubText docs for test")
    print("PubText PXDs:", sorted(pubtext_df["PXD"].tolist()))

    # Step 3: merge on PXD (left join to preserve all 1,659 rows)
    test_df = sub.merge(pubtext_df, on="PXD", how="left")
    missing = test_df["pub_text"].isna().sum()
    if missing > 0:
        print(f"\nWARNING: {missing} rows have no pub_text after merge (PXD mismatch?)")
        print("Affected PXDs:", test_df.loc[test_df["pub_text"].isna(), "PXD"].unique().tolist())
    else:
        print(f"\nAll {test_df.shape[0]} rows have pub_text — merge successful.")

    # Step 4: keep only the required columns and save
    test_df = test_df[["ID", "PXD", "pub_text"]]
    out_path = OUTPUT_DIR / "test_preprocessed.csv"
    test_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape : {test_df.shape}")
    print(test_df.head(3).to_string())


if __name__ == "__main__":
    main()
