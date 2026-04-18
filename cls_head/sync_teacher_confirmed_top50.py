from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"D:\New_python_project\MCG_diagnosis\cls_head")
REVIEW_COPY_CSV = BASE_DIR / "xinshuai_teacher_review_copy.csv"
CHANGE_LOG_CSV = BASE_DIR / "label_xinshuai_teacher_confirmed_top50_changes.csv"
OUTPUT_CSV = BASE_DIR / "xinshuai_teacher_review_copy.csv"


def main():
    review_df = pd.read_csv(REVIEW_COPY_CSV)
    change_df = pd.read_csv(CHANGE_LOG_CSV)

    change_df["subject"] = change_df["subject"].astype(int)
    review_df["subject"] = review_df["subject"].astype(int)
    for column in ["teacher_confirm_status", "teacher_notes"]:
        review_df[column] = review_df[column].astype("object")

    change_map = {
        int(row.subject): {
            "new_xinshuai": int(row.new_xinshuai),
            "reason": str(row.teacher_review_reason),
        }
        for row in change_df.itertuples()
    }

    updated = 0
    for idx, row in review_df.iterrows():
        subject = int(row["subject"])
        if subject not in change_map:
            continue

        change = change_map[subject]
        new_xinshuai = change["new_xinshuai"]
        new_binary = 1 if new_xinshuai == 1 else 0

        review_df.at[idx, "teacher_confirm_status"] = "confirmed_by_teacher"
        review_df.at[idx, "teacher_confirm_binary_label"] = new_binary
        review_df.at[idx, "teacher_confirm_xinshuai_value"] = new_xinshuai
        review_df.at[idx, "teacher_notes"] = f"Confirmed in top50 batch: {change['reason']}"
        updated += 1

    review_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Synced teacher confirmations to: {OUTPUT_CSV}")
    print(f"Updated rows: {updated}")


if __name__ == "__main__":
    main()
