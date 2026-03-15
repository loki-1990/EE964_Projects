import os
from tqdm import tqdm
def download_pose_files_from_gcs(fs, LOCAL_POSE_DIR, POSE_PREFIX, df_clean=None, needed_ids: list = None):
    os.makedirs(str(LOCAL_POSE_DIR), exist_ok=True)

    POSE_PREFIX = POSE_PREFIX.strip("/")

    if df_clean is None and needed_ids is None:
        raise ValueError("Either df_clean or needed_ids must be provided")

    if df_clean is not None:
        inferred_ids = set(df_clean["word_id"]) | set(df_clean["sentence_id"])

        if needed_ids is None:
            needed_ids = sorted(inferred_ids)
        else:
            needed_ids = sorted(set(needed_ids))
            assert set(needed_ids) == inferred_ids, "Mismatch between df_clean and needed_ids"

    else:
        needed_ids = sorted(set(needed_ids))

    print("Total pose files needed:", len(needed_ids))

    downloaded, missing, forbidden = 0, 0, 0

    for uid in tqdm(needed_ids):
        gcs_path = f"{POSE_PREFIX}/{uid}.pose"
        local_path = os.path.join(str(LOCAL_POSE_DIR), f"{uid}.pose")

        if os.path.exists(local_path):
            continue

        try:
            fs.info(gcs_path)
        except FileNotFoundError:
            missing += 1
            continue
        except OSError as e:
            forbidden += 1
            if forbidden <= 5:
                print(f"[FORBIDDEN] {gcs_path} -> {e}")
            continue

        fs.get(gcs_path, local_path)
        downloaded += 1

    print("Downloaded now:", downloaded)
    print("Missing (404):", missing)
    print("Forbidden (403/billing):", forbidden)