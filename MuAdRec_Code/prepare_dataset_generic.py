import argparse
import ast
import gzip
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_line(line: str) -> dict:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def read_gz_records(path: str) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = parse_line(line)
            if rec:
                yield rec


def left_pad_sequence(seq: List[int], max_len: int, pad_id: int) -> Tuple[List[int], int]:
    if len(seq) >= max_len:
        clipped = seq[-max_len:]
        return clipped, max_len
    pad = [pad_id] * (max_len - len(seq))
    return pad + seq, len(seq)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, list):
        return " ".join([safe_text(v) for v in value if safe_text(v)])
    if isinstance(value, dict):
        return " ".join([f"{k}:{safe_text(v)}" for k, v in value.items() if safe_text(v)])
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def get_image_urls(rec: dict, image_keys: List[str]):
    for key in image_keys:
        val = rec.get(key, None)
        if not val:
            continue
        if isinstance(val, str):
            v = val.strip()
            return [v] if v else []
        if isinstance(val, list):
            out = [str(x).strip() for x in val if str(x).strip()]
            if out:
                return out
    return []


def collect_user_interactions(
    review_path: str,
    user_id_field: str,
    item_id_field: str,
    timestamp_field: str,
) -> Dict[str, List[Tuple[int, str]]]:
    user_hist = defaultdict(list)
    for rec in read_gz_records(review_path):
        uid = rec.get(user_id_field)
        asin = rec.get(item_id_field)
        ts = rec.get(timestamp_field)
        if uid is None or asin is None or ts is None:
            continue
        user_hist[str(uid)].append((int(ts), str(asin)))
    return user_hist


def filter_users_by_interactions(user_hist: Dict[str, List[Tuple[int, str]]], min_interactions: int) -> Dict[str, List[Tuple[int, str]]]:
    filtered = {}
    for uid, hist in user_hist.items():
        if len(hist) >= min_interactions:
            filtered[uid] = sorted(hist, key=lambda x: x[0])
    return filtered


def split_users_by_timestamp(user_hist: Dict[str, List[Tuple[int, str]]], train_ratio: float = 0.8, val_ratio: float = 0.1):
    user_last_ts = []
    for uid, hist in user_hist.items():
        last_ts = hist[-1][0]
        user_last_ts.append((last_ts, uid))
    
    user_last_ts.sort()
    
    total = len(user_last_ts)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_users = [uid for _, uid in user_last_ts[:train_end]]
    val_users = [uid for _, uid in user_last_ts[train_end:val_end]]
    test_users = [uid for _, uid in user_last_ts[val_end:]]
    
    return train_users, val_users, test_users


def build_item_mapping(user_hist: Dict[str, List[Tuple[int, str]]]) -> Dict[str, int]:
    item_set = set()
    for seq in user_hist.values():
        for _, asin in seq:
            item_set.add(asin)
    item_list = sorted(item_set)
    return {asin: i for i, asin in enumerate(item_list)}


def split_to_samples(
    user_hist: Dict[str, List[Tuple[int, str]]],
    user_list: List[str],
    item2id: Dict[str, int],
    seq_size: int,
) -> pd.DataFrame:
    item_num = len(item2id)
    pad_id = item_num
    rows = []
    
    for uid in user_list:
        hist = user_hist[uid]
        items = [item2id[asin] for _, asin in hist if asin in item2id]
        if len(items) < 3:
            continue
        
        for i in range(1, len(items) - 2):
            seq, l = left_pad_sequence(items[:i], seq_size, pad_id)
            rows.append({"seq": seq, "len_seq": l, "next": items[i]})
        
        v_seq, v_len = left_pad_sequence(items[:-2], seq_size, pad_id)
        rows.append({"seq": v_seq, "len_seq": v_len, "next": items[-2]})
        
        t_seq, t_len = left_pad_sequence(items[:-1], seq_size, pad_id)
        rows.append({"seq": t_seq, "len_seq": t_len, "next": items[-1]})
    
    return pd.DataFrame(rows)


def build_meta_csv_and_image_manifest(
    meta_path: str,
    item2id: Dict[str, int],
    item_id_field: str,
    text_fields: List[str],
    image_fields: List[str],
    out_meta_csv: str,
    out_image_manifest_csv: str,
    image_root: str,
):
    rows = []
    img_rows = []
    found = set()

    for rec in read_gz_records(meta_path):
        asin = rec.get(item_id_field)
        if asin is None:
            continue
        asin = str(asin)
        if asin not in item2id:
            continue
        found.add(asin)
        
        text_parts = []
        for field in text_fields:
            val = safe_text(rec.get(field, ""))
            if val:
                text_parts.append(val)
        
        image_urls = get_image_urls(rec, image_fields)

        rows.append(
            {
                "item_token": asin,
                "item_id": item2id[asin],
                "text": " ".join(text_parts).strip(),
            }
        )
        local_dir = os.path.join(image_root, asin)
        local_file = os.path.join(local_dir, "image_0.jpg")
        first_url = image_urls[0] if len(image_urls) > 0 else ""
        img_rows.append(
            {
                "item_token": asin,
                "item_id": item2id[asin],
                "image_url": first_url,
                "local_dir": local_dir,
                "local_file": local_file,
            }
        )
        ensure_dir(local_dir)

    id2token = [None] * len(item2id)
    for t, i in item2id.items():
        id2token[i] = t
    row_map = {r["item_token"]: r for r in rows}
    img_map = {r["item_token"]: r for r in img_rows}
    full_rows = []
    full_imgs = []
    for i, asin in enumerate(id2token):
        if asin in row_map:
            full_rows.append(row_map[asin])
            full_imgs.append(img_map[asin])
        else:
            local_dir = os.path.join(image_root, asin)
            local_file = os.path.join(local_dir, "image_0.jpg")
            ensure_dir(local_dir)
            full_rows.append(
                {
                    "item_token": asin,
                    "item_id": i,
                    "text": "",
                }
            )
            full_imgs.append(
                {
                    "item_token": asin,
                    "item_id": i,
                    "image_url": "",
                    "local_dir": local_dir,
                    "local_file": local_file,
                }
            )

    meta_df = pd.DataFrame(full_rows).sort_values("item_id").reset_index(drop=True)
    img_df = pd.DataFrame(full_imgs).sort_values("item_id").reset_index(drop=True)
    meta_df.to_csv(out_meta_csv, index=False, encoding="utf-8")
    img_df.to_csv(out_image_manifest_csv, index=False, encoding="utf-8")
    print(f"Meta matched in source: {len(found)}/{len(item2id)}")


def calculate_popularity(
    user_hist: Dict[str, List[Tuple[int, str]]],
    user_list: List[str],
    item2id: Dict[str, int],
) -> np.ndarray:
    pop = np.zeros((len(item2id),), dtype=np.int64)
    for uid in user_list:
        hist = user_hist[uid]
        for _, asin in hist:
            if asin in item2id:
                pop[item2id[asin]] += 1
    return pop


def main():
    parser = argparse.ArgumentParser(description="Prepare generic dataset with custom filtering and splitting.")
    
    # Dataset paths
    parser.add_argument("--review_gz", type=str, required=True, help="Path to review json.gz file")
    parser.add_argument("--meta_gz", type=str, required=True, help="Path to meta json.gz file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    
    # Dataset field mappings
    parser.add_argument("--user_id_field", type=str, default="reviewerID", help="Field name for user ID in review data")
    parser.add_argument("--item_id_field", type=str, default="asin", help="Field name for item ID in review and meta data")
    parser.add_argument("--timestamp_field", type=str, default="unixReviewTime", help="Field name for timestamp in review data")
    
    # Meta data fields
    parser.add_argument("--text_fields", type=str, default="title,categories,brand,description", 
                       help="Comma-separated list of text fields from meta data")
    parser.add_argument("--image_fields", type=str, default="imageURLHighRes,imageURL,imUrl", 
                       help="Comma-separated list of image URL fields from meta data")
    
    # Preprocessing parameters
    parser.add_argument("--seq_size", type=int, default=10, help="Maximum sequence length for padding")
    parser.add_argument("--min_interactions", type=int, default=5, help="Minimum interactions per user")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of users for training set")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of users for validation set")
    
    # Output paths
    parser.add_argument("--out_data_root", type=str, default="../ours_DiT/data", help="Output directory for data files")
    parser.add_argument("--out_aux_root", type=str, default="multimodal_preprocess/datasets", help="Output directory for auxiliary files")
    
    args = parser.parse_args()
    
    # Parse list fields
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]
    image_fields = [x.strip() for x in args.image_fields.split(",") if x.strip()]
    
    data_dir = os.path.join(args.out_data_root, args.dataset_name)
    aux_dir = os.path.join(args.out_aux_root, args.dataset_name)
    image_root = os.path.join(aux_dir, "image")
    ensure_dir(data_dir)
    ensure_dir(aux_dir)
    ensure_dir(image_root)

    print("=" * 60)
    print("Dataset Configuration:")
    print(f"  Dataset name: {args.dataset_name}")
    print(f"  User ID field: {args.user_id_field}")
    print(f"  Item ID field: {args.item_id_field}")
    print(f"  Timestamp field: {args.timestamp_field}")
    print(f"  Text fields: {text_fields}")
    print(f"  Image fields: {image_fields}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Step 1: Reading review interactions...")
    print("=" * 60)
    user_hist = collect_user_interactions(
        args.review_gz,
        args.user_id_field,
        args.item_id_field,
        args.timestamp_field,
    )
    print(f"Original users: {len(user_hist)}")
    
    print("\n" + "=" * 60)
    print(f"Step 2: Filtering users with >= {args.min_interactions} interactions...")
    print("=" * 60)
    filtered_user_hist = filter_users_by_interactions(user_hist, args.min_interactions)
    print(f"Users after filtering: {len(filtered_user_hist)}")
    
    print("\n" + "=" * 60)
    print("Step 3: Building item mapping...")
    print("=" * 60)
    item2id = build_item_mapping(filtered_user_hist)
    print(f"Total items: {len(item2id)}")
    
    print("\n" + "=" * 60)
    print(f"Step 4: Splitting users by last interaction time ({args.train_ratio:.0%}:{args.val_ratio:.0%}:{1-args.train_ratio-args.val_ratio:.0%})...")
    print("=" * 60)
    train_users, val_users, test_users = split_users_by_timestamp(
        filtered_user_hist,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Train users: {len(train_users)}, Val users: {len(val_users)}, Test users: {len(test_users)}")
    
    print("\n" + "=" * 60)
    print("Step 5: Building train/val/test samples...")
    print("=" * 60)
    train_df = split_to_samples(filtered_user_hist, train_users, item2id, args.seq_size)
    val_df = split_to_samples(filtered_user_hist, val_users, item2id, args.seq_size)
    test_df = split_to_samples(filtered_user_hist, test_users, item2id, args.seq_size)
    
    train_df.to_pickle(os.path.join(data_dir, "train_data.df"))
    val_df.to_pickle(os.path.join(data_dir, "val_data.df"))
    test_df.to_pickle(os.path.join(data_dir, "test_data.df"))
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    data_statis = pd.DataFrame([{"seq_size": args.seq_size, "item_num": len(item2id)}])
    data_statis.to_pickle(os.path.join(data_dir, "data_statis.df"))

    print("\n" + "=" * 60)
    print("Step 6: Calculating item popularity (TRAINING SET ONLY)...")
    print("=" * 60)
    pop = calculate_popularity(filtered_user_hist, train_users, item2id)
    np.save(os.path.join(data_dir, "items_pop.npy"), pop)
    print(f"Popularity saved (train set only): min={pop.min()}, max={pop.max()}, mean={pop.mean():.2f}")

    item_dict_path = os.path.join(aux_dir, "item_dict.pickle")
    with open(item_dict_path, "wb") as f:
        pickle.dump(item2id, f)

    print("\n" + "=" * 60)
    print("Step 7: Building meta csv and image manifest...")
    print("=" * 60)
    build_meta_csv_and_image_manifest(
        meta_path=args.meta_gz,
        item2id=item2id,
        item_id_field=args.item_id_field,
        text_fields=text_fields,
        image_fields=image_fields,
        out_meta_csv=os.path.join(aux_dir, "meta_for_text.csv"),
        out_image_manifest_csv=os.path.join(aux_dir, "image_manifest.csv"),
        image_root=image_root,
    )

    print("\n" + "=" * 60)
    print("Done! All files generated successfully.")
    print("=" * 60)
    print(f"\nAlphaFuse data files in: {data_dir}")
    print("  - train_data.df")
    print("  - val_data.df")
    print("  - test_data.df")
    print("  - data_statis.df")
    print("  - items_pop.npy (TRAINING SET ONLY)")
    print(f"\nMultimodal auxiliary files in: {aux_dir}")
    print("  - item_dict.pickle")
    print("  - meta_for_text.csv")
    print("  - image_manifest.csv")
    print("  - image/<item_token>/ (folders created for images)")


if __name__ == "__main__":
    main()
