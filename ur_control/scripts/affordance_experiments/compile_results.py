#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_results.py

用途：
  掃描某個物件的一批 trial log，自動抓出每筆有 [Affordance] 記錄的有效數據，
  整理成 CSV，並把 CSV + 對應的成功 log 複製進 affordance_experiments/，
  取代原本每次實驗跑完手動整理的流程。

用法：
  python3 compile_results.py <object_name> <trials_log_dir>

  例如：
  python3 compile_results.py spatula /tmp/.../scratchpad/spatula_trials

輸出：
  affordance_experiments/<object_name>_affordance_results.csv
  affordance_experiments/logs/<object_name>_trial_N.log（每筆有效數據對應的原始 log）
"""

import csv
import os
import re
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = SCRIPT_DIR
LOGS_DEST_DIR = os.path.join(DEST_DIR, "logs")

AFFORDANCE_RE = re.compile(
    r"\[Affordance\]\s+(\S+):\s+左手接觸點主軸位置=([\d.]+)%\s+落在握柄GT=(是|否)"
    r"(?:\s+obj_pos=\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\))?"
)
INFER_RE = re.compile(r"推論時間\s*\(LLM \+ AnyGrasp\)\s*:\s*([\d.]+)\s*s")
GRASP_RE = re.compile(r"夾取耗時\s*:\s*([\d.]+)\s*s")
TOTAL_RE = re.compile(r"(?:交接總耗時 \(夾取→完成\)|總耗時 \(觸發→完成\))\s*:\s*([\d.]+)\s*s")
ROT_RE = re.compile(r"旋轉量 \(rot\)\s*:\s*([+-]?[\d.]+)")
HOE_RE = re.compile(r"(?:HOE_func|HOE_exec)\s*:\s*([+-]?[\d.]+)")


def parse_trial_log(path):
    """回傳 dict（有效數據）或 None（這筆不算數）"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = AFFORDANCE_RE.search(text)
    if not m:
        return None

    object_name, ratio, hit_zh, ox, oy, oz = m.groups()
    row = {
        "object": object_name,
        "affordance_ratio_pct": ratio,
        "affordance_result": "HIT" if hit_zh == "是" else "MISS",
        "obj_pos_z": oz if oz is not None else "",
        "note": "",
    }

    for key, pattern in [
        ("inference_time_s", INFER_RE),
        ("grasp_time_s", GRASP_RE),
        ("total_time_s", TOTAL_RE),
        ("rotation_deg", ROT_RE),
        ("hoe_func_deg", HOE_RE),
    ]:
        mm = pattern.search(text)
        row[key] = mm.group(1) if mm else ""

    return row


def main():
    if len(sys.argv) != 3:
        print(f"用法: python3 {sys.argv[0]} <object_name> <trials_log_dir>")
        sys.exit(1)

    object_name = sys.argv[1]
    trials_dir = sys.argv[2]

    trial_files = sorted(
        (f for f in os.listdir(trials_dir) if re.match(r"trial_\d+[a-z]?\.log$", f)),
        key=lambda f: [int(x) if x.isdigit() else x
                        for x in re.findall(r"\d+|[a-z]+", f)]
    )

    rows = []
    valid_logs = []
    for fname in trial_files:
        path = os.path.join(trials_dir, fname)
        row = parse_trial_log(path)
        if row is not None:
            row["trial_log"] = fname
            rows.append(row)
            valid_logs.append(fname)

    if not rows:
        print("沒有找到任何有效數據（沒有 [Affordance] 記錄的 log）")
        sys.exit(1)

    fieldnames = [
        "index", "trial_log", "object", "affordance_ratio_pct", "affordance_result",
        "rotation_deg", "hoe_func_deg", "inference_time_s", "grasp_time_s",
        "total_time_s", "obj_pos_z", "note",
    ]

    os.makedirs(LOGS_DEST_DIR, exist_ok=True)
    csv_path = os.path.join(DEST_DIR, f"{object_name}_affordance_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row["index"] = i
            writer.writerow(row)

    for fname in valid_logs:
        src = os.path.join(trials_dir, fname)
        dst = os.path.join(LOGS_DEST_DIR, f"{object_name}_{fname}")
        shutil.copy2(src, dst)

    hits = sum(1 for r in rows if r["affordance_result"] == "HIT")
    print(f"寫入 {csv_path}")
    print(f"{len(rows)} 筆有效數據，HIT {hits}/{len(rows)} = {hits/len(rows)*100:.1f}%")
    print(f"已複製 {len(valid_logs)} 個 log 到 {LOGS_DEST_DIR}/")


if __name__ == "__main__":
    main()
