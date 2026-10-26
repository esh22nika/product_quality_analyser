import glob
import json
import os
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    reports = sorted(glob.glob(str(root / "reports" / "*.json")))
    print(f"Total reports found: {len(reports)}")
    
    mvtec_cats = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    visa_cats = [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]

    mvtec_data = []
    visa_data = []

    for r in reports:
        try:
            with open(r, "r") as f:
                d = json.load(f)
            cat = d["dataset_info"]["category"]
            ds = d["dataset_info"]["name"]
            m = d["metrics"]
            item = {
                "category": cat,
                "dataset": ds,
                "auroc": float(m.get("auroc", 0.0)),
                "average_precision": float(m.get("average_precision", 0.0)),
                "pixel_auroc": float(m.get("pixel_auroc", 0.0)),
                "pixel_ap": float(m.get("pixel_average_precision", 0.0)),
                "samples": int(m.get("sample_count", 0)),
                "recall": float(m.get("recall", 1.0)),
                "f1": float(m.get("f1_score", 0.0)),
            }
            if cat in mvtec_cats or ds == "mvtec_ad":
                mvtec_data.append(item)
            else:
                visa_data.append(item)
        except Exception as e:
            print(f"Error parsing {r}: {e}")

    # Deduplicate by category if any duplicate runs exist (keep newest / highest samples)
    def dedup(lst):
        res = {}
        for x in lst:
            res[x["category"]] = x
        return list(res.values())

    mvtec_data = dedup(mvtec_data)
    visa_data = dedup(visa_data)

    print("\n" + "="*80)
    print("MVTEC AD BENCHMARK RESULTS (15 CATEGORIES)")
    print("="*80)
    print(f"{'Category':<15} | {'Image AUROC':<12} | {'AP':<10} | {'Pixel AUROC':<12} | {'Pixel AP':<10} | {'Test N':<8}")
    print("-" * 80)
    for row in sorted(mvtec_data, key=lambda x: x["category"]):
        print(f"{row['category']:<15} | {row['auroc']:<12.4f} | {row['average_precision']:<10.4f} | {row['pixel_auroc']:<12.4f} | {row['pixel_ap']:<10.4f} | {row['samples']:<8}")
    
    m_auroc = sum(x["auroc"] for x in mvtec_data) / len(mvtec_data) if mvtec_data else 0
    m_ap = sum(x["average_precision"] for x in mvtec_data) / len(mvtec_data) if mvtec_data else 0
    m_pix = sum(x["pixel_auroc"] for x in mvtec_data) / len(mvtec_data) if mvtec_data else 0
    m_pix_ap = sum(x["pixel_ap"] for x in mvtec_data) / len(mvtec_data) if mvtec_data else 0
    print("-" * 80)
    print(f"{'MVTEC MEAN':<15} | {m_auroc:<12.4f} | {m_ap:<10.4f} | {m_pix:<12.4f} | {m_pix_ap:<10.4f} | {len(mvtec_data)} cats")

    print("\n" + "="*80)
    print("AMAZON VISA BENCHMARK RESULTS (12 CATEGORIES)")
    print("="*80)
    print(f"{'Category':<15} | {'Image AUROC':<12} | {'AP':<10} | {'Pixel AUROC':<12} | {'Pixel AP':<10} | {'Test N':<8}")
    print("-" * 80)
    for row in sorted(visa_data, key=lambda x: x["category"]):
        print(f"{row['category']:<15} | {row['auroc']:<12.4f} | {row['average_precision']:<10.4f} | {row['pixel_auroc']:<12.4f} | {row['pixel_ap']:<10.4f} | {row['samples']:<8}")

    v_auroc = sum(x["auroc"] for x in visa_data) / len(visa_data) if visa_data else 0
    v_ap = sum(x["average_precision"] for x in visa_data) / len(visa_data) if visa_data else 0
    v_pix = sum(x["pixel_auroc"] for x in visa_data) / len(visa_data) if visa_data else 0
    v_pix_ap = sum(x["pixel_ap"] for x in visa_data) / len(visa_data) if visa_data else 0
    print("-" * 80)
    print(f"{'VISA MEAN':<15} | {v_auroc:<12.4f} | {v_ap:<10.4f} | {v_pix:<12.4f} | {v_pix_ap:<10.4f} | {len(visa_data)} cats")

    all_data = mvtec_data + visa_data
    o_auroc = sum(x["auroc"] for x in all_data) / len(all_data) if all_data else 0
    o_ap = sum(x["average_precision"] for x in all_data) / len(all_data) if all_data else 0
    o_pix = sum(x["pixel_auroc"] for x in all_data) / len(all_data) if all_data else 0
    o_pix_ap = sum(x["pixel_ap"] for x in all_data) / len(all_data) if all_data else 0
    print("\n" + "="*80)
    print(f"{'OVERALL (27 CATS)':<15} | {o_auroc:<12.4f} | {o_ap:<10.4f} | {o_pix:<12.4f} | {o_pix_ap:<10.4f} | {len(all_data)} cats")
    print("="*80)

if __name__ == "__main__":
    main()
