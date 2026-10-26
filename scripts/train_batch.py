import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# All 15 MVTec AD categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# All 12 Amazon VisA categories
VISA_CATEGORIES = [
    "pcb1", "pcb2", "pcb3", "pcb4", "candle",
    "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pipe_fryum"
]

def build_job_list():
    jobs = []
    checkpoint_dir = ROOT_DIR / "checkpoints"
    
    # 1. MVTec jobs (coreset_ratio 0.01)
    for cat in MVTEC_CATEGORIES:
        ckpt = checkpoint_dir / f"patchcore_{cat}.pt"
        if not ckpt.exists():
            jobs.append({
                "dataset": "mvtec_ad",
                "category": cat,
                "root": "./mvtec_anomaly_detection",
                "coreset_ratio": 0.01
            })
            
    # 2. VisA jobs (coreset_ratio 0.001)
    for cat in VISA_CATEGORIES:
        ckpt = checkpoint_dir / f"patchcore_{cat}.pt"
        if not ckpt.exists():
            jobs.append({
                "dataset": "visa",
                "category": cat,
                "root": "./datasets/visa",
                "coreset_ratio": 0.001
            })
            
    return jobs

def run_job(job, index, total):
    dataset = job["dataset"]
    cat = job["category"]
    root = job["root"]
    ratio = job["coreset_ratio"]

    print(f"\n=======================================================", flush=True)
    print(f"[{index}/{total}] STARTING: {dataset.upper()} - {cat} (Ratio: {ratio})", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Train
    train_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train.py"),
        "--model", "patchcore",
        "--dataset", dataset,
        "--category", cat,
        "--root", root,
        "--coreset-ratio", str(ratio)
    ]
    print(f"Running Training: {' '.join(train_cmd)}", flush=True)
    ret_train = subprocess.run(train_cmd, cwd=str(ROOT_DIR))
    if ret_train.returncode != 0:
        print(f"ERROR: Training failed for {cat} with code {ret_train.returncode}", flush=True)
        return False

    # 2. Evaluate
    eval_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "evaluate.py"),
        "--model", "patchcore",
        "--dataset", dataset,
        "--category", cat,
        "--root", root
    ]
    print(f"\nRunning Evaluation: {' '.join(eval_cmd)}", flush=True)
    ret_eval = subprocess.run(eval_cmd, cwd=str(ROOT_DIR))
    if ret_eval.returncode != 0:
        print(f"ERROR: Evaluation failed for {cat} with code {ret_eval.returncode}", flush=True)
        return False

    print(f"SUCCESS: Finished {dataset.upper()} - {cat}!", flush=True)
    return True

def main():
    jobs = build_job_list()
    total = len(jobs)
    print(f"Found {total} remaining categories to train for PatchCore across MVTec AD and VisA.", flush=True)
    if total == 0:
        print("All categories have already been trained!", flush=True)
        return

    results = {}
    for i, job in enumerate(jobs, start=1):
        success = run_job(job, i, total)
        results[job['category']] = "SUCCESS" if success else "FAILED"

    print("\n=======================================================", flush=True)
    print("ALL REMAINING CATEGORIES COMPLETED:")
    for cat, status in results.items():
        print(f" - {cat}: {status}", flush=True)
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
