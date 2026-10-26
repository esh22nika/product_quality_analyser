import base64
import io
import os
import sys
import uuid
from pathlib import Path

# Automatically ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import torch

from visionguard.inference.image_inference import ImageInspectionPipeline
from visionguard.data.validation import DatasetValidator
from visionguard.storage.repository import InspectionRepository
from visionguard.storage.artifacts import ArtifactManager

def create_flask_app():
    template_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    
    repo = InspectionRepository()
    artifact_mgr = ArtifactManager()

    def np_to_base64(img_rgb: np.ndarray) -> str:
        pil_img = Image.fromarray(img_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @app.route("/")
    def index():
        device_str = "CUDA (GPU Acceleration)" if torch.cuda.is_available() else "CPU (Standard)"
        return render_template("index.html", device=device_str)

    @app.route("/api/inspect", methods=["POST"])
    def inspect():
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        category = request.form.get("category", "bottle")
        model_name = request.form.get("model_name", "patchcore")
        threshold = float(request.form.get("threshold", 0.5))

        try:
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_rgb = np.array(pil_img)

            pipeline = ImageInspectionPipeline(model_name=model_name, category=category)
            res = pipeline.inspect(img_rgb, decision_threshold=threshold)

            if not res["success"]:
                return jsonify({"success": False, "error": res.get("error")}), 500

            insp_id = f"INSP_{uuid.uuid4().hex[:8]}"
            artifact_mgr.save_image_artifact(insp_id, "overlay.png", res["annotated_overlay"])
            repo.insert_inspection(
                inspection_id=insp_id,
                input_filename=file.filename or "uploaded.png",
                input_type="image",
                category=category,
                model_name=model_name,
                image_score=res["image_score"],
                decision_status=res["predicted_label"]
            )

            # Package base64 images
            images = {
                "original": np_to_base64(res["original_image"]),
                "heatmap": np_to_base64(res["heatmap"]),
                "overlay": np_to_base64(res["annotated_overlay"]),
                "mask": np_to_base64(res["thresholded_mask"])
            }

            return jsonify({
                "success": True,
                "inspection_id": insp_id,
                "predicted_label": res["predicted_label"],
                "image_score": res["image_score"],
                "inference_ms": res["inference_ms"],
                "region_analysis": res["region_analysis"],
                "quality": res["quality"],
                "images": images
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/datasets")
    def datasets():
        mvtec_root = "./datasets/mvtec_ad" if Path("./datasets/mvtec_ad").exists() else "./mvtec_anomaly_detection"
        res = DatasetValidator.validate_mvtec_ad(mvtec_root)
        visa_p = Path("./datasets/visa")
        if visa_p.exists():
            visa_cats = [d.name for d in visa_p.iterdir() if d.is_dir() and d.name != "split_csv"]
            res["visa_categories"] = visa_cats
            res["total_categories"] = res.get("total_categories", 0) + len(visa_cats)
            res["valid"] = True
        return jsonify(res)

    @app.route("/api/history")
    def history():
        df = repo.list_inspections(limit=50)
        return jsonify(df.to_dict(orient="records"))

    return app

if __name__ == "__main__":
    app = create_flask_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
