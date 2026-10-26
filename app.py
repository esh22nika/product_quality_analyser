import os
import sys
from pathlib import Path

# Automatically ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from visionguard.ui.layout import create_app
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.app")

def main():
    import torch
    torch.set_num_threads(int(os.environ.get("VISIONGUARD_CPU_THREADS", "4")))
    logger.info("Initializing VisionGuard Gradio Application...")
    demo = create_app()
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Launching server on port {port}...")
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=port, share=False, max_file_size="2gb")

if __name__ == "__main__":
    main()
