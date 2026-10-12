import os
from visionguard.ui.layout import create_app
from visionguard.logging_config import setup_logger

logger = setup_logger("visionguard.app")

def main():
    logger.info("Initializing VisionGuard Gradio Application...")
    demo = create_app()
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Launching server on port {port}...")
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)

if __name__ == "__main__":
    main()
