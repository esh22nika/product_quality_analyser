"""Launch the light VisionGuard workspace."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
if __name__ == '__main__':
    import torch
    from visionguard.web.app_flask import create_flask_app
    torch.set_num_threads(int(os.environ.get('VISIONGUARD_CPU_THREADS','4')))
    create_flask_app().run(host='127.0.0.1',port=int(os.environ.get('PORT','5000')),threaded=True,debug=False)
