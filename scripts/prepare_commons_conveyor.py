"""Download and prepare the licensed real conveyor clip; no recording required."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from visionguard.data.public_videos import prepare_conveyor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/public_videos')
    args = parser.parse_args()
    print(prepare_conveyor(args.output))
