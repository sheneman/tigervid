# Animal Count
## Scans video looking for frames which include animals, produced clips and a final report

Usage: python acount.py <INPUT_DIR> <OUTPUT_DIR> <MODEL_PATH> <SAMPLE_INTERVAL>

Where:
  <INPUT_DIR> is a single directory containing one or more videos to scan
  <OUTPUT_DIR> is the directory that will contain the extracted clips and final report
  <MODEL_PATH> is the path to a YOLOv5 detection model weights file (e.g. MegaDetector)
  <SAMPLE_INTERVAL> is the number of video frames 

## Installation

python3 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt


