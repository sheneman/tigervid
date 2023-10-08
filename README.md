# Animal Count
## Scans video looking for frames which include animals, produced clips and a final report

Usage: python acount.py <INPUT_DIR> <OUTPUT_DIR> <MODEL_PATH> <SAMPLE_INTERVAL>

Where:  
  **<INPUT_DIR>** is a single directory containing one or more **MP4** videos to scan  
  **<OUTPUT_DIR>** is the directory that will contain the extracted clips and final report  
  **<MODEL_PATH>** is the path to a YOLOv5 detection model weights file (e.g. MegaDetector)  
  **<SAMPLE_INTERVAL>** is the number of video frames   

## Installation

python3 -m venv venv  
source venv/bin/activate  

pip install -U pip  
pip install -r requirements.txt  

## Interpreting the results

The derived cliips will include some annotation on the frames which shows:  
<img width="529" alt="image" src="https://github.com/sheneman/animal_count/assets/3028345/17e3cdac-edba-4852-a821-314bec28c7ed">
. The frame number within the **ORIGINAL** video file  
. The timestamp within the **ORIGINAL** video file
