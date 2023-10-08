# TigerVid
## Scans video looking for frames which include animals, produced clips and a final report

Usage: python tigervid.py <INPUT_DIR> <OUTPUT_DIR> <MODEL_PATH> <SAMPLE_INTERVAL>

Where:  
  * **<INPUT_DIR>** is a single directory containing one or more **MP4** videos to scan  
  * **<OUTPUT_DIR>** is the directory that will contain the extracted clips and final report  
  * **<MODEL_PATH>** is the path to a YOLOv5 detection model weights file (e.g. MegaDetector)  
  * **<SAMPLE_INTERVAL>** is the number of video frames.  I recommend using 30, which is often about 1 second of video 

## Installation

python3 -m venv venv  
source venv/bin/activate  
  
pip install -U pip  
pip install torch  
pip install pillow  
pip install opencv-python  
pip install imageio[ffmpeg]  
pip install tqdm  
pip install pandas  
pip install requests  

**or** pip install -r requirements.txt  



## Interpreting the results

The derived cliips will include some annotation on the frames which shows:  

<img width="474" alt="image" src="https://github.com/sheneman/tigervid/assets/3028345/3ded327d-6a0e-4b34-9b02-acccb867bf94">  


* The frame number within the **ORIGINAL** video file
* The timestamp within the **ORIGINAL** video file

The reports file is a CSV file containing the following columns:  

**ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION**  

For Example, this report shows one source video file produced 2 clips and another source video produced one clip:  

ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION  
"videos/tigertrue2.mp4", "output/tigertrue2_000.mp4", 60, 2.000000, 540, 18.000000, 480, 16.000000  
"videos/tigertrue2.mp4", "output/tigertrue2_001.mp4", 660, 22.000000, 1200, 40.000000, 540, 18.000000  
"videos/a.mp4", "output/a_000.mp4", 60, 2.000000, 540, 18.000000, 480, 16.000000  


