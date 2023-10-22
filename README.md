# TigerVid
## Scans video looking for frames which include animals, produce clips and a final report
  
usage: tigervid [-h] [-m MODEL] [-i INTERVAL] [-b BUFFER] [-r REPORT] [-j JOBS] [-s BATCHSIZE] [-g | -c] INPUT_DIR OUTPUT_DIR  
  
Analyze videos and extract clips and metadata which contain animals.  
  
positional arguments:  
  INPUT_DIR             Path to input directory containing MP4 videos  
  OUTPUT_DIR            Path to output directory for clips and metadatas  
  
optional arguments:  
  -h, --help            show this help message and exit  
  -m MODEL, --model MODEL  
                        Path to the PyTorch model weights file (DEFAULT: md_v5a.0.0.pt)  
  -i INTERVAL, --interval INTERVAL  
                        Number of frames to read between sampling with AI (DEFAULT: 30)  
  -b BUFFER, --buffer BUFFER  
                        Number of seconds to prepend and append to clip (DEFAULT: 5)  
  -r REPORT, --report REPORT  
                        Name of report metadata (DEFAULT: report.csv)  
  -j JOBS, --jobs JOBS  Number of concurrent (parallel) processes (DEFAULT: 1)  
  -s BATCHSIZE, --batchsize BATCHSIZE  
                        The batch size for inference (DEFAULT: 8)  
  -g, --gpu             Use GPU if available (DEFAULT)  
  -c, --cpu             Use CPU only  



## Installation

python3 -m venv venv  
**Linux:** source venv/bin/activate   
**Windows:**  venv\bin\activate  

pip install -U pip  
pip install torch  
pip install pillow  
pip install opencv-python  
pip install imageio[ffmpeg]  
pip install tqdm  
pip install pandas  
pip install requests  

**or** pip install -r requirements.txt  

**A Note on GPUs**: This tool uses deep learning methods to detect animals, specifically using [PyTorch](https://pytorch.org) and [YOLOv5](https://github.com/ultralytics/yolov5). A CUDA-compatible GPU is recommended but not required.  

## Interpreting the results

The derived video clips will be named based on the original video and include some annotation on the frames which shows:  

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


