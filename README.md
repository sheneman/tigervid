# TigerVid
## Scans video looking for frames which include animals, extracts video clips and a final report

## Usage  

```
usage: tigervid [-h] [-m MODEL] [-i INTERVAL] [-b BUFFER] [-r REPORT] [-j JOBS] [-s BATCHSIZE] [-g | -c] INPUT_DIR OUTPUT_DIR  
```
  
### Efficiently analyze videos and extract clips and metadata which contain animals.  

```
positional arguments:
  INPUT_DIR             Path to input directory containing MP4 videos
  OUTPUT_DIR            Path to output directory for clips and metadatas

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the PyTorch model weights file (DEFAULT: md_v5a.0.0.pt)
  -i INTERVAL, --interval INTERVAL
                        Number of seconds between AI sampling/detection (DEFAULT: 1.0)
  -p PADDING, --padding PADDING
                        Number of seconds of video to pad on front and end of a clip (DEFAULT: 5.0)
  -r REPORT, --report REPORT
                        Name of report metadata (DEFAULT: report.csv)
  -j JOBS, --jobs JOBS  Number of concurrent (parallel) processes (DEFAULT: 4)
  -l LOGGING, --logging LOGGING
                        The directory for log files (DEFAULT: logs)
  -n, --nobar           Turns off the Progress Bar during processing. (DEFAULT: Use Progress Bar)
  -g, --gpu             Use GPU if available (DEFAULT)
  -c, --cpu             Use CPU only
```



## Installation

python3 -m venv venv  
**Linux:** source venv/bin/activate   
**Windows:**  venv\bin\activate  

pip install -U pip  
pip install torch  
pip install pillow 
pip install nvidia-ml-py3  
pip install opencv-python  
pip install imageio[ffmpeg]  
pip install tqdm  
pip install pandas  
pip install requests  

**or** pip install -r requirements.txt  

**A Note on GPUs**: This tool uses deep learning methods to detect animals, specifically using [PyTorch](https://pytorch.org) and [YOLOv5](https://github.com/ultralytics/yolov5). A CUDA-compatible GPU is recommended but not required. Currently TigerVid will only use one GPU, even if your system has multiple GPUs. 

## Optimizing Settings
This tool is intended to allow the user to optimize the runtime settings to optimize performance.  Optimal settings will depend on a variety of factors, including whether or not you have a GPU, how much GPU RAM you have, how many CPU cores are available, and how fast your storage is.  Play around with the settings until you acheive optimal speeds for your hardware.  

## Interpreting the results

The derived video clips will be named based on the original video filename and will include sequential numbering.

The reports file is a CSV file containing the following columns:  

**ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, CONFIDENCE_MIN, CONFIDENCE_MAX, CONFIDENCE MEAN**  

For Example, this report shows one source video file produced 2 clips and another source video produced one clip:  
```
ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, MIN_CONF, MAX_CONF, MEAN_CONF
"inputs/d.mp4", "outputs/d_000.mp4", 60, 2.000000, 540, 18.000000, 480, 16.000000, 0.67, 0.91, 0.81
"inputs/d.mp4", "outputs/d_001.mp4", 660, 22.000000, 1200, 40.000000, 540, 18.000000, 0.32, 0.90, 0.71
"inputs/y.mp4", "outputs/y_000.mp4", 6966, 241.539528, 7464, 258.807212, 498, 17.267684, 0.28, 0.92, 0.67
```

