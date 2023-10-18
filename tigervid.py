####################################################################
#
# tigervid.py
#
# Luke Sheneman
# sheneman@uidaho.edu
# October 2023
#
# Given a directory of videos, process each video to look for animals
# Extracts video clips which include animals into destination directory
# Writes summary log
#
# Usage:  python tigervid.py <INPUT_DIR> <OUTPUT_DIR> <MODEL_FILE> <NUM_FRAMES_BETWEEN_SAMPLES>
#
####################################################################


import os, sys, time
import argparse
import cv2
from PIL import Image
import torch
import glob
import numpy as np
import imageio
from tqdm import tqdm	
from general import non_max_suppression

DEFAULT_MODEL           = 'md_v5a.0.0.pt'
DEFAULT_INTERVAL        = 30  # number of frames between samples
DEFAULT_BUFFER_TIME     = 5   # number of seconds of video to include before first detection and after last detection
DEFAULT_REPORT_FILENAME = "report.csv"
DEFAULT_NPROCS          = 4
DEFAULT_BATCH_SIZE      = 8


parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default="input",  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default="output", help='Path to output directory for clips and metadatas')

parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Path to the PyTorch model weights file (DEFAULT: '+DEFAULT_MODEL+')')
parser.add_argument('-i', '--interval', type=int, default=DEFAULT_INTERVAL, help='Number of frames to read between sampling with AI (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-b', '--buffer', type=int, default=DEFAULT_BUFFER_TIME, help='Number of seconds to prepend and append to clip (DEFAULT: '+str(DEFAULT_BUFFER_TIME)+')')
parser.add_argument('-r', '--report', default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-p', '--processes', type=int, default=DEFAULT_NPROCS, help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-s', '--batchsize', type=int, default=DEFAULT_BATCH_SIZE, help='The batch size for inference (DEFAULT: '+str(DEFAULT_BATCH_SIZE)+')')

group = parser.add_mutually_exclusive_group()
group.add_argument('-g', '--gpu', action='store_true',  default=True, help='Use GPU if available (DEFAULT)')
group.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU only')

args = parser.parse_args()
#for k, v in args.__dict__.items():print(f"{k}: {v}")

if(not os.path.isfile(args.model)):
	print("Error:  Could not find model weights '%s'" %args.model)
	parser.print_usage()
	exit(-1)

if(not os.path.exists(args.input)):
	print("Error:  Could not find input directory path '%s'" %args.input)
	parser.print_usage()
	exit(-1)

if(not os.path.exists(args.output)):
	print("Error:  Could not find output directory path '%s'" %args.output)
	parser.print_usage()
	exit(-1)

if(args.cpu==True):
	device = "cpu"
	torch.device(device)
	print("Using CPU")
	usegpu = False
else:
	if(torch.cuda.is_available()):
	    device = "cuda"
	    usegpu = True
	    print("Using GPU")
	else:
	    device = "cpu"
	    usegpu = False

	torch.device(device)
    


print('''
****************************************************
*                       __,,,,_                    *
*        _ __..-;''`--/'/ /.',-`-.                 *
*    (`/' ` |  \ \ \\ / / / / .-'/`,_               *
*   /'`\ \   |  \ | \| // // / -.,/_,'-,           *
*  /<7' ;  \ \  | ; ||/ /| | \/    |`-/,/-.,_,/')  *
* /  _.-, `,-\,__|  _-| / \ \/|_/  |    '-/.;.\'    * 
* `-`  f/ ;      / __/ \__ `/ |__/ |               *
*      `-'      |  -| =|\_  \  |-' |               *
*            __/   /_..-' `  ),'  //               *
*           ((__.-'((___..-'' \__.'                *
*                                                  *
****************************************************
''')

print("           BEGINNING PROCESSING          ")
print("*********************************************")
print("        INPUT_DIR: ", args.input)
print("       OUTPUT_DIR: ", args.output)
print("    MODEL WEIGHTS: ", args.model)
print("SAMPLING INTERVAL: ", args.interval, "frames")
print("  BUFFER DURATION: ", args.buffer, "seconds")
print(" CONCURRENT PROCS: ", args.processes)
print("       BATCH SIZE: ", args.batchsize)
print("          USE GPU: ", usegpu)
print("*********************************************\n\n")


model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.to(device)


def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def grouper(iterable):
	prev = None
	group = []
	for item in iterable:
		if prev is None or item - prev <= 3*args.interval:  
			group.append(item)
		else:
			yield group
			group = [item]
		prev = item
	if group:
		yield group


def confidence(group, frames):
	min = 9999.0
	max = 0
	sum = 0
	count = 0
	for i in group:
		detections = frames[i]
		for d in detections:
			c = d[4]
			if(c < min):
			    min = c
			if(c > max):
			    max = c

			count += 1
			sum   += c
	avg = sum/float(count)
    
	return(min, max, avg)
			

report_file = open(args.report, "w")
report_file.write("ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, MIN_CONF, MAX_CONF, MEAN_CONF\n")

path = os.path.join(args.input, "*.mp4")

# pre-allocate our inference buffer
inference_buffer = np.empty((4096, 640, 640, 3), dtype=np.uint8)

for filename in glob.glob(path):

	start_time = time.time()

	# Use imageio[ffmpeg] to determine the number of frames	
	v=imageio.get_reader(filename,  'ffmpeg')
	nframes  = v.count_frames()
	metadata = v.get_meta_data()
	fps = metadata['fps']
	duration = metadata['duration']
	size = metadata['size']
	(width,height) = size
	buffer_frames = int(fps*args.buffer)

	print("\n")
	print(" PROCESSING VIDEO:", filename)
	print("     TOTAL FRAMES:", nframes)
	print("              FPS:", fps)
	print("         DURATION:", duration, "seconds")
	print("       FRAME SIZE:", size)
	
	print("\n")

	invid = cv2.VideoCapture(filename)

	count        = 0
	tiger_frames = 0
	detections   = []

	pbar = tqdm(range(nframes),ncols=100,unit=" frames")
	pbar.set_postfix({'tigers detected': 0})

	count = 0	
	goo = []
	for i in pbar:
		success, image = invid.read()
		if success:
			if((i % args.interval)==0):
				inference_buffer[count] = cv2.resize(image, (640,640))
				count += 1
		else:
			break

	print("%d images in inference buffer.  Now performing inference" %count)

	nbatches = (count + args.batchsize - 1) // args.batchsize
	print("NBATCHES: ", nbatches)

	# Iterate over the array to copy batches
	tiger_frames = {}	
	for b in range(nbatches):
		start_idx = b * args.batchsize
		end_idx = start_idx + args.batchsize
		if(end_idx > count):
			end_idx = count
		batch_images = inference_buffer[start_idx:end_idx] 
		#print(start_idx, end_idx, batch_images.shape)
	
		image_tensors = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float() / 255.0  
		detections_tensor = model(image_tensors)
	
		detections = non_max_suppression(detections_tensor)

		for i, d in enumerate(detections):
			frame_idx = (b*args.batchsize+i)*args.interval
			dn = d.cpu().detach().numpy()
			if len(dn):
				tiger_frames[frame_idx] = dn
		
	groups = dict(enumerate(grouper(tiger_frames.keys()), 0))
	print("IDENTIFIED %d CLIPS THAT INCLUDE TIGERS" %(len(groups)))

	for g in groups:

		min_conf, max_conf, mean_conf = confidence(groups[g], tiger_frames) 

		fn = os.path.basename(filename)
		clip_name = os.path.splitext(fn)[0] + "_{:03d}".format(g) + ".mp4"
		clip_path = os.path.join(args.output, clip_name)

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
		outvid = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))

		start_frame = groups[g][0]-buffer_frames
		if(start_frame < 0):
			start_frame = 0
	
		end_frame = groups[g][len(groups[g])-1]+buffer_frames
		if(end_frame >= nframes):
			end_frame = nframes-1;

		invid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		s = "\"%s\", \"%s\", %d, %f, %d, %f, %d, %f, %.02f, %.02f, %.02f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps, min_conf, max_conf, mean_conf)
		report_file.write(s)
		print("CLIP %d: start=%d, end=%d" %(g,start_frame,end_frame))
		print("SAVING CLIP: ", clip_path, "...")
		for f in range(start_frame, end_frame):
			success, image = invid.read()
			if(success):
				outvid.write(label(image,f,fps))
			else:
				break
		outvid.release()
		print("CREATED CLIP: ", clip_path)

	end_time = time.time()
	print("TIME TAKEN: %.03f seconds" %(end_time - start_time))
	
	invid.release()
	

report_file.close()

print("DONE\n\n")
