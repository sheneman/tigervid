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


import os, sys
import cv2
from PIL import Image
import torch
import glob
import imageio
from tqdm import tqdm	


if(len(sys.argv)!=5):
	print("Usage:  python tigervid.py <INPUT_DIR> <OUTPUT_DIR> <MODEL_WEIGHTS> <NUM_FRAMES_BETWEEN_SAMPLES>")
	exit(-1)

INPUT_DIR  = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
MODEL      = sys.argv[3]
INTERVAL   = int(sys.argv[4])

BUFFER_TIME = 5 # seconds one either end of clip
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.csv")


print(" INPUT_DIR: ", INPUT_DIR)
print("OUTPUT_DIR: ", OUTPUT_DIR)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)
	

def grouper(iterable):
	prev = None
	group = []
	for item in iterable:
		if prev is None or item - prev <= int(3*INTERVAL):
			group.append(item)
		else:
			yield group
			group = [item]
		prev = item
	if group:
		yield group

report_file = open(REPORT_FILE, "w")
report_file.write("ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION\n")

path = os.path.join(INPUT_DIR, "*.mp4")
for filename in glob.glob(path):

	# Use imageio[ffmpeg] to determine the number of frames	
	v=imageio.get_reader(filename,  'ffmpeg')
	nframes  = v.count_frames()
	metadata = v.get_meta_data()
	fps = metadata['fps']
	duration = metadata['duration']
	size = metadata['size']
	(width,height) = size
	buffer_frames = int(fps*BUFFER_TIME)

	print("\n")
	print(" PROCESSING VIDEO:", filename)
	print("     TOTAL FRAMES:", nframes)
	print("              FPS:", fps)
	print("         DURATION:", duration, " seconds")
	print("       FRAME SIZE:", size)
	print("SAMPLING INTERVAL:", INTERVAL, " frames")
	print("\n")

	invid = cv2.VideoCapture(filename)

	count        = 0
	tiger_frames = 0
	detections   = []

	pbar = tqdm(range(nframes),ncols=100,unit=" frames")
	pbar.set_postfix({'tigers detected': 0})
	
	success, image = invid.read()
	for i in pbar:
		if success:
			if((count % INTERVAL)==0):
				results = model(image).pandas().xyxy[0]
				ntargets = results.shape[0]
				if(ntargets):
					detections.append((count,ntargets,results["confidence"].mean()))
					tiger_frames += 1
					pbar.set_postfix({'frames w/tigers': tiger_frames})
					

			success, image = invid.read()
			count += 1
		else:
			break


	frames_list = [t[0] for t in detections]
	groups = dict(enumerate(grouper(frames_list), 0))
	print("IDENTIFIED %d CLIPS THAT INCLUDE TIGERS" %(len(groups)))
	for g in groups:

		fn = os.path.basename(filename)
		clip_name = os.path.splitext(fn)[0] + "_{:03d}".format(g) + ".mp4"
		clip_path = os.path.join(OUTPUT_DIR, clip_name)

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
		outvid = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))

		start_frame = groups[g][0]-buffer_frames
		if(start_frame < 0):
			start_frame = 0
	
		end_frame = groups[g][len(groups[g])-1]+buffer_frames
		if(end_frame >= nframes):
			end_frame = nframes-1;

		invid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		s = "\"%s\", \"%s\", %d, %f, %d, %f, %d, %f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps)
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
	
	invid.release()

report_file.close()

print("DONE\n\n")
