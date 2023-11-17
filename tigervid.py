####################################################################
#
# tigervid.py
#
# Luke Sheneman
# sheneman@uidaho.edu
# November 2023
#
# Given a directory of videos, process each video to look for animals
# Extracts video clips which include animals into destination directory
# Writes summary log
#
####################################################################


import os, sys, time, pathlib
import argparse
from multiprocessing import Process, current_process, freeze_support, Lock, RLock, Manager
import bisect
import cv2
import uuid
import math
import logging
import random
from PIL import Image
import torch
from torch.cuda.amp import autocast
import glob
import numpy as np
import imageio
from tqdm import tqdm	
from functools import reduce


DEFAULT_INPUT_DIR	= "inputs"
DEFAULT_OUTPUT_DIR	= "outputs"
DEFAULT_LOGGING_DIR 	= "logs"

DEFAULT_MODEL            = 'md_v5a.0.0.pt'
DEFAULT_INTERVAL         = 30  # number of frames between samples
DEFAULT_BUFFER_INTERVALS = 5   # number of intervals of video to include before first detection and after last detection
DEFAULT_REPORT_FILENAME  = "report.csv"
DEFAULT_NPROCS           = 1
DEFAULT_PROGRESSBAR	 = 'TQDM'

YOLODIR = "yolov5"


parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default=DEFAULT_INPUT_DIR,  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR, help='Path to output directory for clips and metadatas')

parser.add_argument('-m', '--model', 	   type=str, default=DEFAULT_MODEL, help='Path to the PyTorch model weights file (DEFAULT: '+DEFAULT_MODEL+')')
parser.add_argument('-i', '--interval',    type=int, default=DEFAULT_INTERVAL, help='Number of frames to read between sampling with AI (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-b', '--buffer', 	   type=int, default=DEFAULT_BUFFER_INTERVALS, help='Number of frames to prepend and append to clip (DEFAULT: '+str(DEFAULT_BUFFER_INTERVALS)+')')
parser.add_argument('-r', '--report', 	   type=str, default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-j', '--jobs', 	   type=int, default=DEFAULT_NPROCS, help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-l', '--logging', 	   type=str, default=DEFAULT_LOGGING_DIR, help='The directory for log files (DEFAULT: '+str(DEFAULT_LOGGING_DIR)+')')

parser.add_argument('-p', '--progressbar', type=str, default=DEFAULT_PROGRESSBAR, help='The mode of the progress bar.  Either \'TQDM\' or \'none\' (DEFAULT: '+str(DEFAULT_PROGRESSBAR)+')')


group = parser.add_mutually_exclusive_group()
group.add_argument('-g', '--gpu', action='store_true',  default=True, help='Use GPU if available (DEFAULT)')
group.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU only')

args = parser.parse_args()


if(not os.path.isfile(args.model)):
	print("Error:  Could not find model weights '%s'" %args.model, flush=True)
	parser.print_usage()
	exit(-1)

if(not os.path.exists(args.input)):
	print("Error:  Could not find input directory path '%s'" %args.input, flush=True)
	parser.print_usage()
	exit(-1)

if(not os.path.exists(args.output)):
	print("Could not find output directory path '%s'...Creating Directory!" %args.output, flush=True)
	os.makedirs(args.outputs)

if(not os.path.exists(args.logging)):
	print("Could not find logging directory path '%s'...Creating Directory!" %args.logging, flush=True)
	os.makedirs(args.logging)

if(args.cpu==True):
	device = "cpu"
	torch.device(device)
	if __name__ == '__main__':
	    print("Using CPU", flush=True)
	usegpu = False
else:
	if(torch.cuda.is_available()):
		device = "cuda"
		usegpu = True
		if __name__ == '__main__':
		    print("Using GPU", flush=True)
	else:
		device = "cpu"
		usegpu = False

torch.device(device)

if __name__ != '__main__':
	logging.getLogger('torch.hub').setLevel(logging.ERROR)
	if(os.path.exists(YOLODIR)):
		model = torch.hub.load(YOLODIR, 'custom', path=args.model, _verbose=False, verbose=False, source='local')
	else:
		model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model, _verbose=False, verbose=False)
	model.to(device)



def report(pid, report_list):

	filename,clip_path,fps,start_frame,end_frame,min_conf,max_conf,mean_conf = report_list
	s = "\"%s\", \"%s\", %d, %f, %d, %f, %d, %f, %.02f, %.02f, %.02f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps, min_conf, max_conf, mean_conf)

	try:
		report_file = open(args.report, "a")
		report_file.write(s)	
		report_file.flush()
		report_file.close()
	except:
		print("Warning:  Could not open report file %s for writing in report()" %(args.report), flush=True)




def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)


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


def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def reset_screen():
	if(os.name != 'nt'):
		os.system('reset')



def chunks(filenames, n):
	if n <= 0:
		return []

	chunk_size = len(filenames) // n
    
	remainder = len(filenames) % n
    
	chunks = []
	start_index = 0

	for i in range(n):
		end_index = start_index + chunk_size + (1 if i < remainder else 0)
		if(end_index > start_index):
			chunks.append(filenames[start_index:end_index])
		start_index = end_index

	return chunks




	

#
# retrieves chunk of video frames of size interval_sz
# returns: 
#     interval frame buffer and chunk id
#     detection (True/False)
#     success (True/False) 
#
def get_video_chunk(invid, model, interval_sz, pu_lock):

	global chunk_idx

	print("Getting chunk: %d" %chunk_idx)
	#print("Interval Size: %d" %interval_sz)

	buf = []
	for i in range(interval_sz):
		success, image = invid.read()
		if(success):
			#print("success")
			buf.append(image)
		else:
			#print("failure")
			res = (chunk_idx, buf)
			chunk_idx += 1
			return(res, False, False)

	inference_frame = cv2.resize(image, (640,640))
	#print("About to acquire pu_lock", flush=True)
	with pu_lock:
		try:
			results = model(inference_frame).pandas().xyxy[0]
		except:
			print("Error: Could not run model inference on this frame")
			res = (chunk_idx, buf)
			chunk_idx += 1
			return(res, False, False)

	#print("After lock...", flush=True)
	

	ntargets = results.shape[0]
	if(ntargets):
		detection = True
	else:
		detection = False
    
	print("----> Detection: ", detection)

	res = (chunk_idx, buf)
	chunk_idx+=1
	return(res, detection, True)


def write_clip(clip, frame_chunk):
	frame_chunk_idx, buf = frame_chunk

	print("Writing: ", frame_chunk_idx)
	
	for frame in buf:
		clip.write(frame)
		

def get_debug_buffer(buf):
	indices = [] 
	for f in buf: 
		indices.append(f[0])

	return(indices)
	




def process_chunk(pid, chunk, pu_lock, report_lock):

	global args
	global model
	global chunk_idx

	# lets pace ourselves on startup to help avoid general race conditions
	time.sleep(pid*1)

	for filename in chunk:

		while(True):
			try:
				v=imageio.get_reader(filename,  'ffmpeg')
				nframes  = v.count_frames()
				metadata = v.get_meta_data()
				v.close()

				fps = metadata['fps']
				duration = metadata['duration']
				size = metadata['size']
	
				break
			except:
				print("WARNING: imageio timeout.   Trying again.", flush=True)
				time.sleep(0.25)

		(width,height) = size

		try:
			invid = cv2.VideoCapture(filename)
		except:
			print("Could not read video file: ", filename, " skipping...", flush=True)
			continue

		DETECTION = 500
		SCANNING  = 501

		state = SCANNING

		chunk_idx = 0
		clip_number = 0
		buffer_chunks = []
		frame_chunk, this_detection, success = get_video_chunk(invid, model, args.interval, pu_lock)
		
		
		print("NUMBER OF FRAMES: ", nframes)
		print("NUMBER OF CHUNKS: ", math.ceil(nframes/args.interval))

		while(success):


			# state transition from SCANNING blanks to DETECTION
			if(state == SCANNING and this_detection == True):
				print("State transition from SCANNING blanks to DETECTION", flush=True)
				state = DETECTION

				# create a clip
				fn = os.path.basename(filename)
				clip_name = os.path.splitext(fn)[0] + "_{:03d}".format(clip_number) + ".mp4"
				clip_path = os.path.join(args.output, clip_name)
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
				clip = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))
				clip_number += 1
				
				# flush the current sliding window buffer to the new clip
				for fc in buffer_chunks:
					write_clip(clip, fc)
				buffer_chunks = []
				write_clip(clip, frame_chunk)

			# possible state transition from DETECTION back to SCANNING
			elif(state == DETECTION and this_detection == False):
    
				print("state  == DETECTION, this_detection == False", flush=True)
				print("grabbing forward_buffer...")
    
				# lets look into the future 2X to make sure we can split the clip
				forward_buf = []
				forward_buf.append(frame_chunk)
				last_forward_detection_idx = -1
				for i in range(2*args.buffer-1):
					frame_chunk, forward_detection, success = get_video_chunk(invid, model, args.interval, pu_lock)
					if(success):
						forward_buf.append(frame_chunk)
						if(forward_detection):
							if(i > last_forward_detection_idx):
								last_forward_detection_idx = i

				if(last_forward_detection_idx < 0):   # no positive detections in forward buffer
					print("   NO positive detections in the forward buffer.  last_forward_detection_idx=", last_forward_detection_idx,  flush=True)
   
					# some debugging of buffers
					debug_buffer = get_debug_buffer(buffer_chunks)
					print("   Primary buffer: ", debug_buffer)
					debug_buffer = get_debug_buffer(forward_buf)
					print("   Forward Buffer: ", debug_buffer)

					print("  Flushing primary buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					# flush the first part of the forward buffer (up to length args.buffer) to disk
					if(len(forward_buf)>args.buffer):
						extent = args.buffer  
					else:
						extent = len(forward_buf)	

					for i in range(extent):
						frame_chunk = forward_buf.pop(0)		
						write_clip(clip, frame_chunk)	

					# put whatever is left of the forward buffer onto the end of primary buffer
					buffer_chunks += forward_buf
					forward_buf = []

					clip.release()	
					print("***WROTE CLIP TO DISK***")
					# some debugging of buffers
					debug_buffer = get_debug_buffer(buffer_chunks)
					print("   Primary buffer: ", debug_buffer)
					debug_buffer = get_debug_buffer(forward_buf)
					print("   Forward Buffer: ", debug_buffer)
					print("Changing state back to SCANNING...", flush=True)
					state = SCANNING     # complete state transition back to SCANNING

				else:   # positive detection in the forward buffer
    
					print("  Positive detections in the forward buffer.  last_forward_detection_idx=", last_forward_detection_idx, flush=True)

					# some debugging of buffers
					debug_buffer = get_debug_buffer(buffer_chunks)
					print("   Primary buffer: ", debug_buffer)
					debug_buffer = get_debug_buffer(forward_buf)
					print("   Forward Buffer: ", debug_buffer)

					print("  Flushing buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					#write_clip(clip, frame_chunk)
				
					print("  Flushing all chunks in forward buffer up to the last_forward_detection_idx: ", last_forward_detection_idx)
					for i in range(last_forward_detection_idx):  # flush all chunks in forward buffer up to the last positive detection
						f = forward_buf.pop(0)		
						write_clip(clip, f)	
					
					buffer_chunks = forward_buf	


			elif(state == DETECTION and this_detection == True):

				print("state == DETECTION, and this_detection == TRUE", flush=True)
				write_clip(clip, frame_chunk)
	
			else:   # state == SCANNING, this_detection == FALSE
				print("state == SCANNING, this_detection == FALSE.  Continuing to see nothing....", flush=True) 

				# add this new chunk to the sliding window
				#print("Adding new chunk to sliding window...", flush=True)
				buffer_chunks.append(frame_chunk)
				if(len(buffer_chunks)>args.buffer):
					buffer_chunks.pop(0)
    
		
			frame_chunk, this_detection, success = get_video_chunk(invid, model, args.interval, pu_lock)

			#with report_lock:
			#	report(pid, [filename, clip_path, fps, start_frame, end_frame, min_conf, max_conf, mean_conf])

		try:
			clip.release()
		except:
			break
			 
		invid.release()

		exit(0)


########################################
#
# Main Execution Section
#
#
def main():

	all_start_time = time.time()

	freeze_support()  # For Windows support - multiprocessing with tqdm

	try:
		report_file = open(args.report, "w")
	except:
		print("Error: Could not open report file %s in main()" %(args.report), flush=True)
		exit(-1)

	report_file.write("ORIGINAL, CLIP, START_FRAME, START_TIME, END_FRAME, END_TIME, NUM FRAMES, DURATION, MIN_CONF, MAX_CONF, MEAN_CONF\n")
	report_file.flush()
	report_file.close()


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
	''', flush=True)

	print("           BEGINNING PROCESSING          ")
	print("*********************************************")
	print("        INPUT_DIR: ", args.input)
	print("       OUTPUT_DIR: ", args.output)
	print("    MODEL WEIGHTS: ", args.model)
	print("SAMPLING INTERVAL: ", args.interval, "frames")
	print("  BUFFER DURATION: ", args.buffer, "seconds")
	print(" CONCURRENT PROCS: ", args.jobs)
	print("          USE GPU: ", usegpu)
	print("*********************************************\n\n", flush=True)

	path = os.path.join(args.input, "*.mp4")
	files = glob.glob(path)
	random.shuffle(files)
	ch = chunks(files,args.jobs)

	manager = Manager()

	pu_lock     = manager.Lock()
	report_lock = manager.Lock()

	if(usegpu==True):
		torch.cuda.empty_cache()

	processes = []
	for pid,chunk in enumerate(ch):
		p = Process(target = process_chunk, args=(pid, chunk, pu_lock, report_lock))
		processes.append(p)
		p.start()

	for p in processes:
		p.join()

	print("Total time to process %d videos: %.02f seconds" %(len(files), time.time()-all_start_time))
	print("Report file saved to %s" %args.report)
	print("\nDONE\n")


if __name__ == '__main__':

	torch.multiprocessing.set_start_method('spawn')

	main()

