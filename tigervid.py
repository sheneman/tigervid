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
import cv2
import math
import nvidia_smi
import logging
import random
import torch
import glob
import numpy as np
import imageio
from tqdm import tqdm	


DEFAULT_INPUT_DIR	= "inputs"
DEFAULT_OUTPUT_DIR	= "outputs"
DEFAULT_LOGGING_DIR 	= "logs"

DEFAULT_MODEL            = 'md_v5a.0.0.pt'
DEFAULT_INTERVAL         = 1.0   # number of seconds between samples
DEFAULT_PADDING		 = 5.0   # number of seconds of video to include before first detection and after last detection in a clip
DEFAULT_REPORT_FILENAME  = "report.csv"
DEFAULT_NPROCS           = 1
DEFAULT_PROGRESSBAR	 = 'TQDM'

YOLODIR = "yolov5"


parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default=DEFAULT_INPUT_DIR,  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR, help='Path to output directory for clips and metadatas')

parser.add_argument('-m', '--model',	type=str,   default=DEFAULT_MODEL, help='Path to the PyTorch model weights file (DEFAULT: '+DEFAULT_MODEL+')')
parser.add_argument('-i', '--interval', type=float, default=DEFAULT_INTERVAL, help='Number of seconds between AI sampling/detection (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-p', '--padding',  type=float, default=DEFAULT_PADDING, help='Number of seconds of video to pad on front and end of a clip (DEFAULT: '+str(DEFAULT_PADDING)+')')
parser.add_argument('-r', '--report',   type=str,   default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-j', '--jobs',	type=int,   default=DEFAULT_NPROCS, help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-l', '--logging',	type=str,   default=DEFAULT_LOGGING_DIR, help='The directory for log files (DEFAULT: '+str(DEFAULT_LOGGING_DIR)+')')

parser.add_argument('-t', '--tqdm',	type=str, default=DEFAULT_PROGRESSBAR, help='The mode of the progress bar.  Either \'TQDM\' or \'none\' (DEFAULT: '+str(DEFAULT_PROGRESSBAR)+')')


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
	    
	try:
		if(os.path.exists(YOLODIR)):
			model = torch.hub.load(YOLODIR, 'custom', path=args.model, _verbose=False, verbose=False, source='local')
		else:
			model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model, _verbose=False, verbose=False)

		model.to(device)
	except:
		print("COULD NOT DEPLOY MODEL TO DEVICE (GPU, etc.)")
		sys.exit(-1)



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

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
	return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def get_gpu_info():
	nvidia_smi.nvmlInit()

	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	gpu_info = []
	gpu_info.append(deviceCount)
	for i in range(deviceCount):
		handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
		mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

		mem_free  = mem_info.free
		mem_total = mem_info.total
		mem_used  = mem_info.used

		gpu_info.append((mem_info.total, mem_info.used, mem_info.free))

		#print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*mem_info.free/mem_info.total, human_size(mem_info.total), human_size(mem_info.free), human_size(mem_info.used)))

	nvidia_smi.nvmlShutdown()

	return(gpu_info)



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

	#print("Getting chunk: %d" %chunk_idx)

	res = {}
	res["chunk_idx"] = chunk_idx

	buf = []
	for i in range(interval_sz):
		success, image = invid.read()
		if(success):
			buf.append(image)
		else:
			#print("Error:  Could not read frame chunk: %d" %chunk_idx)
			chunk_idx += 1
			return(None, False)
			

	inference_frame = cv2.resize(image, (640,640))
	with pu_lock:
		try:
			results = model(inference_frame).pandas().xyxy[0]
		except:
			print("Error: Could not run model inference on frame from chunk index: %d" %chunk_idx)
			sys.exit(-1)
			#chunk_idx += 1
			#return(None, False)


	if(results.shape[0]):
		detection = True
	else:
		detection = False
    
	#print("----> Detection is [%s] for chunk index: %d" %(str(detection), chunk_idx))

	res["buffer"]	 = buf
	res["detection"] = detection

	chunk_idx+=1

	return(res, True)


def write_clip(clip, frame_chunk):

	global most_recent_written_chunk 


	if(frame_chunk["chunk_idx"] <= most_recent_written_chunk):
		print("***ALERT:  Trying to write the same chunk %d twice or out of order!!!  MOST RECENT CHUNK WRITTEN: %d" %(frame_chunk["chunk_idx"], most_recent_written_chunk))
		return


	#print("Writing: [%d, %s]" %(frame_chunk["chunk_idx"], str(frame_chunk["detection"])))

	most_recent_written_chunk = frame_chunk["chunk_idx"]
	
	for frame in frame_chunk["buffer"]:
		clip.write(frame)
		


def get_debug_buffer(frame_chunk):

	debug_info = ""
	for fc in frame_chunk: 
		debug_info += "[%d|%s] " %(fc["chunk_idx"],fc["detection"]) 

	return(debug_info)




def process_chunk(pid, chunk, pu_lock, report_lock):

	global args
	global model
	global chunk_idx
	global most_recent_written_chunk

	# lets pace ourselves on startup to help avoid general race conditions
	time.sleep(pid*1)

	for fcnt, filename in enumerate(chunk):


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

		interval_frames	    = int(args.interval*fps)
		padding_intervals   = math.ceil(args.padding*fps/interval_frames)
		nchunks		    = math.ceil(nframes/interval_frames)
		chunk_idx     = 0
		clip_number   = 0
		buffer_chunks = []
		forward_buf   = []

		most_recent_written_chunk = -1
	
		#print("NUMBER OF FRAMES: ", nframes)
		#print("NUMBER OF CHUNKS: ", nchunks)
		#print("FRAMES PER INTERVAL: ", interval_frames)	
		#print("PADDING INTERVALS: ", padding_intervals)

		#clear_screen()
		pbar = tqdm(total=nframes,position=pid,ncols=100,unit=" frames",leave=False,mininterval=0.5,file=sys.stdout)
		pbar.set_description("pid=%s Processing video %d/%d: %s" %(str(pid).zfill(2),fcnt+1,len(chunk),filename))

		frame_chunk, success = get_video_chunk(invid, model, interval_frames, pu_lock)
		pbar.update(interval_frames)

		while(success):
	
			#print("CHUNK_IDX: %d/%d" %(chunk_idx, nchunks))
			if(chunk_idx > nchunks):
				#print("OUT OF CHUNKS TO READ.  BAILING")
				return

			# state transition from SCANNING blanks to DETECTION
			if(state == SCANNING and frame_chunk["detection"] == True):
				#print("State transition from SCANNING blanks to DETECTION", flush=True)
				state = DETECTION

				# some debugging of buffers
				#debug_buffer = get_debug_buffer(buffer_chunks)
				#print("   Primary buffer: ", debug_buffer)
				#debug_buffer = get_debug_buffer(forward_buf)
				#print("   Forward Buffer: ", debug_buffer)

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
			elif(state == DETECTION and frame_chunk["detection"] == False):
    
				#print("state  == DETECTION, detection == False", flush=True)
				#
				# some debugging of buffers
				#debug_buffer = get_debug_buffer(buffer_chunks)
				#print("   Primary buffer: ", debug_buffer)
				#debug_buffer = get_debug_buffer(forward_buf)
				#print("   Forward Buffer: ", debug_buffer)
				#
				#print("grabbing forward_buffer...")
    
				# lets look into the future 2X to make sure we can split the clip
				forward_buf = []
				forward_buf.append(frame_chunk)
				forward_detection_flag = frame_chunk["detection"]
				for i in range(0,2*padding_intervals+1):   #SHENEMAN
					frame_chunk, success = get_video_chunk(invid, model, interval_frames, pu_lock)
					pbar.update(interval_frames)
					if(success and frame_chunk["chunk_idx"]<=nchunks):
						forward_buf.append(frame_chunk)
						if(frame_chunk["detection"]):
							forward_detection_flag = True
				#	else:
				#		print("ELSE: success and frame_chunk[0]<=nchunks")
					

				if(forward_detection_flag == False):   # no positive detections in forward buffer
					#print("   NO positive detections in the forward buffer.", flush=True)
   
					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)

					#print("  Flushing primary buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					# flush the first part of the forward buffer disk up to padding_intervals
					if(len(forward_buf)>padding_intervals):
						extent = padding_intervals
					else:
						extent = len(forward_buf)	

					for i in range(extent):
						frame_chunk = forward_buf.pop(0)		
						write_clip(clip, frame_chunk)	

					# put whatever is left of the forward buffer onto the end of primary buffer
					buffer_chunks += forward_buf
					forward_buf = []

					clip.release()	
					#print("***WROTE CLIP TO DISK***")

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
					#
					#print("Changing state back to SCANNING...", flush=True)
					state = SCANNING     # complete state transition back to SCANNING

				else:   # positive detection in the forward buffer
    
					#print("  Positive detections in the forward buffer.", flush=True)

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
                                        #
					#print("  Flushing buffer", flush=True)
					for f in buffer_chunks:
						write_clip(clip, f)
					buffer_chunks = []

					#write_clip(clip, frame_chunk)

					last_forward_detection_idx = -1

					for i,f in enumerate(forward_buf):
						if(f["detection"]):
							last_forward_detection_idx = i

					#print("  Flushing all chunks in forward buffer up to and including the last_forward_detection_idx: ", last_forward_detection_idx)

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)


					for i in range(last_forward_detection_idx+1):
						write_clip(clip, forward_buf[i])
   
					if(last_forward_detection_idx < len(forward_buf)-1): 
						forward_buf = forward_buf[last_forward_detection_idx+1:]
					else:
						forward_buf = []
				
					buffer_chunks = forward_buf	    
					forward_buf = []

					# some debugging of buffers
					#debug_buffer = get_debug_buffer(buffer_chunks)
					#print("   Primary buffer: ", debug_buffer)
					#debug_buffer = get_debug_buffer(forward_buf)
					#print("   Forward Buffer: ", debug_buffer)
                                        #
					#print("\n")


			elif(state == DETECTION and frame_chunk["detection"] == True):

				if(len(buffer_chunks)>0):
					#print("Flushing Primary Buffer...")		    
					for ch in buffer_chunks:
						write_clip(clip, ch)
					buffer_chunks = []

				#print("state == DETECTION, and detection == TRUE", flush=True)
				write_clip(clip, frame_chunk)
	
			else:   # state == SCANNING, frame_chunk["detection"] == FALSE
				#print("state == SCANNING, detection == FALSE.  Continuing to see nothing....", flush=True) 

				# add this new chunk to the sliding window
				#print("Adding new chunk to sliding window...", flush=True)
				buffer_chunks.append(frame_chunk)
				if(len(buffer_chunks)>padding_intervals):
					buffer_chunks.pop(0)
    
		
			frame_chunk, success = get_video_chunk(invid, model, interval_frames, pu_lock)
			pbar.update(interval_frames)

			#with report_lock:
			#	report(pid, [filename, clip_path, fps, start_frame, end_frame, min_conf, max_conf, mean_conf])

		try:
			clip.release()
		except:
			break
			 
		invid.release()

	pbar.close()
	#clear_screen()



########################################
#
# Main Execution Section
#
#
def main():

	all_start_time = time.time()

	if(args.gpu == True):
		gpu_info = get_gpu_info()
		print("Detected %d CUDA GPUs" %(gpu_info[0]))
		for g in range(1,len(gpu_info)):
			mem_total, mem_used, mem_free = gpu_info[g]
			print("GPU:{}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(g-1, 100*mem_free/mem_total, human_size(mem_total), human_size(mem_free), human_size(mem_used)))

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

	print("          BEGINNING PROCESSING          ")
	print("*********************************************")
	print("         INPUT_DIR: ", args.input)
	print("        OUTPUT_DIR: ", args.output)
	print("     MODEL WEIGHTS: ", args.model)
	print(" SAMPLING INTERVAL: ", args.interval, "seconds")
	print("  PADDING DURATION: ", args.padding, "seconds")
	print("  CONCURRENT PROCS: ", args.jobs)
	print("           USE GPU: ", usegpu)
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


	while any(p.is_alive() for p in processes):
		for p in processes:
			if p.exitcode is not None and p.exitcode != 0:
				print(f"Terminating due to failure in process {p.pid}")

				for p in processes:
					p.terminate()

				time.sleep(2)
				clear_screen()
				reset_screen()

				print("\n")
				print("*****************************************************************************")
				print("SOMETHING WENT HORRIBLY WRONG:")
				print("Failure to run model within system resources (e.g. GPU RAM).")
				print("Please reduce the number of concurrent jobs (i.e., --jobs <n>) and try again!")
				print("*****************************************************************************")
				print("\n\n")

				return

		time.sleep(0.5)  # Check periodically

	clear_screen()
	reset_screen()

	print("Total time to process %d videos: %.02f seconds" %(len(files), time.time()-all_start_time))
	print("Report file saved to %s" %args.report)
	print("\nDONE\n")


if __name__ == '__main__':

	torch.multiprocessing.set_start_method('spawn')

	main()

