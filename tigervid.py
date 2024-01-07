####################################################################
#
# tigervid.py
#
# Luke Sheneman
# sheneman@uidaho.edu
# December 2023
#
# Given a directory of videos, process each video to look for animals
# Extracts video clips which include animals into destination directory
# Writes summary log
#
####################################################################


import os, sys, time, pathlib
import argparse
from multiprocessing import Process, freeze_support
#from torch.multiprocessing import freeze_support, Lock,  Manager, Pool
import queue
import cv2
import math
import nvidia_smi
import logging
import resource
import random
import torch
import glob
import numpy as np
import imageio
import signal
from tqdm import tqdm	


DEFAULT_INPUT_DIR	= "inputs"
DEFAULT_OUTPUT_DIR	= "outputs"
DEFAULT_LOGGING_DIR 	= "logs"

DEFAULT_MODEL            = 'md_v5a.0.0.pt'
DEFAULT_INTERVAL         = 1.0   # number of seconds between samples
DEFAULT_PADDING		 = 5.0   # number of seconds of video to include before first detection and after last detection in a clip
DEFAULT_REPORT_FILENAME  = "report.csv"
DEFAULT_WORKERS          = 4
DEFAULT_NOBAR		 = False
DEFAULT_MAX_FD		 = 16384

YOLODIR = "yolov5"


parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default=DEFAULT_INPUT_DIR,  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default=DEFAULT_OUTPUT_DIR, help='Path to output directory for clips and metadata')

parser.add_argument('-m', '--model',	type=str,   default=DEFAULT_MODEL,           help='Path to the PyTorch model weights file (DEFAULT: '+DEFAULT_MODEL+')')
parser.add_argument('-i', '--interval', type=float, default=DEFAULT_INTERVAL,        help='Number of seconds between AI sampling/detection (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-p', '--padding',  type=float, default=DEFAULT_PADDING,         help='Number of seconds of video to pad on front and end of a clip (DEFAULT: '+str(DEFAULT_PADDING)+')')
parser.add_argument('-r', '--report',   type=str,   default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-w', '--workers',	type=int,   default=DEFAULT_WORKERS,         help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_WORKERS)+')')
parser.add_argument('-l', '--logging',  type=str,   default=DEFAULT_LOGGING_DIR,     help='The directory for log files (DEFAULT: '+str(DEFAULT_LOGGING_DIR)+')')

parser.add_argument('-n', '--nobar',    action='store_true',  default=DEFAULT_NOBAR,     help='Turns off the Progress Bar during processing.  (DEFAULT: Use Progress Bar)')

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




def report(pid, report_list):


	filename,clip_path,fps,start_frame,end_frame,confidences = report_list

	min_conf  = min(confidences)
	max_conf  = max(confidences)
	mean_conf = 0 if len(confidences) == 0 else sum(confidences)/len(confidences)

	s = "\"%s\", \"%s\", %d, %.02f, %d, %.02f, %d, %.02f, %.02f, %.02f, %.02f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps, min_conf, max_conf, mean_conf)

	try:
		report_file = open(args.report, "a")
		report_file.write(s)	
		report_file.flush()
		report_file.close()

	except:
		print("Warning:  Could not open report file %s for writing in report()" %(args.report), flush=True)

	return(clip_path)




def signal_handler(signum, frame):
	print(f"Received signal {signum}!", flush=True)

	if(signum == 15):
		print(f"Intercepted SIGTERM.  Ignoring...", flush=True)
		time.sleep(1)
		return

	else:
		exit(0)



def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)


def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def reset_screen():
	if(os.name != 'nt'):
		os.system('reset')

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
	return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def get_gpu_info():

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



def inference_worker(frame_queue):

	#torch.multiprocessing.set_start_method('spawn')

	if(args.cpu==True):
		device = "cpu"
		torch.device(device)
		print("Using CPU", flush=True)
		usegpu = False
	else:
		if(torch.cuda.is_available()):
			device = "cuda"
			usegpu = True
			print("Using GPU", flush=True)
		else:
			device = "cpu"
			usegpu = False

	torch.device(device)

	if(usegpu == True):
		gpu_info = get_gpu_info()
		print("Detected %d CUDA GPUs" %(gpu_info[0]))
		for g in range(1,len(gpu_info)):
			mem_total, mem_used, mem_free = gpu_info[g]
			print("GPU:{}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(g-1, 100*mem_free/mem_total, human_size(mem_total), human_size(mem_free), human_size(mem_used)))

	
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



	while True:
		try:
			frame = frame_queue.get(timeout=90)  # adjust timeout as needed
			if frame is None:  # using None as the signal to stop
				break

			# Perform inference on the frame
			result = your_ai_library.infer(frame)

			# Process the result as needed

		except queue.Empty:
			continue
	
	





#
# retrieves chunk of video frames of size interval_sz
# returns: 
#     result as dict with keys:  {frame_buffer, detection (boolean), confidence score}
#     success (True/False) 
#
def get_video_chunk(pid, invid, model, interval_sz, pu_lock):

	global chunk_idx

	print("pid=%s: Getting chunk: %d" %(str(pid).zfill(2),chunk_idx))

	res = {}
	res["chunk_idx"] = chunk_idx

	buf = []
	for i in range(interval_sz):
		success, image = invid.read()
		if(success):
			buf.append(image)
		else:
			print("Error:  Could not read frame chunk: %d" %chunk_idx)
			chunk_idx += 1
			return(None, False)
			
	print("pid=%s: got video chunk %d" %(str(pid).zfill(2),chunk_idx), flush=True)

	inference_frame = cv2.resize(image, (640,640))
	
	if(True):
		try:
			print("pid=%s: starting inference on chunk %d..." %(str(pid).zfill(2),chunk_idx), flush=True)
			results = model(inference_frame).pandas().xyxy[0]
			print("pid=%s: inference done on chunk %d" %(str(pid).zfill(2),chunk_idx), flush=True)
		except:
			print("Error: Could not run model inference on frame from chunk index: %d" %chunk_idx)
			sys.exit(-1)
			#chunk_idx += 1
			#return(None, False)


	if(results.shape[0]):
		detection = True
		confidence = results["confidence"].mean()
	else:
		detection = False
		confidence = None
    
	#print("----> Detection is [%s] for chunk index: %d" %(str(detection), chunk_idx))

	res["buffer"]	  = buf
	res["detection"]  = detection
	res["confidence"] = confidence

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
		try:
			clip.write(frame)
		except:
			print("Failed to write to clip in write_clip().  Exception calling clip.write()", flush=True)
		


def get_debug_buffer(frame_chunk):

	debug_info = ""
	for fc in frame_chunk: 
		debug_info += "[%d|%s] " %(fc["chunk_idx"],fc["detection"]) 

	return(debug_info)



def process_chunk(pid_chunk_pair, pu_lock, report_lock):

	global args
	global model
	global chunk_idx
	global most_recent_written_chunk

	pid, chunk = pid_chunk_pair

	resource.setrlimit(resource.RLIMIT_NOFILE, (DEFAULT_MAX_FD, DEFAULT_MAX_FD))

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	# lets pace ourselves on startup to help avoid general race conditions
	time.sleep(pid*1)

	for fcnt, filename in enumerate(chunk):

		while(True):
			try:
				print("pid=%s:, imageio() start" %(str(pid).zfill(2)), flush=True)
				v=imageio.get_reader(filename,  'ffmpeg')
				nframes  = v.count_frames()
				metadata = v.get_meta_data()
				v.close()

				fps = metadata['fps']
				duration = metadata['duration']
				size = metadata['size']
				
				print("pid=%s: imageio() end" %(str(pid).zfill(2)), flush=True)
	
				break
			except:
				print("WARNING: imageio timeout.   Trying again.", flush=True)
				time.sleep(0.25)

		(width,height) = size

		try:
			print("pid=%s. open VideoCapture()" %(str(pid).zfill(2)), flush=True)
			invid = cv2.VideoCapture(filename)
			print("pid=%s. VideoCapture() returned" %(str(pid).zfill(2)), flush=True)
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
		confidences   = []

		most_recent_written_chunk = -1
	
		#print("NUMBER OF FRAMES: ", nframes)
		#print("NUMBER OF CHUNKS: ", nchunks)
		#print("FRAMES PER INTERVAL: ", interval_frames)	
		#print("PADDING INTERVALS: ", padding_intervals)

		latest_reported_clip_path = ""

		#clear_screen()
		if(args.nobar):
			print("pid=%s: Processing video %d/%d: %s" %(str(pid).zfill(2),fcnt+1,len(chunk),filename), flush=True)
		else:
			pbar = tqdm(total=nframes,position=pid,ncols=100,unit=" frames",leave=False,mininterval=0.5,file=sys.stdout)
			pbar.set_description("pid=%s: Processing video %d/%d: %s" %(str(pid).zfill(2),fcnt+1,len(chunk),filename))

		frame_chunk, success = get_video_chunk(pid, invid, model, interval_frames, pu_lock)
		if(frame_chunk["detection"] == True):
			confidences.append(frame_chunk["confidence"])
		
		if(not args.nobar):	
			pbar.update(interval_frames)

		while(success):
	
			#print("CHUNK_IDX: %d/%d" %(chunk_idx, nchunks))
			if(chunk_idx > nchunks):
				print("PID=%d, OUT OF CHUNKS TO READ.  BAILING" %pid, flush=True)
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
				try:
					clip = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))
				except:
					print("Could not create a clip via cv2.VideoWriter!", flush=True)
					continue

				clip_number += 1
			
				# track the first frame of the clip for export to metadata report
				if(len(buffer_chunks)>0):
					clip_start_frame = buffer_chunks[0]["chunk_idx"]*interval_frames
				else:
					clip_start_frame = 0
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
					frame_chunk, success = get_video_chunk(pid, invid, model, interval_frames, pu_lock)
					if(success and frame_chunk["detection"] == True):
						confidences.append(frame_chunk["confidence"])
					if(not args.nobar):
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

					## WRITE CLIP TO DISK AND LOG
					clip.release()	
					clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
					with report_lock:
						if(clip_path != latest_reported_clip_path):  # make sure we don't record the same clip 2x
							latest_reported_clip_path = report(pid, [filename, clip_path, fps, clip_start_frame, clip_end_frame, confidences])
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
    
		
			frame_chunk, success = get_video_chunk(pid, invid, model, interval_frames, pu_lock)
			if(success and frame_chunk["detection"] == True):
				confidences.append(frame_chunk["confidence"])
			if(not args.nobar):
				pbar.update(interval_frames)


		try:
			clip.release()
			clip_end_frame = (most_recent_written_chunk * interval_frames) + interval_frames
			with report_lock:
				if(clip_path != latest_reported_clip_path):  # make sure we don't record the same clip 2x
					latest_reported_clip_path = report(pid, [filename, clip_path, fps, clip_start_frame, clip_end_frame, confidences])
		except:
			break
			 
		invid.release()

	if(not args.nobar):
		pbar.close()
	#clear_screen()



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

	print("            BEGINNING PROCESSING          ")
	print("*********************************************")
	print("           INPUT_DIR: ", args.input)
	print("          OUTPUT_DIR: ", args.output)
	print("       MODEL WEIGHTS: ", args.model)
	print("   SAMPLING INTERVAL: ", args.interval, "seconds")
	print("    PADDING DURATION: ", args.padding, "seconds")
	print("  CONCURRENT WORKERS: ", args.workers)
	print("DISABLE PROGRESS BAR: ", args.nobar)
	print("             USE GPU: ", usegpu)
	print("         REPORT FILE: ", args.report)
	print("*********************************************\n\n", flush=True)

	signal.signal(signal.SIGINT,  signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	path = os.path.join(args.input, "*.mp4")
	files = glob.glob(path)
	random.shuffle(files)
	chs = chunks(files,args.workers)

	frame_queue  = multiprocessing.Queue(maxsize=1024)
	report_queue = multiprocessing.Queue(maxsize=args.workers*2)

	# instantiate the AI inference process
	inference_process = multiprocessing.Process(target=inference_worker, args=(frame_queue,))
	inference_process.start()

	# instantiate the report process
	reporting_process = multiprocessing.Process(target=reporting_worker, args=(report_queue,))
	reporting_process.start()

	# instantiate streaming workers
	streaming_workers = []
	for ch in chs:
		p = multiprocessing.Process(target=streaming_worker, args=(frame_queue, ch))
		p.start()
		streaming_workers.append(p)	

	# Wait for all processes to complete
	for p in streaming_workers:
		p.join()
	inference_process.join()
	reporting_process.join()


	if(not args.nobar):	
		clear_screen()
		reset_screen()
	else:
		print("\n")

	print("Total time to process %d videos: %.02f seconds" %(len(files), time.time()-all_start_time))
	print("Report file saved to %s" %args.report)
	print("\nDONE\n")


if __name__ == '__main__':


	main()

