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
####################################################################


import os, sys, time, pathlib
import argparse
from multiprocessing import Pool, freeze_support, RLock
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
from general import non_max_suppression


DEFAULT_MODEL           = 'md_v5a.0.0.pt'
DEFAULT_INTERVAL        = 30  # number of frames between samples
DEFAULT_BUFFER_TIME     = 5   # number of seconds of video to include before first detection and after last detection
DEFAULT_REPORT_FILENAME = "report.csv"
DEFAULT_NPROCS          = 1
DEFAULT_BATCH_SIZE      = 8

STALE_LOCK_CLEANUP	= 1 # hour


parser = argparse.ArgumentParser(prog='tigervid', description='Analyze videos and extract clips and metadata which contain animals.')

parser.add_argument('input',  metavar='INPUT_DIR',  default="input",  help='Path to input directory containing MP4 videos')
parser.add_argument('output', metavar='OUTPUT_DIR', default="output", help='Path to output directory for clips and metadatas')

parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Path to the PyTorch model weights file (DEFAULT: '+DEFAULT_MODEL+')')
parser.add_argument('-i', '--interval', type=int, default=DEFAULT_INTERVAL, help='Number of frames to read between sampling with AI (DEFAULT: '+str(DEFAULT_INTERVAL)+')')
parser.add_argument('-b', '--buffer', type=int, default=DEFAULT_BUFFER_TIME, help='Number of seconds to prepend and append to clip (DEFAULT: '+str(DEFAULT_BUFFER_TIME)+')')
parser.add_argument('-r', '--report', default=DEFAULT_REPORT_FILENAME, help='Name of report metadata (DEFAULT: '+DEFAULT_REPORT_FILENAME+')')
parser.add_argument('-j', '--jobs', type=int, default=DEFAULT_NPROCS, help='Number of concurrent (parallel) processes (DEFAULT: '+str(DEFAULT_NPROCS)+')')
parser.add_argument('-s', '--batchsize', type=int, default=DEFAULT_BATCH_SIZE, help='The batch size for inference (DEFAULT: '+str(DEFAULT_BATCH_SIZE)+')')

group = parser.add_mutually_exclusive_group()
group.add_argument('-g', '--gpu', action='store_true',  default=True, help='Use GPU if available (DEFAULT)')
group.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU only')

args = parser.parse_args()



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
	if __name__ == '__main__':
	    print("Using CPU")
	usegpu = False
else:
	if(torch.cuda.is_available()):
		device = "cuda"
		usegpu = True
		if __name__ == '__main__':
		    print("Using GPU")
	else:
		device = "cpu"
		usegpu = False

torch.device(device)

if (usegpu==False):
	if __name__ == '__main__':
	    print("Forcing batchsize=1 (using CPU)")
	args.batchsize = 1

if __name__ != '__main__':
	logging.getLogger('torch.hub').setLevel(logging.ERROR)
	model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model, _verbose=False, verbose=False)
	model.to(device)


def stale_lock_cleanup():
	curr_time = time.time()

	files = glob.glob(".*.lck")
	for f in files:
		p = pathlib.Path(f)
		st = p.stat()
		if(curr_time - st.st_mtime > 60*60*STALE_LOCK_CLEANUP):
			print("Removing stale lock file: ", f)
			os.remove(f)




def report(report_lockfile, pid, report_list):
	# get the lock and wait for it if we have to
	cnt = 0
	while(True):
		rl = lock_get(report_lockfile)
		if (rl < 0) or (rl == pid):
			lock_set(report_lockfile, pid) 
			break
		else:
			time.sleep(0.25)

			if(cnt>100):
				print("Error:  Could not acquire report lock file %s" %(report_lockfile))
				exit(-1)

			cnt+=1

	try:
		report_file = open(args.report, "a")
	except:
		print("Error:  Could not open report file %s for writing in report()" %(args.report))
		exit(-1)

	filename,clip_path,fps,start_frame,end_frame,min_conf,max_conf,mean_conf = report_list
	s = "\"%s\", \"%s\", %d, %f, %d, %f, %d, %f, %.02f, %.02f, %.02f\n" %(filename, clip_path, start_frame, start_frame/fps, end_frame, end_frame/fps, end_frame-start_frame, (end_frame-start_frame)/fps, min_conf, max_conf, mean_conf)

	report_file.write(s)	

	report_file.flush()
	report_file.close()

	lock_release(report_lockfile, pid)


def locknames(session):
	gpu_lockfile    = ".gpu_"+session+".lck"
	report_lockfile = ".rpt_"+session+".lck"		

	return(gpu_lockfile, report_lockfile)



def lock_release(lockfile, pid):
	if(os.path.isfile(lockfile)):
		os.remove(lockfile) 


def lock_set(lockfile, pid):
	lock = open(lockfile, "w")
	lock.write(str(pid)+"\n")
	lock.flush()
	lock.close()


def lock_get(lockfile):
	try:
		lock = open(lockfile, "r")
		pid = int(lock.readline())
	except:
		pid = -1

	return pid


def label(img, frame, fps):
	s = "frame: %d, time: %s" %(frame, "{:0.3f}".format(frame/fps))
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0,0,0), 6, cv2.LINE_AA) 	
	cv2.putText(img, s, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 3, cv2.LINE_AA) 	
	return(img)

def split(a, n):
	k, m = divmod(len(a), n)
	return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def pairwise(iterable):
	it = iter(iterable)
	a = next(it, None)

	for b in it:
	    yield (a, b)
	    a = b

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
        chunks.append(filenames[start_index:end_index])
        start_index = end_index

    return chunks




def process_chunk(session, pid, chunk):

	global args

	gpu_lockfile, report_lockfile = locknames(session)


	# lets pace ourselves to help avoid race conditions
	time.sleep(pid*2)

	fcnt = 1
	for filename in chunk:
	    clear_screen() 
	    if('pbar' in locals()):
		    pbar.refresh() 

	    # Use imageio[ffmpeg] to determine the number of frames	    
	    while(True):
		    try:
			    v=imageio.get_reader(filename,  'ffmpeg')
			    nframes  = v.count_frames()
			    metadata = v.get_meta_data()
			    fps = metadata['fps']
			    duration = metadata['duration']
			    size = metadata['size']

			    break
		    except:
			    print("imageio timeout.  Trying again")
			    time.sleep(1)

	    (width,height) = size
	    buffer_frames = int(fps*args.buffer)

	    # pre-allocate our inference buffer
	    inference_buffer_size = (int(nframes/args.interval)+1,640,640,3)
	    inference_buffer      = np.empty(inference_buffer_size, dtype=np.uint8)

	    try:
		    invid = cv2.VideoCapture(filename)
	    except:
		    print("Could not read video file: ", filename, " skipping...")
		    continue

	    count        = 0
	    tiger_frames = 0
	    detections   = []

	    nbatches = (math.ceil(nframes/args.interval))

	    #print("Sampling video...")
	    pbar = tqdm(total=nframes,position=pid,ncols=100,unit=" frames",leave=True,file=sys.stdout)

	    pbar.set_description("pid=%s Reading File %d/%d: %s" %(str(pid).zfill(2),fcnt,len(chunk),filename))


	    #
	    # Sample frames from the video at the specified sampling interval
	    # and put in a buffer
	    #
	    count = 0	
	    for i in range(nframes):
		    success, image = invid.read()
		    if success:
			    if((i % args.interval)==0):
				    inference_buffer[count] = cv2.resize(image, (640,640))
				    count += 1
			    if((i % 5000)==0):
				    clear_screen()
				    pbar.refresh()
		    else:
			    break
	    
		    pbar.update(1)


	    clear_screen()
	    pbar.reset(total=nbatches*args.interval)	
	    pbar.refresh()

	    if(usegpu==True):
		    pbar.set_description("pid=%s  WAITING for GPU: %s" %(str(pid).zfill(2),filename))
	    else:
		    pbar.set_description("pid=%s  WAITING for CPU: %s" %(str(pid).zfill(2),filename))

	    while(True):
		    lck = lock_get(gpu_lockfile)
		    
		    if lck < 0 or lck == pid:
			    lock_set(gpu_lockfile, pid)
			    break
		    else:
			    pbar.refresh()
			    time.sleep(0.5)

	    pbar.set_description("pid=%s AI Detection %d/%d: %s" %(str(pid).zfill(2),fcnt,len(chunk),filename))
	    pbar.refresh()


	    # Iterate over the array to copy batches
	    tiger_frames = {}	
	    for b in range(nbatches):
	    
		    start_idx = b * args.batchsize
		    end_idx = start_idx + args.batchsize
		    if(end_idx > count):
			    end_idx = count
		    batch_images = inference_buffer[start_idx:end_idx] 
	    
		    it = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float() / 255.0  
    
		    with(torch.no_grad()):
			    with(autocast()):
				    image_tensors = it.to(device)
		    
				    try:
					    detections_tensor = model(image_tensors)
					    detections = non_max_suppression(detections_tensor)
				    except RuntimeError as e:
					    if "CUDA out of memory" in str(e):
						    print("CUDA out of memory error encountered.")
						    exit(0)

		    for i, d in enumerate(detections):
			    frame_idx = (b*args.batchsize+i)*args.interval
			    dn = d.cpu().detach().numpy()
			    if len(dn):
				    tiger_frames[frame_idx] = dn
			    del dn

		    if(usegpu):
			    torch.cuda.empty_cache()

		    pbar.update(args.interval)

	    if(usegpu==True):
		    torch.cuda.empty_cache()

	    lock_release(gpu_lockfile, pid)
		    
	    groups = dict(enumerate(grouper(tiger_frames.keys()), 0))

	    #
	    # Merge groups which overlap due to buffer_frames
	    #
	    extents = []
	    for g in groups:
		    start_frame = groups[g][0]-buffer_frames
		    if(start_frame < 0):
			    start_frame = 0

		    end_frame = groups[g][len(groups[g])-1]+buffer_frames
		    if(end_frame >= nframes):
			    end_frame = nframes-1
	
		    extents.append([start_frame,end_frame])

	    dels = []
	    for i in range(len(extents) - 1):
		    cur = extents[i]
		    nxt = extents[i+1]

		    if(cur[1] >= nxt[0]):
			    extents[i+1][0]=cur[0]
			    dels.append(i)

	    new_extents = []
	    cnt = 0
	    for i in range(len(extents)):
		    if(not i in dels):
			    new_extents.append(extents[i])
	

	    i = 0
	    new_groups = {}
	    tkeys = list(tiger_frames.keys())
	    for e in new_extents:
		    min_index = bisect.bisect_left(tkeys,  e[0])
		    max_index = bisect.bisect_right(tkeys, e[1])	

		    new_groups[i] = []
		    for c in range(min_index, max_index):
			    new_groups[i].append(tkeys[c])
		    
		    i+=1


	    for i,g in enumerate(new_groups):
		    pbar.set_description("pid=%s  Saving Clip %d/%d: %s" %(str(pid).zfill(2),i+1,len(new_groups),filename))
		    #clear_screen()
		    pbar.refresh()

		    min_conf, max_conf, mean_conf = confidence(new_groups[g], tiger_frames) 

		    fn = os.path.basename(filename)
		    clip_name = os.path.splitext(fn)[0] + "_{:03d}".format(g) + ".mp4"
		    clip_path = os.path.join(args.output, clip_name)

		    fourcc = cv2.VideoWriter_fourcc(*'mp4v')	
		    outvid = cv2.VideoWriter(clip_path, fourcc, fps, (width,height))

		    start_frame = new_groups[g][0]-buffer_frames
		    if(start_frame < 0):
			    start_frame = 0
	    
		    end_frame = new_groups[g][len(new_groups[g])-1]+buffer_frames
		    if(end_frame >= nframes):
			    end_frame = nframes-1;

		    pbar.reset(total=end_frame-start_frame)	
		    invid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
		    for f in range(start_frame, end_frame):
			    pbar.refresh()
			    success, image = invid.read()
			    if(success):
				    outvid.write(label(image,f,fps))
				    pbar.update(1)
			    else:
				    break
		    outvid.release()

		    report(report_lockfile, pid, [filename, clip_path, fps, start_frame, end_frame, min_conf, max_conf, mean_conf])
	    
	    invid.release()
	    clear_screen()
	    pbar.refresh()
	    fcnt += 1

	pbar.close()



########################################
#
# Main Execution Section
#
#
def main():

	all_start_time = time.time()

	stale_lock_cleanup()

	session = str(uuid.uuid4())
	gpu_lockfile, report_lockfile = locknames(session)

	freeze_support()  # For Windows support - multiprocessing with tqdm

	# release our lock files before starting
	lock_release(gpu_lockfile,    -1)
	lock_release(report_lockfile, -1)

	try:
		report_file = open(args.report, "w")
	except:
		print("Error: Could not open report file %s in main()" %(args.report))
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
	''')

	print("           BEGINNING PROCESSING          ")
	print("*********************************************")
	print("        INPUT_DIR: ", args.input)
	print("       OUTPUT_DIR: ", args.output)
	print("    MODEL WEIGHTS: ", args.model)
	print("SAMPLING INTERVAL: ", args.interval, "frames")
	print("  BUFFER DURATION: ", args.buffer, "seconds")
	print(" CONCURRENT PROCS: ", args.jobs)
	print("       BATCH SIZE: ", args.batchsize)
	print("          USE GPU: ", usegpu)
	print("*********************************************\n\n")

	path = os.path.join(args.input, "*.mp4")
	files = glob.glob(path)
	random.shuffle(files)
	ch = chunks(files,args.jobs)

	pool = Pool(processes=args.jobs, initargs=(RLock(),), initializer=tqdm.set_lock)
	jobs = [pool.apply_async(process_chunk, args=(session, i, c)) for i,c in enumerate(ch)]

	if(usegpu==True):
		torch.cuda.empty_cache()

	r = [job.get() for job in jobs]
	pool.close()

	#clear_screen()	
	reset_screen()	
    
	lock_release(gpu_lockfile,    -1)
	lock_release(report_lockfile, -1)

	print("Total time to process %d videos: %.02f seconds" %(len(files), time.time()-all_start_time))
	print("Report file saved to %s" %args.report)
	print("\nDONE\n")


if __name__ == '__main__':

	torch.multiprocessing.set_start_method('spawn')

	main()

