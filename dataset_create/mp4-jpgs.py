import cv2
from pathlib import Path
import re
import os, shutil
import utils
FRAMES_PER_IMAGE = 10

working_dir = 'data/'
video_dir = 'videos'
jpg_dir = 'jpgs'
# Opens the Video file
def video_to_jpg(vid_dir, jpg_dir, video):
    print("Video: " + video)
    cap = cv2.VideoCapture(vid_dir + '/' + video)
    filename = strip_mp4(video)
    os.mkdir(jpg_dir + '/' + filename)
    i=0
    while(cap.isOpened()):
       
        ret, frame = cap.read()
        if ret == False:
            break
        if (i % FRAMES_PER_IMAGE != 0):
            i+=1
            continue
        cv2.imwrite(jpg_dir + '/' + filename + '/' + filename + "_" + str(i)+'.jpg',frame)
        i+=1
     
    cap.release()
    cv2.destroyAllWindows()
    print("done\n")

def strip_mp4(video):
    return re.sub(r'\.mp4$','', video)

def run_in_dirs(vid_dir, jpg_dir):
    print("Clearing jpg directory")
    utils.clear_directory(jpg_dir)
    videos_dir = Path(vid_dir)
    print("Start iteration\n")
    for video in videos_dir.iterdir():
        video_to_jpg(vid_dir, jpg_dir, video.name)


def main():
    run_in_dirs(working_dir + video_dir, working_dir + jpg_dir)

main()
