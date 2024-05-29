import cv2
from pathlib import Path
import re
import os, shutil

FRAMES_PER_IMAGE = 10

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
        cv2.imwrite(jpg_dir + '/' + filename + '/' +str(i)+'.jpg',frame)
        i+=1
     
    cap.release()
    cv2.destroyAllWindows()
    print("done\n")

def strip_mp4(video):
    return re.sub(r'\.mp4$','', video)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def run_in_dirs(vid_dir, jpg_dir):
    print("Clearing jpg directory")
    clear_directory(jpg_dir)
    videos_dir = Path(vid_dir)
    print("Start iteration\n")
    for video in videos_dir.iterdir():
        video_to_jpg(vid_dir, jpg_dir, video.name)


def main():
    run_in_dirs('videos_beta', 'jpgs_beta')

main()
