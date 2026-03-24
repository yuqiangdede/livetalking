from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
import face_detection
from pathlib import Path


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=4)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def _load_batch_images(image_paths):
    batch_images = []
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f'Failed to read image: {image_path}')
        batch_images.append(frame)
    return batch_images


def face_detect(image_paths):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		results = []
		try:
			for i in tqdm(range(0, len(image_paths), batch_size), desc='face detect'):
				batch_paths = image_paths[i:i + batch_size]
				batch_images = _load_batch_images(batch_paths)
				batch_predictions = detector.get_detections_for_batch(np.array(batch_images))
				pady1, pady2, padx1, padx2 = args.pads
				for rect, image in zip(batch_predictions, batch_images):
					if rect is None:
						cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
						raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

					y1 = max(0, rect[1] - pady1)
					y2 = min(image.shape[0], rect[3] + pady2)
					x1 = max(0, rect[0] - padx1)
					x2 = min(image.shape[1], rect[2] + padx2)
					results.append([x1, y1, x2, y2])
				del batch_predictions
				del batch_images
		except RuntimeError as exc:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			print('Recovering from OOM error; New batch size: {} ({})'.format(batch_size, exc))
			continue
		break

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)

	del detector
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return boxes

if __name__ == "__main__":
    avatar_path = PROJECT_ROOT / "data" / "avatars" / args.avatar_id
    full_imgs_path = avatar_path / "full_imgs"
    face_imgs_path = avatar_path / "face_imgs"
    coords_path = avatar_path / "coords.pkl"
    osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
    print(args)

    #if os.path.isfile(args.video_path):
    video2imgs(args.video_path, str(full_imgs_path), ext = 'png')
    input_img_list = sorted(glob(os.path.join(str(full_imgs_path), '*.[jpJP][pnPN]*[gG]')))
    print('prepared {} full frames'.format(len(input_img_list)))
    face_det_boxes = face_detect(input_img_list)
    coord_list = []
    idx = 0
    print('writing face crops...')
    for img_path, (x1, y1, x2, y2) in tqdm(zip(input_img_list, face_det_boxes), total=len(input_img_list), desc='face crop'):
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f'Failed to read image: {img_path}')
        coords = (int(y1), int(y2), int(x1), int(x2))
        cropped = frame[coords[0]: coords[1], coords[2]: coords[3]]
        if cropped.size == 0:
            raise ValueError(f'Invalid face crop for image: {img_path}')

        resized_crop_frame = cv2.resize(cropped,(args.img_size, args.img_size)) #,interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(str(face_imgs_path / f"{idx:08d}.png"), resized_crop_frame)
        coord_list.append(coords)
        idx = idx + 1

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
