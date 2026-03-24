###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

#from .utils import *
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp


from .wav2lip_asr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from ..core.base_real import BaseReal

#from imgcache import ImgCache

from tqdm import tqdm
from ..utils.app_logger import logger

device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
model_dtype = torch.float16 if device == "cuda" else torch.float32
print('Using {} for inference.'.format(device))


def _resolve_avatar_path(avatar_root: str, avatar_id: str) -> str:
    """
    Resolve avatar directories with a small compatibility fallback.

    The project historically uses both `avatar_2` and `avatar2` style names.
    We prefer the exact name first, then retry a normalized variant.
    """
    candidate_ids = [avatar_id]
    if "_" in avatar_id:
        candidate_ids.append(avatar_id.replace("_", ""))
    else:
        candidate_ids.append(avatar_id)

    seen = set()
    for candidate in candidate_ids:
        if candidate in seen:
            continue
        seen.add(candidate)
        avatar_path = os.path.join(avatar_root, candidate)
        if os.path.isdir(avatar_path):
            return avatar_path

    available = []
    if os.path.isdir(avatar_root):
        available = sorted(
            entry.name
            for entry in os.scandir(avatar_root)
            if entry.is_dir()
        )
    raise FileNotFoundError(
        f"Avatar directory not found for '{avatar_id}'. "
        f"Tried: {', '.join(os.path.join(avatar_root, c) for c in seen)}. "
        f"Available avatars: {available}"
    )

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path) #,weights_only=True
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	logger.info("Load checkpoint from: %s", path)
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	if model_dtype == torch.float16:
		model = model.half()
		logger.info("Wav2Lip inference dtype: float16")
	else:
		logger.info("Wav2Lip inference dtype: float32")
	first_param = next(model.parameters(), None)
	if first_param is not None:
		logger.info("Wav2Lip model device: %s", first_param.device)
	return model.eval()

def load_avatar(avatar_id, avatar_root="./data/avatars"):
    avatar_path = _resolve_avatar_path(avatar_root, avatar_id)
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    silent_frame = _load_silent_frame(avatar_path, frame_list_cycle)

    return frame_list_cycle,face_list_cycle,coord_list_cycle,silent_frame

def _resolve_silent_frame_path(avatar_path: str) -> str | None:
    candidate_names = [
        "silent.png",
        "silent.jpg",
        "silent.jpeg",
        "silence.png",
        "idle.png",
        "idle.jpg",
        "neutral.png",
        "still.png",
        "closemouth.png",
        "closed_mouth.png",
        "mouth_closed.png",
    ]
    for name in candidate_names:
        candidate = os.path.join(avatar_path, name)
        if os.path.isfile(candidate):
            return candidate
    return None

def _load_silent_frame(avatar_path: str, frame_list_cycle):
    silent_frame_path = _resolve_silent_frame_path(avatar_path)
    if silent_frame_path:
        silent_frame = cv2.imread(silent_frame_path)
        if silent_frame is not None:
            if frame_list_cycle:
                base_h, base_w = frame_list_cycle[0].shape[:2]
                if silent_frame.shape[:2] != (base_h, base_w):
                    silent_frame = cv2.resize(silent_frame, (base_w, base_h))
            return silent_frame
    if frame_list_cycle:
        return copy.deepcopy(frame_list_cycle[0])
    raise ValueError("Avatar has no frames to use as a silence fallback.")

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres, device=device, dtype=model_dtype)
    mel_batch = torch.ones(batch_size, 1, 80, 16, device=device, dtype=model_dtype)
    model(mel_batch, img_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    if size <= 0:
        return 0
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 


def _safe_cycle_index(size: int, index: int) -> int:
    if size <= 0:
        return 0
    return index % size

def inference(quit_event,batch_size,face_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,model):
    
    #model = load_model("./models/wav2lip.pth")
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)
    
    #input_latent_list_cycle = torch.load(latents_out_path)
    length = len(face_list_cycle)
    index = 0
    count=0
    counttime=0
    logged_runtime_device = False
    logger.info('start inference')
    if length <= 0:
        logger.error("Avatar face frame list is empty")
        return
    while not quit_event.is_set():
        starttime=time.perf_counter()
        mel_batch = []
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
            
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type==0:
                is_all_silence=False

        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # print('infer=======')
            t=time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                face = face_list_cycle[_safe_cycle_index(length, idx)]
                img_batch.append(face)
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            img_batch = torch.tensor(np.transpose(img_batch, (0, 3, 1, 2)), dtype=model_dtype, device=device)
            mel_batch = torch.tensor(np.transpose(mel_batch, (0, 3, 1, 2)), dtype=model_dtype, device=device)
            if not logged_runtime_device:
                logger.info(
                    "Wav2Lip runtime tensors: img=%s/%s mel=%s/%s",
                    img_batch.device,
                    img_batch.dtype,
                    mel_batch.device,
                    mel_batch.dtype,
                )
                logged_runtime_device = True

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.

            counttime += (time.perf_counter() - t)
            count += batch_size
            #_totalframe += 1
            if count>=25:
                logger.info(
                    "lip-sync infer fps=%.4f batch=%s feat_q=%s res_q=%s",
                    count / counttime,
                    batch_size,
                    audio_feat_queue.qsize(),
                    res_frame_queue.qsize(),
                )
                count=0
                counttime=0
            for i,res_frame in enumerate(pred):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            #print('total batch time:',time.perf_counter()-starttime)            
    logger.info('lipreal inference processor stop')

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        # self.W = opt.W
        # self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        self.model = model
        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle,self.silence_frame = avatar

        self.asr = LipASR(opt,self)
        self.asr.warm_up()
        
        self.render_event = mp.Event()
    
    # def __del__(self):
    #     logger.info(f'lipreal({self.sessionid}) delete')

    def paste_back_frame(self,pred_frame,idx:int):
        frame_count = len(self.frame_list_cycle)
        coord_count = len(self.coord_list_cycle)
        if frame_count <= 0 or coord_count <= 0:
            raise ValueError("Avatar frames or coordinates are empty")
        safe_idx = _safe_cycle_index(min(frame_count, coord_count), idx)
        bbox = self.coord_list_cycle[safe_idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[safe_idx])
        #combine_frame = copy.deepcopy(self.imagecache.get_img(idx))
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        #combine_frame = get_image(ori_frame,res_frame,bbox)
        #t=time.perf_counter()
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame

    def get_silence_frame(self, idx, audio_frames):
        if not self.silence_gate_enabled:
            return None
        return copy.deepcopy(self.silence_frame)

    def reload_avatar(self, avatar_id):
        self.opt.avatar_id = avatar_id
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle, self.silence_frame = load_avatar(
            avatar_id,
            self.opt.AVATAR_DIR,
        )
        self.idx = 0
        return avatar_id

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        output_count = 0
        output_time = 0.0
        enable_transition = False

        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1
            _last_silent_frame = None
            _last_speaking_frame = None

        while not quit_event.is_set():
            try:
                frame_start = time.perf_counter()
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            speech_event = None
            for _, _, eventpoint in audio_frames:
                if isinstance(eventpoint, dict) and eventpoint.get("status") == "end":
                    speech_event = eventpoint
                    break
                if speech_event is None and isinstance(eventpoint, dict) and eventpoint.get("status") == "start":
                    speech_event = eventpoint

            was_speaking = self.speaking
            now_speaking = not (audio_frames[0][1] != 0 and audio_frames[1][1] != 0)
            if now_speaking and not was_speaking:
                self._speech_started_at = time.perf_counter()
                self._speech_text = (speech_event or {}).get("text", "")
                self._speech_dialog_id = str((speech_event or {}).get("dialog_id", ""))
                self._speech_dialog_start_at = (speech_event or {}).get("dialog_start_ts") or (speech_event or {}).get("llm_start_ts")
                self._speech_first_frame_marked = False
                logger.info(
                    "Avatar speaking start text=%s llm_elapsed=%s",
                    self._speech_text,
                    (speech_event or {}).get("llm_elapsed"),
                )
            elif not now_speaking and was_speaking and self._speech_started_at is not None:
                avatar_elapsed = time.perf_counter() - self._speech_started_at
                logger.info(
                    "Avatar speaking done in %.3fs text=%s",
                    avatar_elapsed,
                    self._speech_text,
                )
                if self._speech_dialog_id:
                    self.update_dialog_meta(self._speech_dialog_id, {"avatar_elapsed": avatar_elapsed})
                self._speech_started_at = None
                self._speech_text = ""
                self._speech_dialog_id = ""
                self._speech_dialog_start_at = None
                self._speech_first_frame_marked = False

            if enable_transition:
                current_speaking = now_speaking
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} -> {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:
                self.speaking = False
                target_frame = self.get_silence_frame(idx, audio_frames)
                if target_frame is None:
                    target_frame = self.frame_list_cycle[_safe_cycle_index(len(self.frame_list_cycle), idx)]

                if enable_transition:
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1 - alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame, idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if (
                    not self._speech_first_frame_marked
                    and self._speech_dialog_id
                    and self._speech_dialog_start_at is not None
                ):
                    avatar_first_frame_elapsed = time.perf_counter() - float(self._speech_dialog_start_at)
                    logger.info(
                        "Avatar first frame in %.3fs text=%s",
                        avatar_first_frame_elapsed,
                        self._speech_text,
                    )
                    self.update_dialog_meta(
                        self._speech_dialog_id,
                        {"avatar_first_frame_elapsed": avatar_first_frame_elapsed},
                    )
                    self._speech_first_frame_marked = True
                if enable_transition:
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1 - alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame, None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame, type, eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame, eventpoint)), loop)
                self.record_audio_data(frame)

            output_time += time.perf_counter() - frame_start
            output_count += 1
            if output_count >= 25:
                res_qsize = self.res_frame_queue.qsize()
                video_qsize = video_track._queue.qsize() if video_track is not None else -1
                audio_qsize = audio_track._queue.qsize() if audio_track is not None else -1
                logger.info(
                    "lip-sync process avg fps=%.4f res_q=%s video_q=%s audio_q=%s speaking=%s",
                    output_count / output_time,
                    res_qsize,
                    video_qsize,
                    audio_qsize,
                    self.speaking,
                )
                output_count = 0
                output_time = 0.0
        logger.info("lipreal process_frames thread stop")
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.init_customindex()
        self._tts_quit_event = quit_event
        self.tts.render(quit_event)
        
        infer_quit_event = Event()
        infer_thread = Thread(target=inference, args=(infer_quit_event,self.batch_size,self.face_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,))  #mp.Process
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,loop,audio_track,video_track))
        process_thread.start()

        #self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            if video_track and video_track._queue.qsize()>=5:
                logger.debug('sleep qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        logger.info('lipreal thread stop')

        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()
            
