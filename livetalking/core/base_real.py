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

import subprocess
import os
import time
import uuid
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction
from threading import Lock

from ..providers.sherpa_tts import SherpaOnnxVitsTTS
from ..providers.tts_engines import EdgeTTS
from ..utils.app_logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        self.tts = self.build_tts()
        self._tts_quit_event = None

        self.speaking = False
        self._speech_started_at = None
        self._speech_text = ""
        self._speech_dialog_id = ""
        self._speech_dialog_start_at = None
        self._speech_first_frame_marked = False
        self.dialog_lock = Lock()
        self.dialog_history: list[dict] = []

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.silence_gate_enabled = bool(getattr(opt, "SILENCE_GATE_ENABLED", False))

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

    def build_tts(self):
        if self.opt.tts in {"vits_zh", "sherpa_onnx_vits", "sherpa-onnx-vits", "vits_melo_zh_en", "sherpa_onnx_vits_zh_en", "vits_melo_tts_zh_en"}:
            return SherpaOnnxVitsTTS(self.opt, self)
        if self.opt.tts == "edgetts":
            return EdgeTTS(self.opt, self)
        raise ValueError(f"Unsupported TTS provider: {self.opt.tts}")

    def reload_tts(self):
        self.tts = self.build_tts()
        if self._tts_quit_event is not None and hasattr(self.tts, "render"):
            self.tts.render(self._tts_quit_event)
        return self.tts

    def put_msg_txt(self,msg,datainfo:dict={}):
        self.tts.put_msg_txt(msg,datainfo)

    def append_dialog(self, role: str, text: str, source: str = "", meta: dict | None = None, dialog_id: str = "") -> str:
        text = (text or "").strip()
        if not text:
            return ""

        entry = {
            "id": dialog_id or str(uuid.uuid4()),
            "role": role,
            "text": text,
            "source": source,
            "ts": time.time(),
            "meta": dict(meta or {}),
        }
        with self.dialog_lock:
            self.dialog_history.append(entry)
            if len(self.dialog_history) > 100:
                self.dialog_history = self.dialog_history[-100:]
        return entry["id"]

    def update_dialog_meta(self, dialog_id: str, meta: dict | None = None) -> None:
        if not dialog_id:
            return
        updates = dict(meta or {})
        if not updates:
            return
        with self.dialog_lock:
            for item in reversed(self.dialog_history):
                if item.get("id") == dialog_id:
                    current = dict(item.get("meta") or {})
                    current.update(updates)
                    item["meta"] = current
                    return

    def get_dialog_history(self, limit: int = 50) -> list[dict]:
        with self.dialog_lock:
            if limit <= 0:
                return list(self.dialog_history)
            return list(self.dialog_history[-limit:])

    def clear_dialog_history(self) -> None:
        with self.dialog_lock:
            self.dialog_history.clear()
    
    def put_audio_frame(self,audio_chunk,datainfo:dict={}): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,datainfo)

    def put_audio_file(self,filebyte,datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk],datainfo)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in getattr(self.opt, "customopt", []):
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream

    def get_silence_frame(self, idx, audio_frames):
        return None

    def set_silence_gate(self, enabled: bool) -> None:
        self.silence_gate_enabled = bool(enabled)

    def is_silence_gate_enabled(self) -> bool:
        return bool(self.silence_gate_enabled)
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        output_count = 0
        output_time = 0.0
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        while not quit_event.is_set():
            try:
                frame_start = time.perf_counter()
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
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
                # 检测状态变化
                current_speaking = now_speaking
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking
            
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
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
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
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
        logger.info('basereal process_frames thread stop') 
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1
