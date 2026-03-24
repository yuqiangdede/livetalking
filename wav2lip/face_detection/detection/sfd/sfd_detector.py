import os
import cv2
import torch
from pathlib import Path
from urllib.request import urlretrieve

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


def _looks_like_git_lfs_pointer(path_to_detector: str) -> bool:
    try:
        with open(path_to_detector, 'rb') as file:
            header = file.read(64)
        return header.startswith(b'version https://git-lfs.github.com/spec/v1')
    except OSError:
        return False


def _ensure_local_weights(path_to_detector: str) -> str:
    if os.path.isfile(path_to_detector) and not _looks_like_git_lfs_pointer(path_to_detector):
        return path_to_detector

    target_path = Path(path_to_detector)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + '.download')
    urlretrieve(models_urls['s3fd'], temp_path)
    temp_path.replace(target_path)
    return str(target_path)


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth'), verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        path_to_detector = _ensure_local_weights(path_to_detector)
        try:
            model_weights = torch.load(path_to_detector, map_location=device, weights_only=False)
        except TypeError:
            model_weights = torch.load(path_to_detector, map_location=device)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
