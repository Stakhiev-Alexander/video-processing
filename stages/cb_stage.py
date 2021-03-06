import shutil
from glob import glob
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

import utils.logger as logger
from stages.sequence_stage_base import SequenceStage
from utils.cb_utils import cb_seq
from utils.scene_detection import find_scenes

logger = logger.get_logger(__name__)


class CBStage(SequenceStage):
    """
    Color balance stage class
    """

    def __init__(self, percent=0.01, output_path='./output/cb_stage_output/'):
        self._percent = percent
        self._output_path = str(Path(output_path).absolute())

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute())
        Path(self._input_path).mkdir(exist_ok=True)
        shutil.rmtree(self._output_path, ignore_errors=True)
        Path(self._output_path).mkdir(exist_ok=True)

        base_path = Path(__file__).parent.absolute()

        print(self._input_path)
        imgs_paths = glob(self._input_path + '/*.png')
        imgs_paths.sort()

        slices = find_scenes(input_path, return_slices=True)

        for s in tqdm(slices):
            if isinstance(s, tuple):
                imgs = []
                for img_path in imgs_paths[slice(*s)]:
                    imgs.append(cv.imread(img_path))
                out = cb_seq(imgs, 0.01)
                for i, img in enumerate(out):
                    cv.imwrite(self._output_path + '/' + str(s[0] + i + 1).zfill(6) + '.png', img)
            else:  # 1 frame
                shutil.copy(imgs_paths[s], self._output_path)

    @property
    def output_path(self):
        return self._output_path
