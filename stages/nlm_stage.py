import shutil
from glob import glob
from pathlib import Path

import cv2
from tqdm import tqdm

import utils.logger as logger
from stages.sequence_stage_base import SequenceStage

logger = logger.get_logger(__name__)


class NLMStage(SequenceStage):
    def __init__(self, grayscale,
                 output_path='./output/nlm_stage_output/',
                 h=7,
                 templateWindowSize=9,
                 searchWindowSize=11):
        self._output_path = str(Path(output_path).absolute())
        self._h = h
        self._templateWindowSize = templateWindowSize
        self._searchWindowSize = searchWindowSize
        if grayscale:
            self._img_mode = cv2.IMREAD_GRAYSCALE
        else:
            self._img_mode = cv2.IMREAD_COLOR

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute()) + '/'
        shutil.rmtree(self._output_path, ignore_errors=True)
        Path(self._output_path).mkdir(exist_ok=True, parents=True)

        logger.info(f'Starting NLMStage\nInput path: {self._input_path}\nOutput path: {self._output_path}')

        imgs_paths = sorted(glob(self._input_path + '*.png'))
        imgs = [cv2.imread(path, self._img_mode) for path in imgs_paths[:2]]

        first_frame = cv2.fastNlMeansDenoising(imgs[0], dst=None,
                                               h=self._h,
                                               templateWindowSize=self._templateWindowSize,
                                               searchWindowSize=self._searchWindowSize)
        cv2.imwrite(self._output_path + '/' + str(0).zfill(6) + '.png', first_frame)

        for i, img_path in enumerate(tqdm(imgs_paths[2:])):
            if i != 0:
                imgs = imgs[1:]
            imgs.append(cv2.imread(img_path, self._img_mode))

            result = cv2.fastNlMeansDenoisingMulti(imgs, imgToDenoiseIndex=1,
                                                   temporalWindowSize=3,
                                                   dst=None,
                                                   h=self._h,
                                                   templateWindowSize=self._templateWindowSize,
                                                   searchWindowSize=self._searchWindowSize)

            cv2.imwrite(self._output_path + '/' + str(i + 1).zfill(6) + '.png', result)

        last_frame = cv2.fastNlMeansDenoising(cv2.imread(imgs_paths[-1], self._img_mode), dst=None,
                                              h=self._h,
                                              templateWindowSize=self._templateWindowSize,
                                              searchWindowSize=self._searchWindowSize)

        cv2.imwrite(self._output_path + '/' + str(len(imgs_paths) - 1).zfill(6) + '.png', last_frame)

        logger.info('Finished NLMStage')

    @property
    def output_path(self):
        return self._output_path


if __name__ == '__main__':
    original_img_path = 'path/to/imgs'
    nlm_stage = NLMStage(grayscale=False, output_path='path/to/output', h=3)
    nlm_stage.execute(original_img_path)