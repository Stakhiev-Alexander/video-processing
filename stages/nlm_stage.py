import shutil
from glob import glob
from pathlib import Path

from tqdm import tqdm
import cv2 as cv

import utils.logger as logger
from stages.sequence_stage_base import SequenceStage

logger = logger.get_logger(__name__)
IMG_EXTENTION = 'png'


class NLMStage(SequenceStage):
    def __init__(self, output_path='./output/nlm_stage_output/',
                 h=7,
                 templateWindowSize=9,
                 searchWindowSize=11,
                 grayscale=True):
        self._output_path = str(Path(output_path).absolute())
        self._h = h
        self._templateWindowSize = templateWindowSize
        self._searchWindowSize = searchWindowSize
        if grayscale:
            self._img_mode = cv.IMREAD_GRAYSCALE

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute()) + '/'
        Path(self._input_path).mkdir(exist_ok=True)
        shutil.rmtree(self._output_path, ignore_errors=True)
        Path(self._output_path).mkdir(exist_ok=True)

        logger.info(f'Starting NLMStage\nInput path: {self._input_path}\nOutput path: {self._output_path}')

        imgs_paths = sorted(glob(self._input_path + '*.' + IMG_EXTENTION))
        imgs = [cv.imread(path, self._img_mode) for path in imgs_paths[:3]]

        firstframe = cv.fastNlMeansDenoising(imgs[0], dst=None,
                                                h=self._h,
                                                templateWindowSize=self._templateWindowSize,
                                                searchWindowSize=self._searchWindowSize)

        cv.imwrite(self._output_path + '/' + str(0).zfill(6) + '.' + IMG_EXTENTION, firstframe)

        for i, img_path in enumerate(tqdm(imgs_paths[2:])):
            if i == 0:
                imgs.append(cv.imread(img_path, self._img_mode))
            else:
                imgs = imgs[1:]
                imgs.append(cv.imread(img_path, self._img_mode))

            result = cv.fastNlMeansDenoisingMulti(imgs, imgToDenoiseIndex=1,
                                                        temporalWindowSize=3,
                                                        dst=None,
                                                        h=self._h,
                                                        templateWindowSize=self._templateWindowSize,
                                                        searchWindowSize=self._searchWindowSize)

            cv.imwrite(self._output_path + '/' + str(i+1).zfill(6) + '.' + IMG_EXTENTION, result)


        lastframe = cv.fastNlMeansDenoising(cv.imread(imgs_paths[-1], self._img_mode), dst=None,
                                        h=self._h,
                                        templateWindowSize=self._templateWindowSize,
                                        searchWindowSize=self._searchWindowSize)

        cv.imwrite(self._output_path + '/' + str(len(imgs_paths)-1).zfill(6) + '.' + IMG_EXTENTION, lastframe)

        logger.info('Finished NLMStage')


    @property
    def output_path(self):
        return self._output_path
