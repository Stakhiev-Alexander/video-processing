from pathlib import Path

import utils.logger as logger
from stages.sequence_stage_base import SequenceStage
from utils.video_tools import assemble_video

logger = logger.get_logger(__name__)


class AssembleStage(SequenceStage):
    def __init__(self, framerate, filename, output_path='./output/'):
        self._framerate = framerate
        self._filename = filename
        self._output_path = str(Path(output_path).absolute())

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute())

        try:
            assemble_video(imgs_path=self._input_path, framerate=self._framerate, filename=self._filename,
                           output_dir=self._output_path)
        except Exception as err:
            logger.error(err)
            raise err

    @property
    def output_path(self):
        return self._output_path
