import mimetypes
import os

from sequence_stage_base import SequenceStage
from denoise_stage import DenoiseStage
from nlm_stage import NLMStage
import process_sequence_logger as ps_logger
from utils.video_tools import cut_frames


logger = ps_logger.get_logger(__name__) 


class UnknownInputType(Exception):
    pass


class ProcessSequence(object):
    @staticmethod
    def get_input_frames_dir(input_path):
        if os.path.isdir(input_path):
            logger.info(f'Input dir: {input_path}')
            return input_path

        mimestart = mimetypes.guess_type(input_path)[0] 
        if mimestart is None:
            raise UnknownInputType  

        file_type = mimestart.split('/')[0]
        if file_type == 'video':
            input_dir = cut_frames(input_path)
            logger.info(f'Input dir: {input_dir}')
            return input_dir
        else:
            raise UnknownInputType


    def __init__(self, input_path, stages=None):
        """
        input_path (str): path to image sequence or video
        stages (list<SequenceStage>): list of processing stages 
        """
        self._input_path = ProcessSequence.get_input_frames_dir(input_path)
        
        self._stages = []
        if stages:
          if not isinstance(stages, (list, tuple)):
            stages = [stages]
          for stage in stages:
            self.add(stage)


    def add(self, stage):
        assert isinstance(stage, SequenceStage)
        self._stages.append(stage)


    def execute(self):
        input_path = self._input_path

        for stage in self._stages:
            stage.execute(input_path)
            input_path = stage.output_path


if __name__ == '__main__':
    ps = ProcessSequence(input_path='/home/quadro/videoproc/datasets/hockey_1_7GB.mov')
    ps.add(NLMStage())
    ps.execute()
