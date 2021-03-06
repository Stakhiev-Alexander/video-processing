import shutil
import subprocess
from pathlib import Path

import utils.logger as logger
from stages.sequence_stage_base import SequenceStage

logger = logger.get_logger(__name__)


class DenoiseStage(SequenceStage):
    def __init__(self, sigma, output_path='./output/denoise_stage_output/'):
        self._sigma = sigma
        self._output_path = str(Path(output_path).absolute())

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute())
        Path(self._input_path).mkdir(exist_ok=True)
        shutil.rmtree(self._output_path, ignore_errors=True)
        Path(self._output_path).mkdir(exist_ok=True)

        script_path = '../run_scripts/run_fastdvdnet.sh'
        parameters = f'--input_path {self._input_path} --sigma {self._sigma} --output_path {self._output_path}'
        run_cmd = script_path + ' ' + parameters

        try:
            result = subprocess.run([run_cmd], shell=True, encoding='utf-8', check=True)  
        except subprocess.CalledProcessError as err:
            logger.error(err)
            raise err

    @property
    def output_path(self):
        return self._output_path
