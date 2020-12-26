import subprocess
import os
import shutil
from pathlib import Path

from sequence_stage_base import SequenceStage
import process_sequence_logger as ps_logger


logger = ps_logger.get_logger(__name__) 


class ArtefactsStage(SequenceStage):
    def __init__(self, output_path='./artefacts_stage_output/'):
        self._output_path = str(Path(output_path).absolute())

    def execute(self, input_path):
        self._input_path = str(Path(input_path).absolute())
        Path(self._input_path).mkdir(exist_ok=True)
        shutil.rmtree(self._output_path, ignore_errors=True)
        Path(self._output_path).mkdir(exist_ok=True)

        base_path = Path(__file__).parent.absolute()
        script_path = str(base_path / 'run_scripts' / 'run_dl_hifill.sh')
        parameters = f'--input_path {self._input_path} --output_path {self._output_path}'
        run_cmd = script_path + ' ' + parameters

        try:
            result = subprocess.run([run_cmd], shell=True, encoding='utf-8', check=True)  
        except subprocess.CalledProcessError as err:
            logger.error(err)
            raise err

    @property
    def output_path(self):
        return self._output_path
