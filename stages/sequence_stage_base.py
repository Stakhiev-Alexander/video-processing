from abc import ABC, abstractmethod


class SequenceStage(ABC):
    @abstractmethod
    def execute(self, input_path):
        pass

    @property
    @abstractmethod
    def output_path(self):
        raise NotImplementedError        
