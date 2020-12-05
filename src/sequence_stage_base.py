from abc import ABC, abstractmethod


class SequenceStage(ABC):
    @abstractmethod
    def execute():
        pass

    @property
    @abstractmethod
    def output_path(self):
        raise NotImplementedError        
