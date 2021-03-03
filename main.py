from stages.artefacts_stage import ArtefactsStage
from stages.assemble_stage import AssembleStage
from stages.process_sequence import ProcessSequence
from stages.nlm_stage import NLMStage
from stages.sr_stage import SRStage


if __name__ == '__main__':
    input_path = '/home/quadro/videoproc/datasets/test_Neiro_frames/'
    input_path = '/home/quadro/videoproc/video-processing/output/nlm_stage_output/'
    ps = ProcessSequence(input_path=input_path)
#    ps.add(NLMStage(grayscale=False))
    ps.add(ArtefactsStage())
    ps.add(AssembleStage(framerate=25, filename='test_Neiro_nlm_rife'))
    ps.execute()
