from stages.artefacts_stage import ArtefactsStage
from stages.assemble_stage import AssembleStage
from stages.process_sequence import ProcessSequence
from stages.nlm_stage import NLMStage
from stages.sr_stage import SRStage


if __name__ == '__main__':
    input_path = '~/videoproc/datasets/umatic_frames/'
    ps = ProcessSequence(input_path=input_path)
    ps.add(NLMStage(grayscale=False))
    ps.add(SRStage())
    ps.add(AssembleStage(framerate=25, filename='umatic_out'))
    ps.execute()
