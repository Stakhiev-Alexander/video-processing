from stages.artefacts_stage import ArtefactsStage
from stages.process_sequence import ProcessSequence
from stages.nlm_stage import NLMStage
from stages.sr_stage import SRStage


if __name__ == '__main__':
    ps = ProcessSequence(input_path='/home/quadro/videoproc/datasets/umatic_frames/')
    ps.add(NLMStage(grayscale=False))
    ps.add(SRStage())
    ps.execute()

    # assemble_video_lossless(imgs_path='/home/quadro/videoproc/video-processing/sr_stage_output/', framerate=25,filename='hockey_nlm_cb_sr')
