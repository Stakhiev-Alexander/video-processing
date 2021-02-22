from stages.artefacts_stage import ArtefactsStage
from stages.process_sequence import ProcessSequence

if __name__ == '__main__':
    ps = ProcessSequence(input_path='/home/quadro/videoproc/datasets/small2_5/')
    ps.add(ArtefactsStage())
    ps.execute()

    # assemble_video_lossless(imgs_path='/home/quadro/videoproc/video-processing/sr_stage_output/', framerate=25,filename='hockey_nlm_cb_sr')
