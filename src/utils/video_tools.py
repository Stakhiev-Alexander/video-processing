import os

import skvideo.io
import ffmpeg


IMG_EXTENTION = '.png'


def get_nb_frames(video_path):
    videometadata = skvideo.io.ffprobe(video_path)
    nb_frames = int(videometadata['video']['@nb_frames'])
    return nb_frames


def get_metadata(video_path):
    return skvideo.io.ffprobe(video_path)


def get_resolution(video_path):
    videometadata = skvideo.io.ffprobe(video_path)
    width = int(videometadata['video']['@width'])
    height = int(videometadata['video']['@height'])
    return (width, height)


def get_avg_fps(video_path): 
    videometadata = skvideo.io.ffprobe(video_path)
    avg_frame_rate = videometadata['video']['@avg_frame_rate']
    frames_secs = avg_frame_rate.split('/')
    avg_fps = int(frames_secs[0])/int(frames_secs[1])
    return avg_fps


def cut_frames(video_path, output_dir='./cut_frames/', nb_frames_cut=-1):
    '''
        Cuts all frames from video by default. 
        If nb_frames_cut is given then specified number of frames
         are cut with equal interval between these frames. 
    '''

    os.makedirs(output_dir, exist_ok=True) # make output dirs recursively
    out_path_pattern = output_dir + '%05d' + IMG_EXTENTION
    print(f'Images path pattern: {out_path_pattern}')

    nb_frames = get_nb_frames(video_path)

    if 0 < nb_frames_cut < nb_frames:
        thumbnail = int(nb_frames/nb_frames_cut) - 1

        # ffmpeg -i <video_path> -vf thumbnail=<nb_frames/nb_frames_cut - 1>, setpts=N/TB -r 1 -vframes <nb_frames_cut> <output_dir>%05d.png
        try:
            (
                ffmpeg
                .input(video_path)
                .filter('thumbnail', thumbnail)
                .filter('setpts', 'N/TB')
                .output(out_path_pattern, r=1, vframes=nb_frames_cut)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
    else:
        # ffmpeg -i <video_path> <output_dir>%05d.png
        try:
            (
                ffmpeg
                .input(video_path)
                .output(out_path_pattern)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
    
  
def assemble_video_lossless(imgs_path, framerate, filename='output', output_dir='./'):
    in_path_pattern = imgs_path + '*' + IMG_EXTENTION
    out_path = output_dir + filename + '.mkv'

    # ffmpeg -framerate 10 -i <imgs_path>/%05d.png -c:v copy <output_dir>/<filename>.mkv
    try:
        (
            ffmpeg
            .input(in_path_pattern, pattern_type='glob', framerate=framerate)
            .output(out_path, vcodec='copy')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))


def main():
    video_path = '/run/media/alex/Data/video_enhancment/10sec.mp4'

    print(get_nb_frames(video_path))
    print(get_resolution(video_path))
    print(get_avg_fps(video_path))
    cut_frames(video_path, nb_frames_cut=10)
    assemble_video_lossless('./cut_frames/', framerate=25)


if __name__ == '__main__':
    main()
