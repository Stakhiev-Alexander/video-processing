import os
import glob

import skvideo.io
import ffmpeg
import image_slicer
from PIL import Image


IMG_EXTENTION = 'png'


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
    out_path_pattern = output_dir + '%05d' + '.' + IMG_EXTENTION
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
    in_path_pattern = imgs_path + '*.' + IMG_EXTENTION
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


def split_imgs(imgs_path, output_dir, split_factor=9):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in glob.glob(imgs_path +  '*.' + IMG_EXTENTION):
        tiles = image_slicer.slice(img_path, split_factor, save=False)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image_slicer.save_tiles(tiles, directory=output_dir, prefix=img_name, format=IMG_EXTENTION)


def join_imgs(imgs_path, output_dir, split_factor):
    os.makedirs(output_dir, exist_ok=True)
    base_names = set()
    for img_path in glob.glob(imgs_path +  '*.' + IMG_EXTENTION):
        img_base_name = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
        base_names.add(img_base_name)

    for base_name in base_names:
        tiles = []
        i = 0
        for img_path in glob.glob(imgs_path +  f'{base_name}_*.' + IMG_EXTENTION):
            pos = image_slicer.get_image_column_row(os.path.basename(img_path))
            im = Image.open(img_path)
            position_xy = [0, 0]
            count = 0
            for a, b in zip(pos, im.size):
                position_xy[count] = a * b
                count = count + 1
            print(position_xy)
            tiles.append(
                image_slicer.Tile(
                    image=im,
                    position=pos,
                    number=i + 1,
                    coords=position_xy,
                    filename=img_path,
                )
            )
            i = i + 1
        return
        joined_img = image_slicer.join(tiles)
        joined_img.save(output_dir + base_name + '.' + IMG_EXTENTION)


def main():
    video_path = '/run/media/alex/Data/video_enhancment/10sec.mp4'

    get_nb_frames(video_path) 
    print(get_resolution(video_path))
    print(get_avg_fps(video_path))
    cut_frames(video_path, nb_frames_cut=10)
    assemble_video_lossless('./cut_frames/', framerate=25)
    split_imgs('./cut_frames/', './splited_frames/')
    join_imgs('./splited_frames/', './joined_frames/', 2)


if __name__ == '__main__':
    main()
