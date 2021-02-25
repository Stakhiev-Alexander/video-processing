import glob
import os
import subprocess

import cv2
import ffmpeg
import image_slicer
from PIL import Image


def get_n_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n_frames


def get_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (width, height)


def get_fps(video_path): 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def cut_frames(video_path, output_dir='./cut_frames/', n_frames_cut=-1):
    '''
        Cuts all frames from video by default. 
        If n_frames_cut is given then specified number of frames
         are cut with equal interval between these frames. 
    '''

    os.makedirs(output_dir, exist_ok=True) # make output dirs recursively
    out_path_pattern = output_dir + '%05d' + '.png'
    print(f'Images path pattern: {out_path_pattern}')

    n_frames = get_n_frames(video_path)

    if 0 < n_frames_cut < n_frames:
        thumbnail = int(n_frames/n_frames_cut) - 1

        # ffmpeg -i <video_path> -vf thumbnail=<n_frames/n_frames_cut - 1>, setpts=N/TB -r 1 -vframes <n_frames_cut> <output_dir>%05d.png
        try:
            (
                ffmpeg
                .input(video_path)
                .filter('thumbnail', thumbnail)
                .filter('setpts', 'N/TB')
                .output(out_path_pattern, r=1, vframes=n_frames_cut)
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

    return output_dir    
    
  
def assemble_video(imgs_path, framerate, filename='output', output_dir='./'):
    in_path_pattern = imgs_path + '*.png'
    out_path = output_dir + filename + '.mkv'

    # ffmpeg -framerate 50 -i ./sr_stage_output/%15d.png -c:v libx264 -pix_fmt yuv420p <output_dir>/<filename>.mkv
    try:
        (
            ffmpeg
            .input(in_path_pattern, pattern_type='glob', framerate=framerate)
            .output(out_path, vcodec='libx264', pix_fmt='yuv420p')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))


def split_imgs(imgs_path, output_dir, split_factor=4):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in glob.glob(imgs_path +  '*.png'):
        tiles = image_slicer.slice(img_path, split_factor, save=False)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image_slicer.save_tiles(tiles, directory=output_dir, prefix=img_name, format="png")


def join_imgs(imgs_path, output_dir, split_factor):
    os.makedirs(output_dir, exist_ok=True)
    base_names = set()
    for img_path in glob.glob(imgs_path + '*.png'):
        img_base_name = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
        base_names.add(img_base_name)


    for base_name in base_names:
        tiles = []
        for i, img_path in enumerate(glob.glob(imgs_path +  f'{base_name}_*.png')):
            pos = image_slicer.get_image_column_row(os.path.basename(img_path))
            im = Image.open(img_path)
            position_xy = [0, 0]
            count = 0
            for a, b in zip(pos, im.size):
                position_xy[count] = a * b
                count = count + 1
            tiles.append(
                image_slicer.Tile(
                    image=im,
                    position=pos,
                    number=i + 1,
                    coords=position_xy,
                    filename=img_path,
                )
            )
        
        joined_img = image_slicer.join(tiles)
        joined_img.save(output_dir + base_name + '.png')


    def copy_audio(input_video, output_video):
        """
        Copy audio stream from one video to another
        
        :param input_video: video to take audio stream from
        :param output_video: video to add audio stream to
        """
        subprocess.run(f"ffmpeg -i {input_video} -vn -acodec copy tmp.mp3")
        subprocess.run(f"ffmpeg -y -i {output_video} -i audio.mp3 -map 0 -map 1:a -c:v copy -shortest {output_video}")
        os.remove("tmp.mp3")


if __name__ == '__main__':
    video_path = './example.mp4'

    print(get_n_frames(video_path))
    print(get_resolution(video_path))
    print(get_fps(video_path))
