import os

import pandas as pd
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager


def _list_to_slices(l):
    slices = []
    for i in l:
        if not slices:
            slices.append((0, 0))
        else:
            slices[-1] = (slices[-1][0], i)
            slices.append((i, 0))

    slices = slices[:-1]
    out = []
    for s in slices:  # remove double trigger, e.g. fading
        if s[0] + 1 == s[1]:
            out.append(s[0])
        else:
            out.append(s)
    return out


def _analyse_metrics(data, last_frame_in_video, min_threshold=20):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df["content_val"].sort_values()  # sorted differences of information between frames
    df = pd.concat([df, df.diff().rename("diff")], axis=1)  # diff of the above
    df1 = df.index[(df["diff"] > 2) & (df["content_val"] > min_threshold)]
    if len(df1) == 0:
        return [0, last_frame_in_video]
    idx = df1[0]  # index from which starts the list of scene borders

    l = sorted(list(df.loc[idx:].index))
    l.insert(0, 0)
    l.append(last_frame_in_video)
    return l


def find_scenes(imgs_path, return_slices, threshold=None):
    """
    Function to find scene borders.

    @param imgs_path: path to images
    @param return_slices: whether to return slices or list of borders.
        Slices can have single element if scene consists of one img. List contains indexes of first imgs of scenes.
    @param threshold: threshold for scenedetector. If None uses automatic estimation.
    @return: slices or list of borders
    """
    n_zeros = len(os.listdir(imgs_path)[0]) - 4
    video_manager = VideoManager([f'{imgs_path}/%0{n_zeros}d.png'])

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    if threshold is None:
        scene_manager.add_detector(ContentDetector())
    else:
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=1))

    try:
        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_borders = []
        if threshold is None:
            scene_borders = _analyse_metrics(stats_manager._frame_metrics, video_manager._frame_length)
        else:
            scene_list = scene_manager.get_scene_list()
            for scene in scene_list:
                scene_borders.append(scene[0].get_frames())
            scene_borders.append(video_manager._frame_length)

        if return_slices:
            scene_borders = _list_to_slices(scene_borders)
        else:
            scene_borders = scene_borders[1:-1]

    finally:
        video_manager.release()
    return scene_borders


if __name__ == '__main__':
    borders = find_scenes('path/to/imgs', return_slices=True)
    print(borders)
