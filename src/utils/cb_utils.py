import cv2
import numpy as np
import pandas as pd


# Color balances image list in place
def cb_seq(imgs, percent):
    step = 10
    chosen = imgs[::step]
    cumstops = (
        imgs[0].shape[0] * imgs[0].shape[1] * len(chosen) * percent / 200.0,
        imgs[0].shape[0] * imgs[0].shape[1] * len(chosen) * (1 - percent / 200.0)
    )

    hists = np.zeros(shape=(3, 256, 1))
    for img in chosen:
        for i, channel in enumerate(cv2.split(img)):
            hist = cv2.calcHist([channel], [0], None, [256], (0, 256))
            hists[i] += hist

    cumhists = np.cumsum(hists, axis=1)
    cuts = []
    for cumhist in cumhists:
        cuts.append(np.searchsorted(cumhist.flatten(), cumstops))

    luts = []
    for cut in cuts:
        lut = np.concatenate((
            np.zeros(cut[0]),
            np.around(np.linspace(0, 255, cut[1] - cut[0] + 1)),
            255 * np.ones(255 - cut[1])
        ))
        luts.append(lut.astype('uint8'))

    for i, img in enumerate(imgs):
        out_channels = []
        for j, channel in enumerate(cv2.split(img)):
            out_channels.append(cv2.LUT(channel, luts[j]))
        imgs[i] = cv2.merge(out_channels)
    return imgs


def list_to_slices(l):
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


def analyse_metrics(data, last_frame_in_video):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df["content_val"].sort_values()  # sorted differences of information between frames
    df = pd.concat([df, df.diff().rename("diff")], axis=1)  # diff of the above
    idx = df.index[df["diff"] > 2][0]  # index from which starts the list of scene borders
    l = sorted(list(df.loc[idx:].index))
    l.insert(0, 0)
    l.append(last_frame_in_video)
    return list_to_slices(l)
