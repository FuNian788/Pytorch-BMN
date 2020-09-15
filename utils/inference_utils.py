# coding: utf-8

import pandas as pd

from utils.nms_utils import soft_nms_proposal

def proposals_select_pervideo(opt, video_list, video_dict, results):
    """
    Select 100 proposals in each video's proposals(soft-NMS), then save.

    Arguements:
        opt: (config): parameters.
        video_list: (list): video names.
        video_dict: (dict): video information.
        results: (multiprocessing.Manager().dict()): a dict which contains selected periods of each video.

    Return Arguements:
        video_name[2:]: (str): name of video, eg: 'c8enCfzqw'.
        proposals: (list): selected proposals of certain video.
    """
    for video_name in video_list:
        df = pd.read_csv("./output/BMN_results/" + video_name + ".csv")

        if len(df) > 1:
            df = soft_nms_proposal(df, opt.soft_nms_alpha, opt.soft_nms_low_thres, opt.soft_nms_high_thres)

        df = df.sort_values(by="score", ascending=False)

        video_info = video_dict[video_name]

        real_video_duration = float(video_info["duration_frame"] // 16 * 16) / video_info["duration_frame"] * video_info["duration_second"]

        proposals = []
        for i in range(min(100, len(df))):
            proposal = {}
            proposal["score"] = df.score.values[i]
            proposal["segment"] = [max(0, df.xmin.values[i]) * real_video_duration, min(1, df.xmax.values[i]) * real_video_duration]
            proposals.append(proposal)

        results[video_name[2:]] = proposals
