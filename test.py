# coding: utf-8

import json
import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import MyDataset
from model import BMN_model
from opt import MyConfig
from utils.inference_utils import proposals_select_pervideo
from utils.nms_utils import soft_nms_proposal
from utils.opt_utils import get_cur_time_stamp
from utils.eval_utils import ANETproposal
from utils.plot_utils import plot_result

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "1"            

# Basic test.
print("Pytorch's version is {}.".format(torch.__version__))
print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
print("CUDA's state is {}.".format(torch.cuda.is_available()))
print("CUDA's version is {}.".format(torch.version.cuda))
print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Weights initialization.
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


# Random seed.
def my_worker_init_fn():
    seed = torch.initial_seed()
    np_seed = seed // (2 ** 32 - 1)
    random.seed(seed)                 
    np.random.seed(np_seed)     
    # torch.manual_seed(seed)             # CPU.
    # torch.cuda.manual_seed(seed)        # One GPU.
    # torch.cuda.manual_seed_all(seed)    # Multiple GPUs.
    # torch.backends.cudnn.deterministic = True


def get_type_data(opt, mode='valid'):

    """Get data of cetrain type. Save information in 'video_dict'. """
    # total video: 19228
    # training:9649, validation:4728, testing:4851.

    video_dict = {}
    
    videos_info = pd.read_csv(opt.video_info_path) 
    with open(opt.video_anno_path) as f:
        videos_anno = json.load(f)
    
    for i in range(len(videos_info)): 
        video_subset = videos_info.subset.values[i]
        if mode in video_subset:
            video_name = videos_info.video.values[i]
            video_anno = videos_anno[video_name]
            video_dict[video_name] = video_anno

    return video_dict


if __name__ == "__main__":

    opt = MyConfig()
    opt.parse()

    start_time = str(get_cur_time_stamp())

    """Load model and data, save scores of all possible proposals without selecting. """
    model = BMN_model(opt)
    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataset = MyDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, 
                                                   num_workers=opt.num_workers, pin_memory=True)

    with torch.no_grad():
        for index, video_feature in test_dataloader:
            video_name = test_dataloader.dataset.video_list[index]
            video_feature = video_feature.cuda()
            bm_confidence_map, start, end = model(video_feature)

            bm_confidence_map_reg = bm_confidence_map[0][0].detach().cpu().numpy()
            bm_confidence_map_cls = bm_confidence_map[0][1].detach().cpu().numpy()
            start = start[0].detach().cpu().numpy()
            end = end[0].detach().cpu().numpy()

            # Iterate over all time conbinations(start & end).
            proposals = []
            for i in range(opt.temporal_scale):
                for j in range(opt.temporal_scale):
                    start_index = i
                    end_index = j + 1
                    if start_index < end_index < opt.temporal_scale:
                        xmin = start_index / opt.temporal_scale
                        xmax = end_index / opt.temporal_scale
                        xmin_score = start[start_index]
                        xmax_score = end[end_index]

                        cls_score = bm_confidence_map_cls[i][j]
                        reg_score = bm_confidence_map_reg[i][j]
                        score = xmin_score * xmax_score * cls_score * reg_score

                        proposals.append([xmin, xmax, xmin_score, xmax_score, cls_score, reg_score, score])
            proposals = np.stack(proposals)

            columns = ["xmin", "xmax", "xmin_score", "xmax_score", "cls_score", "reg_score", "score"]
            df = pd.DataFrame(df, columns=columns)
            df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)

    """Get all videoes' selected proposals in multi-processing.  """
    video_dict = get_type_data(opt, type='valid')
    video_list = list(video_dict.keys())
    num_video = len(video_list)
    num_video_per_thread = num_video // opt.post_process_thread

    # Multi-processing.
    results = mp.Manager().dict()
    processes = []

    for index_thread in range(opt.post_process_thread - 1):
        temp_video_list = video_list[index_thread * num_video_per_thread: (index_thread + 1) * num_video_per_thread]
        p = mp.Process(target=proposals_select_pervideo, args=(opt, temp_video_list, video_dict))
        p.start()
        processes.append(p)

    # final batch.
    temp_video_list = video_list[(opt.post_process_thread - 1) * num_video_per_thread:]
    p = mp.Process(target=proposals_select_pervideo, args=(opt, temp_video_list, video_dict))
    p.start()
    processes.append(p)
    # Make sure that all process is finished.
    for p in processes:
        p.join()

    results = dict(results)
    results_ = {"version": "1.3", "results":results, "external_data": {}}
    with open(opt.result_json_path, 'w') as j:
        json.dump(results_, j)

    """Run evaluation and save figure. """
    anet_proposal = ANETproposal(ground_truth_filename=opt.evaluation_json_path, 
                                 proposal_filename=opt.result_json_path, 
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset='validation',
                                 verbose=True,
                                 check_status=False)
    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    num_proposals = anet_proposal.proposals_per_video

    plot_result(opt, num_proposals, recall, average_recall)
    print( "AR@1 is {}".format(np.mean(recall[:,0])))
    print( "AR@5 is {}".format(np.mean(recall[:,4])))
    print( "AR@10 is {}".format(np.mean(recall[:,9])))
    print( "AR@100 is {}".format(np.mean(recall[:,-1])))