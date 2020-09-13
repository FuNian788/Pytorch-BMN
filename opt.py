from utils.opt_utils import ConfigBase

class MyConfig(ConfigBase):

    def __init__(self, save_txt_flag=True, save_json_flag=False):

        super(MyConfig, self).__init__()
        
        self.save_txt_flag = save_txt_flag
        self.save_json_flag = save_json_flag

        # mode.
        self.mode = 'train'

        # path.
        self.video_info_path = './data/activitynet_annotations/video_info_new.csv'
        self.video_anno_path = './data/activitynet_annotations/anet_anno_action.json'
        self.feature_path = './data/activitynet_feature_cuhk/'
        self.evaluation_json_path = '/data/eval/activity_net_1_3_new.json'
        self.result_json_path = './output/result_proposal.json'

        self.save_path = './save/'
        self.log_path = './save/'
        self.checkpoint_path = './checkpoint/20200910-1022/'
        self.save_fig_path = './output/evaluation_result.jpg'

        # Hyper-parameters.
        self.epochs = 9
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        self.step_size = 7
        self.step_gamma = 0.1
        self.post_process_thread = 8

        # Parameters.
        self.temporal_scale = 100
        self.num_sample = 32
        self.num_sample_perbin = 3
        self.prop_boundary_ratio = 0.5
        self.feat_dim = 400

        self.soft_nms_high_thres = 0.9
        self.soft_nms_low_thres = 0.5
        self.soft_nms_alpha = 0.4

        self.num_workers = 8
