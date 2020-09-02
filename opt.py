from utils.opt_utils import ConfigBase

class MyConfig(ConfigBase):

    def __init__(self):

        super(MyConfig, self).__init__()
            
        # path.
        self.train_data_path = 
        self.test_data_path = 
        self.valid_data_path = 
        self.save_path = 
        self.checkpoint_path = 
        self.log_path = 

        # Hyper-parameters.
        self.epochs = 
        self.batch_size = 
        self.learning_rate = 
        self.weight_decay = 
        self.num_workers = 
        self.step_size =
        self.step_gamma = 

        # Parameters.
        self.soft_nms_high_thres = 
        self.soft_nms_low_thres = 
