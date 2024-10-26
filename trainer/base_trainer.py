import torch
import os
from abc import abstractmethod

class BaseTrainer:
    def __init__(self, args, resumed_epoch=0):
        self.resumed_epoch = resumed_epoch
        self.do_pretrain = args.do_pretrain
        self.do_train = args.do_train
        self.do_eval = args.do_eval
        self.train_csv = args.train_csv
        self.val_csv = args.val_csv
        self.data_path = args.data_path
        self.features_path = args.features_path
        self.num_thread_reader = args.num_thread_reader
        self.lr = args.lr
        self.loss = args.loss
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.lr_decay = args.lr_decay
        self.n_display = args.n_display
        self.video_dim = args.video_dim
        self.seed = args.seed
        self.max_words = args.max_words
        self.max_frames = args.max_frames
        self.feature_framerate = args.feature_framerate
        self.margin = args.margin
        self.hard_negative_rate = args.hard_negative_rate
        self.negative_weighting = args.negative_weighting
        self.n_pair = args.n_pair
        self.regularize = args.regularize
        self.output_dir = args.output_dir
        self.cross_model = args.cross_model
        self.init_model = args.init_model
        self.resume_model = args.resume_model
        self.do_lower_case = args.do_lower_case
        self.warmup_proportion = args.warmup_proportion
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.alpha = args.alpha
        self.dynamic_alpha = args.dynamic_alpha
        self.n_gpu = args.n_gpu
        self.cache_dir = args.cache_dir
        self.fp16 = args.fp16
        self.fp16_opt_level = args.fp16_opt_level
        self.task_type = args.task_type
        self.datatype = args.datatype
        self.sim_lambda = args.sim_lambda
        self.post_process = args.post_process
        self.post_cluster_centroids = args.post_cluster_centroids
        self.world_size = args.world_size
        self.local_rank = args.local_rank
        self.rank = args.rank
        self.coef_lr = args.coef_lr
        self.use_mil = args.use_mil
        self.sampled_use_mil = args.sampled_use_mil
        self.text_num_hidden_layers = args.text_num_hidden_layers
        self.visual_num_hidden_layers = args.visual_num_hidden_layers
        self.cross_num_hidden_layers = args.cross_num_hidden_layers
        self.loose_type = args.loose_type
        self.expand_msrvtt_sentences = args.expand_msrvtt_sentences
        self.train_frame_order = args.train_frame_order
        self.eval_frame_order = args.eval_frame_order
        self.freeze_layer_num = args.freeze_layer_num
        self.slice_framepos = args.slice_framepos
        self.linear_patch = args.linear_patch
        self.sim_header = args.sim_header
        self.pretrained_clip_name = args.pretrained_clip_name
        self.save_feature_path = args.save_feature_path
        self.cluster_algo = args.cluster_algo
        self.cluster_embedding = args.cluster_embedding
        self.cluser_embed_from_clip = args.cluser_embed_from_clip
        self.cluster_frame_embedding = args.cluster_frame_embedding
        self.adaptive_cls = args.adaptive_cls
        self.aggregation = args.aggregation
        self.cluster_iter_limit = args.cluster_iter_limit
        self.cluster_distance = args.cluster_distance
        self.cluster_threshold = args.cluster_threshold
        self.minkowski_norm_p = args.minkowski_norm_p
        self.temperature_new = args.temperature_new
        self.time_embedding = args.time_embedding
        self.pre_norm = args.pre_norm
    
    @abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError
    