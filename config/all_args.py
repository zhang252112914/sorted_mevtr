import os
import argparse
import torch
import numpy as np
import random
import logging

def get_args(description='Me-Retriever on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    # the basic behavior args
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    
    # related file path
    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--loss', type=str, default='balanced', help='loss function used')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
   
    # evaluation batch size is uasually larger than training batch size
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--regularize', type=str, default='none', choices=['none', 'l2', 'dropout'],
                    help='Regularization method. Options are: none, l2, dropout.')

    # args related to models
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--alpha', type=float, default=1.0, help="The relative scale of v2t and t2v loss.")
    parser.add_argument('--dynamic_alpha', action='store_true', help="Dynamically adjust alpha for loss calculation.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--sim_lambda", default=0.0, type=float, help="The coefficient of added similarity term")
    parser.add_argument('--post_process', type=str, default='none', choices=['none', 'cluster'],
                        help="clustering over frames")
    parser.add_argument('--post_cluster_centroids', type=int, default=1, help='clustering frame length')

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    # due to version, I modify the --local_rank arg
    # parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--local_rank', type=int, default=int(os.getenv("LOCAL_RANK", -1)),
                        help='local rank for distributed training')
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf", "maxP"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument('--save_feature_path', type=str,
                        default=None,
                        help='Used to save the CLIP features')
    # cluster algorithms
    parser.add_argument('--cluster_algo', type=str, default='kmediods++',
                        choices=['kmediods++', 'pooling'],
                        help="The type of cluster algorithms.")

    parser.add_argument('--cluster_embedding', type=int, default=0,
                        help="Whether using cluser embedding or not.")

    parser.add_argument('--cluser_embed_from_clip', type=int, default=1,
                        help="Whether using CLIP pretrained positional embedding to initialize cluster embedding.")

    parser.add_argument('--cluster_frame_embedding', type=int, default=0,
                        help="Whether using cluser frame embedding or not.")

    parser.add_argument('--adaptive_cls', type=int, default=0,
                        help="Whether adaptive [CLASS] token fusion.")


    parser.add_argument('--aggregation', type=str, default=None,
                        choices=['mean', 'None'],
                        help="When clustering, how to aggregate a cluster.")

    parser.add_argument('--cluster_iter_limit', type=int, default=100,
                        help="Iteration limits of cluster algorithms.")

    parser.add_argument('--cluster_distance', type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="type of clustering distance.")

    parser.add_argument('--cluster_threshold', type=float, default=1e-5,
                        help="stop threshold for clustering.")

    parser.add_argument('--minkowski_norm_p', type=float, default=2.0,
                        help="p value for the p-norm distance to calculate between each vector pair.")

    # divide the pretrained temperature
    parser.add_argument('--temperature_new', type=float, default=1.0,
                        help='assign a new temperature to CLIP model')

    parser.add_argument('--time_embedding', type=int, default=0,
                        help="Add time embedding in CLIP model.")

    parser.add_argument('--pre_norm', type=int, default=0,
                        help="whether do l2 normalization before clustering.")

    args = parser.parse_args()

    # parameters check and adjust
    if args.sim_header == "tightTransf":
        args.loose_type = False
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args, logger