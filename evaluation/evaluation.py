import torch
import numpy as np

from modules.util_func import parallel_apply_2
from modules.metrics import compute_metrics_together
from modules.cluster.fast_kmeans import batch_fast_kmedoids

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    sim_masks = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, group_mask = b1
        sequence_output, text_id = batch_sequence_output_list[idx1]
        each_row = []
        masks = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, vt_mask = b2
            visual_output = batch_visual_output_list[idx2]
            if model.multi2multi:
                b1b2_logits, sim_mask = model.get_similarity_sphere_eval(sequence_output, visual_output,
                                                                        video_mask, group_mask, text_id, idx2)
            else:
                b1b2_logits, sim_mask = model.get_similarity_logits(sequence_output, visual_output, input_mask,
                                                                    video_mask,
                                                                    group_mask, loose_type=model.loose_type)
                if text_id != idx2:
                    sim_mask = torch.zeros_like(b1b2_logits)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            sim_mask = sim_mask.cpu().detach().numpy()
            if b1b2_logits.shape[0] != sim_mask.shape[0] or b1b2_logits.shape[1] != sim_mask.shape[1]:
                print("ERROR: Shape inconsistency")
                raise AssertionError
            each_row.append(b1b2_logits)
            masks.append(sim_mask)
        each_row = np.concatenate(each_row, axis=-1)
        masks = np.concatenate(masks, axis=-1)
        sim_matrix.append(each_row)
        sim_masks.append(masks)
    return sim_matrix, sim_masks

def eval_epoch(args, model, test_dataloader, device, n_gpu, logger):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    logger.info("model evaluation on {} GPU".format(n_gpu))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, group_mask, video, video_mask, vt_mask = batch
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            if model.cluster_inter:
                video_mask = model.get_video_mask_after_cluster(video_mask)
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, input_mask,
                                                                              video, video_mask, group_mask)
            if args.post_process == 'cluster':
                assign, medoids = batch_fast_kmedoids(visual_output, args.post_cluster_centroids,
                                                      distance=args.cluster_distance,
                                                      threshold=args.cluster_threshold,
                                                      iter_limit=args.cluster_iter_limit)
                idx = torch.arange(visual_output.shape[0], dtype=torch.long, device=visual_output.device).unsqueeze(-1)
                visual_output = visual_output[idx, medoids]
                video_mask = video_mask[idx, medoids]
                vt_mask = vt_mask[idx, :, medoids]
            elif args.post_process == 'perceiver':
                visual_output = model.perceiver_sampler(visual_output, video_mask)
                video_mask = torch.ones(visual_output.shape[0], visual_output.shape[1]).to(visual_output.device)

            batch_sequence_output_list.append((sequence_output, bid))
            batch_list_t.append((input_mask, group_mask,))

            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask, vt_mask))

            if (bid + 1) % args.n_display == 0:
                logger.info("Evaluation step: {}/{}".format(bid, len(test_dataloader)))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [(b[0].to(devc), b[1]) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[did], batch_list_v_splits[did],
                                      batch_t_output_splits[did], batch_v_output_splits[did]) for did in device_ids]
            parallel_outputs, parallel_mask = parallel_apply_2(_run_on_single_gpu, model, parameters_tuple_list,
                                                               device_ids)
            sim_matrix, sim_matrix_mask = [], []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
                sim_matrix_mask += parallel_mask[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_mask = np.concatenate(sim_matrix_mask, axis=0)
        else:
            sim_matrix, sim_matrix_mask = _run_on_single_gpu(model, batch_list_t, batch_list_v,
                                                             batch_sequence_output_list,
                                                             batch_visual_output_list)
            sim_matrix = np.concatenate(sim_matrix, axis=0)
            sim_matrix_mask = np.concatenate(sim_matrix_mask, axis=0)

    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics, vt_metrics = compute_metrics_together(sim_matrix.T, sim_matrix_mask.T)
    logger.info("Text to video:")
    for key in tv_metrics:
        logger.info("{}: {}".format(key, tv_metrics[key]))
    logger.info("Video to text:")
    for key in vt_metrics:
        logger.info("{}: {}".format(key, vt_metrics[key]))
