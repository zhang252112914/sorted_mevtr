import logging
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.meretrieverpretrained import MeRetrieverPretrained
from modules.util_func import show_log, check_attr, update_attr
from modules.module_clip import CLIP, convert_weights
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.util_module import PreTrainedModel, AllGather, CrossEnMulti, CrossEnMulti_unbalanced
from modules.cluster.fast_kmeans import batch_fast_kmedoids
from modules.util_module import all_gather_only as allgather


logger = logging.getLogger(__name__)

class MeRetriever(MeRetrieverPretrained):
    def __init__(self, cross_config, clip_state_dict, task_config, logger):
        super(MeRetriever, self).__init__(cross_config)
        # the task_config actually is the args from script
        self.task_config = task_config
        self.ignore_video_index = -1
        self.logger = logger

        # assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two), self.logger)

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.", self.logger)

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b
                            in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        self.cluster_inter = getattr(task_config, "cluster_inter", 0)
        if self.cluster_inter:
            self.cluster_algo = getattr(task_config, "cluster_algo", None)
            self.deep_cluster = getattr(task_config, "deep_cluster", 0)
            self.video_frames = getattr(task_config, "max_frames", None)
            self.time_embedding = getattr(task_config, "time_embedding", None)
            self.freeze_clip = getattr(task_config, "freeze_clip", 0)
            self.new_added_modules = getattr(task_config, "new_added_modules", [None, ])
            self.final_frames = task_config.target_frames_blocks[-1]
            self.f_frame_duration = self.video_frames // self.final_frames
            self.pre_visual_pooling = getattr(task_config, "pre_visual_pooling", 0)
            self.camoe_dsl = getattr(task_config, "camoe_dsl", False)

        show_log(task_config, "\t embed_dim: {}".format(embed_dim), self.logger)
        show_log(task_config, "\t image_resolution: {}".format(image_resolution), self.logger)
        show_log(task_config, "\t vision_layers: {}".format(vision_layers), self.logger)
        show_log(task_config, "\t vision_width: {}".format(vision_width), self.logger)
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size), self.logger)
        show_log(task_config, "\t context_length: {}".format(context_length), self.logger)
        show_log(task_config, "\t vocab_size: {}".format(vocab_size), self.logger)
        show_log(task_config, "\t transformer_width: {}".format(transformer_width), self.logger)
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads), self.logger)
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers), self.logger)

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch), self.logger)

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer), self.logger)

        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header), self.logger)
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        # cross_config.max_position_embeddings = 1+task_config.max_frames
        # as the exegesis in main file, the ritrieval task apply tight type
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config,
                                       "cross_num_hidden_layers", self.logger)
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        # self.loss_fct = CrossEn()
        if getattr(task_config, "loss", "balanced") == "unbalanced":
            self.loss_fct = CrossEnMulti_unbalanced()
        else:
            self.loss_fct = CrossEnMulti()

        self.regularization = getattr(task_config, "regularize", "none")
        self.multi2multi = (self.sim_header == 'maxP')
        self.post_process = getattr(task_config, 'post_process', 'none')

        self.apply(self.init_weights)

    def forward(self, text, text_mask, group_mask, video, video_mask=None, vt_mask=None):
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        if self.cluster_inter:
            video_mask = self.get_video_mask_after_cluster(video_mask)
            vt_mask = self.get_interval_after_cluster(group_mask, vt_mask)
        
        # this steps transform the text and video into the same space(embedding?)
        sequence_output, visual_output = self.get_sequence_visual_output(text, text_mask,
                                                                         video, video_mask, group_mask, shaped=True,
                                                                         video_frame=video_frame)
        if self.post_process == 'cluster':
            assign, medoids = batch_fast_kmedoids(visual_output, self.task_config.post_cluster_centroids,
                                                  distance=self.task_config.cluster_distance,
                                                  threshold=self.task_config.cluster_threshold,
                                                  iter_limit=self.task_config.cluster_iter_limit)
            idx = torch.arange(visual_output.shape[0], dtype=torch.long, device=visual_output.device).unsqueeze(-1)
            visual_output = visual_output[idx, medoids]
            video_mask = video_mask[idx, medoids]
            vt_mask = vt_mask[idx, :, medoids].permute(0, 2, 1)

        if self.training:
            if self.multi2multi:
                sim_matrix, sim_mask = self.get_similarity_multi2multi_logits(sequence_output, visual_output, video_mask,
                                                                       group_mask, vt_mask)
            else:
                sim_matrix, sim_mask = self.get_similarity_logits(sequence_output, visual_output, text_mask, video_mask,
                                                                  group_mask, shaped=True, loose_type=self.loose_type)
            sim_loss = self.loss_fct(sim_matrix, sim_mask)
            sim_loss2 = self.loss_fct(sim_matrix.T, sim_mask.T)
            reg_loss = None

            return sim_loss, sim_loss2, reg_loss
        else:
            return None

    def get_sequence_output(self, text, attention_mask, group_mask, shaped=False):
        bs = text.shape[0]
        res = []
        for i in range(bs):
            sequence_hidden = self.clip.encode_text(text[i][group_mask[i] > 0]).float()
            sequence_hidden = torch.concat((sequence_hidden, torch.zeros(text[i].shape[0] - sequence_hidden.shape[0],
                                                                         sequence_hidden.shape[1]).to(text.device)))
            res.append(sequence_hidden)
        ret = torch.stack(res)
        return ret

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, text, text_mask, video, video_mask, group_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(text, text_mask, group_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask,
                                                 output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask
    
    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask, ):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
    
    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask, ):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        return text_out, video_out
    
    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, group_mask,
                          sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat(
                (visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original
        
        # collect the visual_output and video_mask from all the devices
        if self.training:
            visual_output = allgather(visual_output, self.task_config, keep_itself=True)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config, keep_itself=True)
            group_mask = allgather(group_mask, self.task_config)
            # noinspection PyUnresolvedReferences
            torch.distributed.barrier()

        # the visual_output and video_mask are reshaped to the same shape
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequences = []
        sequence_mask = []
        for i in range(len(sequence_output)):
            temp = sequence_output[i][group_mask[i] == 1]
            temp = temp / temp.norm(dim=-1, keepdim=True)
            sequences.append(temp)
            temp = torch.zeros(len(temp), len(sequence_output)).to(sequence_output.device)
            temp[:, i] = 1
            sequence_mask.append(temp)

        sequence_output = torch.concat(sequences)
        sequence_mask = torch.concat(sequence_mask)
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits, sequence_mask
    
    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask, group_mask):
        sequences = []
        sequence_mask = []
        for i in range(len(sequence_output)):
            temp = sequence_output[i][group_mask[i] == 1]
            temp = temp / temp.norm(dim=-1, keepdim=True)
            sequences.append(temp)
            temp = torch.zeros(len(temp), len(sequence_output)).to(sequence_output.device)
            temp[:, i] = 1
            sequence_mask.append(temp)

        sequence_output = torch.concat(sequences).unsqueeze(1)
        sequence_mask = torch.concat(sequence_mask)
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        # step_size = b_text  # set smaller to reduce memory cost
        step_size = 1
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1).to(device=attention_mask.device,
                                                                   dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits, sequence_mask
    
    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, group_mask,
                              shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits, sequence_mask = self._loose_similarity(sequence_output, visual_output, attention_mask,
                                                                    video_mask, group_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits, sequence_mask = self._cross_similarity(sequence_output, visual_output, attention_mask,
                                                                    video_mask, group_mask)

        return retrieve_logits, sequence_mask

    def get_video_mask_after_cluster(self, video_mask):
        # an and logical, if any frame of the clustering frames are masked,
        # these clustering frames will be abondoned. Here just use the last mask value
        if self.cluster_algo in ['kmediods++', 'pooling', 'sparse_sampling', 'spectral']:
            inds = torch.arange(self.f_frame_duration - 1, video_mask.shape[-1],
                                video_mask.shape[-1] // self.final_frames,
                                dtype=torch.long, device=video_mask.device)

            return video_mask[:, inds]

        else:
            return video_mask
    
    def get_interval_after_cluster(self, group_mask, vt_mask):
        b, n = group_mask.shape
        temp = vt_mask.view(b, n, self.final_frames, self.f_frame_duration)
        res = torch.max(temp, dim=-1)[0]
        return res
    
    def get_similarity_multi2multi_logits(self, sequence_output, visual_output, video_mask, group_mask, vt_mask):
        """
        sequence_output: bs*27*512
        visual_output: bs*frames*512
        video_mask: bs*frames
        group_mask: bs*27
        vt_mask: bs*27*frames
        """
        if self.training:
            visual_output = allgather(visual_output, self.task_config, keep_itself=True)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config, keep_itself=True)
            group_mask = allgather(group_mask, self.task_config)
            vt_mask = allgather(vt_mask, self.task_config)
            # noinspection PyUnresolvedReferences
            torch.distributed.barrier()
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        all_sim = []
        all_mask = []
        for i in range(len(sequence_output)):
            sim_row = []
            mask_row = []
            seq = sequence_output[i][group_mask[i] > 0]
            seq = seq / seq.norm(dim=-1, keepdim=True)  # n_text * 512
            for j in range(len(visual_output)):
                vis = visual_output[j][video_mask[j] > 0]
                vis = vis / vis.norm(dim=-1, keepdim=True)  # n_frame * 512
                vt = vt_mask[i][group_mask[i] > 0]  # n_text * n_frame
                sim = torch.matmul(seq, vis.T) * self.clip.logit_scale.exp()
                mask = vt[:, video_mask[j] > 0]
                if i != j:
                    mask = torch.zeros_like(mask)
                # assert sim.shape == mask.shape
                sim_row.append(sim)
                mask_row.append(mask)
            sim_row = torch.concat(sim_row, dim=-1)
            mask_row = torch.concat(mask_row, dim=-1)
            all_sim.append(sim_row)
            all_mask.append(mask_row)
        all_sim = torch.concat(all_sim, dim=0)
        all_mask = torch.concat(all_mask, dim=0)
        return all_sim, all_mask