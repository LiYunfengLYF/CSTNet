import torch
from torch import nn

from lib.utils.registry import MODEL_REGISTRY, TRACKER_REGISTRY
from lib.utils.load import load_pretrain




@TRACKER_REGISTRY.register()
class CSTNet(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, cfg, env_num=0, training=False, ):
        super().__init__()

        self.backbone = MODEL_REGISTRY.get(cfg.MODEL.BACKBONE.TYPE)(**cfg.MODEL.BACKBONE.PARAMS)

        if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        if hasattr(self.backbone, 'finetune_track'):
            self.backbone.finetune_track(cfg=cfg, patch_start_index=1)

        self.head_type = cfg.MODEL.HEAD.TYPE
        self.box_head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)

        self.rgbt_fuse_search = nn.Identity()

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )


        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = enc_opt1 + enc_opt2

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()

        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, 16, 16)

        opt_feat = self.rgbt_fuse_search(opt_feat)


        if self.head_type == "center_head":
            # run the center head
            # score_map_ctr, bbox, size_map, offset_map = self.head(opt_feat, gt_score_map)
            out = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = out['pred_boxes']
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': out['score_map'],
                   'size_map': out['size_map'],
                   'offset_map': out['offset_map']}
            return out
        else:
            raise NotImplementedError