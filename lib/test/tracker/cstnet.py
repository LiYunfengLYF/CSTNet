from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.models.cstnet.trackerModel_ar_vit import TRACKER_REGISTRY
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_iou, deep_xywh2xyxy
from lib.utils.kf_utils import decode_muli_bbox, NMS


class CSTNet(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CSTNet, self).__init__(params)

        network = TRACKER_REGISTRY.get(params.cfg.MODEL.NETWORK)(cfg=params.cfg, env_num=None, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'],
                                strict=True)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=[self.z_dict1.tensors[:, :3, :, :], self.z_dict1.tensors[:, 3:, :, :]],
                search=[x_dict.tensors[:, :3, :, :], x_dict.tensors[:, 3:, :, :]], ce_template_mask=self.box_mask_z)

        self.state = self.postprocessing(out_dict['score_map'], out_dict['size_map'], out_dict['offset_map'],
                                         resize_factor, H, W)

        return {"target_bbox": self.state}

    def postprocessing(self, score_map, size_map, offset_map, resize_factor, H, W):
        response = self.output_window * score_map

        pred_boxes = self.network.box_head.cal_bbox(response, size_map, offset_map).view(-1, 4)
        # (cx, cy, w, h) [0,1]
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        # get the final box result
        bbox = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        return bbox

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CSTNet
