import numpy as np
import onnxruntime

from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.models.cstnet.trackerModel_ar_vit import TRACKER_REGISTRY
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor, Preprocessor_onnx
from lib.utils.box_ops import clip_box, box_iou, deep_xywh2xyxy


class CSTNet_ONNX(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CSTNet_ONNX, self).__init__(params)

        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": str(0)}]
        self.ort_sess = onnxruntime.InferenceSession(self.params.checkpoint, providers=providers,
                                                     provider_options=provider_options)

        self.preprocessor = Preprocessor_onnx()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0

        self.template_input = []
        self.cfg = params.cfg
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)

    def initialize(self, image, info: dict):
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # forward the template once
        self.template_input = {'img_z1': template[:, :3, :, :], 'img_z2': template[:, 3:, :, :]}

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        ort_inputs = {'img_z1': self.template_input['img_z1'],
                      'img_z2': self.template_input['img_z2'],
                      'img_x1': search[:, :3, :, :],
                      'img_x2': search[:, 3:, :, :],
                      }

        ort_outs = self.ort_sess.run(None, ort_inputs)

        self.state = self.postprocessing(torch.tensor(ort_outs[1]), torch.tensor(ort_outs[2]),
                                         torch.tensor(ort_outs[3]), resize_factor, H, W)

        return {"target_bbox": self.state}

    def postprocessing(self, score_map, size_map, offset_map, resize_factor, H, W):
        response = self.output_window * score_map

        pred_boxes = self.head_cal_bbox(response, size_map, offset_map).view(-1, 4)
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

    def head_cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox


def get_tracker_class():
    return CSTNet_ONNX
