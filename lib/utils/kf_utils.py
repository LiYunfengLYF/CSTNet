import torch
import torchvision.ops as ops
from lib.utils.box_ops import deep_xywh2xyxy, clip_box


def decode_muli_bbox(response, size_map, offset_map, num_biox, feat_size, state=None, resize_factor=None, H=None,
                     W=None):
    index = torch.where(response.flatten() > torch.sort(response.flatten())[0][- 1 - num_biox])[0]
    response_score = response.flatten()[index].tolist()

    size = size_map.flatten(2)[:, :, index].squeeze(dim=0)
    offset = offset_map.flatten(2)[:, :, index].squeeze(dim=0)
    idx_y = index // feat_size
    idx_x = index % feat_size

    x_ = ((idx_x + offset[:1]) / feat_size)
    y_ = ((idx_y + offset[1:]) / feat_size)

    box_list = []
    #####
    # box_search_list = []

    for i in range(len(index)):
        bbox = [x_[:, i].cpu() * 256 / resize_factor,
                y_[:, i].cpu() * 256 / resize_factor,
                size[0, i].cpu() * 256 / resize_factor,
                size[1, i].cpu() * 256 / resize_factor]
        # bbox2 = [x_[:, i].cpu() * 256,
        #         y_[:, i].cpu() * 256,
        #         size[0, i].cpu() * 256,
        #         size[1, i].cpu() * 256]
        # print(bbox)
        ####
        # box_search_list.append(bbox2)

        box = clip_box(map_box_back(bbox, resize_factor, state), H, W, margin=10)

        box_list.append(box)

    return box_list, response_score


def map_box_back(pred_box: list, resize_factor: float, state, search_size=256):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def NMS(pred_box_list, response_score, iou_threshold=0.6, update_all_list=None):
    # bbox_list_thresholded = []
    # bbox_list_new = []
    # response_score_new = []
    #
    # bboxs_tensor = torch.tensor(pred_box_list)
    # bbox_score_tensor = torch.tensor(response_score).unsqueeze(dim=1)
    #
    # boxes_info = torch.cat((bboxs_tensor, bbox_score_tensor), dim=1)
    #
    # boxes_sorted = sorted(boxes_info, reverse=True, key=lambda x: x[4])
    #
    # # Stage 1
    # for box in boxes_sorted:
    #     #  --nms1 use conf
    #     # if box[4] > 0.5 * max(response_score):
    #     #  --nms2 without conf
    #     if box[4] > 0:
    #         bbox_list_thresholded.append(box.tolist())
    #     else:
    #         pass
    #
    # while len(bbox_list_thresholded):
    #     current_box = bbox_list_thresholded.pop(0)
    #     bbox_list_new.append(current_box[:4])
    #
    #     response_score_new.append(current_box[4])
    #     for box in bbox_list_thresholded:
    #         iou = box_iou(torch.tensor(current_box[:4]).unsqueeze(dim=0), torch.tensor(box[:4]).unsqueeze(dim=0))[0]
    #         if iou > iou_threshold:
    #             bbox_list_thresholded.remove(box)
    # return bbox_list_new, response_score_new

    pred_box_tensor = torch.tensor(pred_box_list).float()

    bbox_score_tensor = torch.tensor(response_score).float()
    index = ops.nms(deep_xywh2xyxy(pred_box_tensor), bbox_score_tensor, iou_threshold)

    new_bbox = []
    new_score = []

    if update_all_list is not None:
        new_update_list = []

    for i in index:
        new_bbox.append(pred_box_tensor[i, :].tolist())
        new_score.append(float(bbox_score_tensor[i]))
        if update_all_list is not None:
            new_update_list.append(update_all_list[i])

    if update_all_list is not None:
        return new_bbox, new_score, new_update_list
    else:
        return new_bbox, new_score
