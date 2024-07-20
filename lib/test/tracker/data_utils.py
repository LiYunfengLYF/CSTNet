import torch
import numpy as np
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449]).view((1, 6, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226]).view((1, 6, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        if img_tensor.shape[1] == 3:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449]).view((1, 6, 1, 1)).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226]).view((1, 6, 1, 1)).cuda()
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)

class Preprocessor_onnx(object):
    def __init__(self):
        self.mean6 = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449]).view((1, 6, 1, 1)).cuda()
        self.std6 = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226]).view((1, 6, 1, 1)).cuda()

        self.mean3 = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std3 = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        if img_tensor.shape[1] == 3:
            img_tensor_norm = ((img_tensor / 255.0) - self.mean3) / self.std3  # (1,3,H,W)
        else:
            img_tensor_norm = ((img_tensor / 255.0) - self.mean6) / self.std6  # (1,6,H,W)
        return img_tensor_norm.cpu().numpy().astype(np.float32)



