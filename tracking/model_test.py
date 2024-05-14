import time
import os
import sys
import torch
from thop import profile, clever_format


from lib.models.cstnet.trackerModel_ar_vit import TRACKER_REGISTRY
from lib.utils.load import load_yaml

model_name = r'cstnet'
yaml_name = r'baseline'

cfg = load_yaml(rf'E:\code\CSTNet\experiments/{model_name}/{yaml_name}.yaml')
model = TRACKER_REGISTRY.get(cfg.MODEL.NETWORK)(cfg, env_num=101, training=True)

model.load_state_dict(torch.load(r'E:\code\CSTNet\output\checkpoints\train\cstnet\baseline\CSTNet_ep0020.pth.tar')['net'])

z = torch.rand(1, 3, 128, 128)
x = torch.rand(1, 3, 256, 256)
input = ([z, z], [x, x])



with torch.no_grad():
    macs, params = profile(model, inputs=input, custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)



# avg_time = []
# for i in range(5):
#     tim = []
#     for i in range(1000):
#         s = time.time()
#         with torch.no_grad():
#             out = model([z, z],[x, x])
#         e = time.time()
#
#         tim.append(e - s)
#     if i == 0:
#         continue
#     tim[0] = tim[50]
#
#     tim_tensor = torch.tensor(tim).mean()
#
#     avg_time.append(tim_tensor)
#
# avg_time[0] = avg_time[2]
# avg_time_tensor = torch.tensor(avg_time)
# print(avg_time_tensor.mean())
# print(1 / avg_time_tensor.mean())
