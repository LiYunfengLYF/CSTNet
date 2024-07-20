import time

import numpy as np
import torch
import onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


from lib.models.cstnet.trackerModel_ar_vit import TRACKER_REGISTRY
from lib.utils.load import load_yaml

model_name = r'cstnet_onnx'
yaml_name = r'small_onnx'

cfg = load_yaml(rf'E:\code\CSTNet\experiments/{model_name}/{yaml_name}.yaml')
torch_model = TRACKER_REGISTRY.get(cfg.MODEL.NETWORK)(cfg, env_num=101, training=True)

# torch_model.load_state_dict(,dtype=torch.int32
#     torch.load(r'E:\code\CSTNet\output\checkpoints\train\cstnet\baseline\CSTNet_ep0020.pth.tar')['net'])

img_z1 = torch.rand(1, 3, 128, 128)
img_z2 = torch.rand(1, 3, 128, 128)
img_x1 = torch.rand(1, 3, 256, 256)
img_x2 = torch.rand(1, 3, 256, 256)
#
# img_z1 = torch.rand((1, 3, 128, 128), dtype=torch.float32)
# img_z2 = torch.rand((1, 3, 128, 128), dtype=torch.float32)
# img_x1 = torch.rand((1, 3, 256, 256), dtype=torch.float32)
# img_x2 = torch.rand((1, 3, 256, 256), dtype=torch.float32)
#
save_name = 'CSTNet_onnx_ep0020.onnx'
torch.onnx.export(torch_model,  # model being run
                  (img_z1, img_z2, img_x1, img_x2),  # model input (a tuple for multiple inputs)
                  save_name,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['img_z1', 'img_z2', 'img_x1', 'img_x2'],  # model's input names
                  output_names=['outputs_coord'],  # the model's output names
                  # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                  #               'output': {0: 'batch_size'}}
                  )


"""########## inference with the onnx model ##########"""
onnx_model = onnx.load(save_name)
onnx.checker.check_model(onnx_model)
print("creating session...")
providers = ["CUDAExecutionProvider"]
provider_options = [{"device_id": str(0)}]

ort_session = onnxruntime.InferenceSession(save_name,
                                           providers=providers,
                                           provider_options=provider_options)
# ort_session.set_providers(["TensorrtExecutionProvider"],
#                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
print("execuation providers:")
print(ort_session.get_providers())
# compute ONNX Runtime output prediction

"""begin the timing"""
t_pyt = 0  # pytorch time
t_ort = 0  # onnxruntime time
N = 1000
order = 0

for input_meta in ort_session.get_inputs():
    print(f"输入名称: {input_meta.name}, 形状: {input_meta.shape}")

for i in range(N):
    # generate data
    img_z1 = to_numpy(torch.rand(1, 3, 128, 128))
    img_z2 = to_numpy(torch.rand(1, 3, 128, 128))
    img_x1 = to_numpy(torch.rand(1, 3, 256, 256))
    img_x2 = to_numpy(torch.rand(1, 3, 256, 256))
    # onnx inference
    ort_inputs = {'img_z1': img_z1,
                  'img_z2': img_z2,
                  'img_x1': img_x1,
                  'img_x2': img_x2,
                  }
    s_ort = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    e_ort = time.time()
    if N < 1:
        continue
    order = order + 1
    lat_ort = e_ort - s_ort
    t_ort += lat_ort

    # print("onnxruntime latency: %.2fms" % (lat_ort * 1000))
print("pytorch model average latency", t_pyt / order * 1000)
print("onnx model average latency:", t_ort / order * 1000)
