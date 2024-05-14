from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.utils.load import load_yaml


def parameters(yaml_name: str, env_num=0):
    params = TrackerParams()
    save_dir = env_settings(env_num).save_dir
    # update default config from yaml file
    prj_dir = env_settings(env_num).prj_dir
    yaml_file = os.path.join(prj_dir, 'experiments/cstnet/%s.yaml' % yaml_name)

    params.cfg = load_yaml(yaml_file)
    params.tracker_param = yaml_name

    # template and search region
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.search_size = params.cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/cstnet/%s/%s_ep%04d.pth.tar" %
                                     (yaml_name, params.cfg.MODEL.NETWORK, params.cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
