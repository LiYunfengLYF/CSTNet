class EnvironmentSettings:
    def __init__(self, env_num):
        if env_num == 101:
            self.workspace_dir = r''
            self.tensorboard_dir = r''
            self.pretrained_networks = r''

            # RGBT
            self.lasher_train_dir = r''
            self.lasher_test_dir = r''


        elif env_num == 102:
            self.workspace_dir = r''
            self.tensorboard_dir = r''
            self.pretrained_networks = r''

            # SOT
            self.lasot_dir = r''
            self.got10k_dir = r''
            self.got10k_val_dir = r''
            self.lasot_lmdb_dir = r''
            self.got10k_lmdb_dir = r''
            self.trackingnet_dir = r''
            self.trackingnet_lmdb_dir = r''
            self.coco_dir = r''
            self.coco_lmdb_dir = r''
            self.lvis_dir = ''
            self.sbd_dir = ''
            self.imagenet_dir = r''
            self.imagenet_lmdb_dir = r''
            self.imagenetdet_dir = ''
            self.ecssd_dir = ''
            self.hkuis_dir = ''
            self.msra10k_dir = ''
            self.davis_dir = ''
            self.youtubevos_dir = ''

            # RGBT
            self.lasher_train_dir = r''
            self.lasher_test_dir = r''
        else:
            print(env_num)
            raise f'env_num:{env_num}'
