from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()
    if env_num == 101:
        settings.prj_dir = r'/home/liyunfeng/code/CSTNet'
        settings.result_plot_path = r'/home/liyunfeng/code/CSTNet/output/test/result_plots'
        settings.results_path = r'/home/liyunfeng/code/CSTNet/output/test/tracking_results'
        settings.save_dir = r'/home/liyunfeng/code/CSTNet'
        settings.segmentation_path = r'/home/liyunfeng/code/CSTNet'

        settings.lasher_path = r'/home/liyunfeng/code/dev/lasher'
        settings.rgbt234_path = r'/home/liyunfeng/code/dev/rgbt234'
        settings.vtuavst_path =r'F:\data\rgbt\vtuav\test_st'

    # Set your local paths here.
    elif env_num == 102:

        settings.prj_dir = r'E:\code\CSTNet'
        settings.result_plot_path = r'E:/code/CSTNet/output/test/result_plots'
        settings.results_path = r'E:/code/CSTNet/output/test/tracking_results'
        settings.save_dir = r'E:/code/CSTNet/output'
        settings.segmentation_path = r''
        settings.lasher_path = r'D:\BaiduNetdiskDownload\LasHeR_Divided_TraningSet&TestingSet'
        settings.vtuavst_path =r'F:\data\rgbt\vtuav\test_st'
    else:
        print('env_num:', env_num)
        raise
    return settings
