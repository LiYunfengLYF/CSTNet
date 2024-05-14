from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()
    if env_num == 101:
        settings.prj_dir = r''
        settings.result_plot_path = r''
        settings.results_path = r''
        settings.save_dir = r''
        settings.segmentation_path = r''
        settings.lasher_path = r''

    # Set your local paths here.
    elif env_num == 102:

        settings.prj_dir = r''
        settings.result_plot_path = r''
        settings.results_path = r''
        settings.save_dir = r''
        settings.segmentation_path = r''
        settings.lasher_path = r''

    else:

        print('env_num:', env_num)
        raise
    return settings
