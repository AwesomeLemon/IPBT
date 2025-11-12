import shutil

def save_explored_ckpt_to_path(task, explored_hps, ckpt_chosen_path, ckpt_path):
    shutil.copy(ckpt_chosen_path, ckpt_path)
