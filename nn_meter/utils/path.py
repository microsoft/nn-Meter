import os


def get_filename(path):
    return os.path.basename(path)

def get_filename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]
