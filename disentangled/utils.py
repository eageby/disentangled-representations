import os 

def disable_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def disable_info_output():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
