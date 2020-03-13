import tqdm
import os 

def disable_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def disable_info_output():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class TrainingProgress(tqdm.tqdm):
    def __init__(self, iterable):
            
        left_bar = '{n_fmt}/{total_fmt} ['
        right_bar = '] - ETA: {remaining} -{rate_inv_fmt}{postfix}'
        bar_format = left_bar +  '{bar}' + right_bar 

        super().__init__(
            iterable=iterable,
            bar_format=bar_format,
            ascii='.>>=',
            unit='it',
            dynamic_ncols=True,
        )

    def update(self, model):
        metrics = {m.name: m.result().numpy() for m in model.metrics}

        self.postfix = 'Loss: {loss:.2f}, '.format(loss=model.losses[0])
        self.postfix += ', '.join(key + ': ' + '{:.2f}'.format(metrics[key]) for key in metrics.keys())

        self.refresh()
