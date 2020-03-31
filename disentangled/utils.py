import tqdm
import os 
import click
import tensorflow as tf

def disable_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def disable_info_output(level=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)

def config():
    tf.random.set_seed(10)

class TrainingProgress(tqdm.tqdm):
    def __init__(self, iterable, **kwargs):
            
        left_bar = '    {n_fmt}/{total_fmt} ['
        right_bar = '] - ETA: {remaining} -{rate_inv_fmt}{postfix}'
        bar_format = left_bar +  '{bar}' + right_bar 

        super().__init__(
            iterable=iterable,
            bar_format=bar_format,
            ascii='.>>=',
            unit='it',
            dynamic_ncols=True,
            position=0,
            **kwargs
        )

    def update(self, loss, metrics, interval=10):
        if self.n % interval == 0:
            self.postfix = 'Loss: {loss:.2f}, '.format(loss=loss) + ', '.join(key + ': ' + '{:.2f}'.format(metrics[key]) for key in metrics.keys())
            self.refresh()

"""https://stackoverflow.com/questions/54073767/command-line-interface-with-multiple-commands-using-click-add-unspecified-optio"""

class AcceptAllCommand(click.Command):

    def make_parser(self, ctx):
        """Hook 'make_parser' and allow the opt dict to find any option"""
        parser = super(AcceptAllCommand, self).make_parser(ctx)
        command = self

        class AcceptAllLongOptsDict(dict):

            def __contains__(self, item):
                """If the parser does no know this option, add it"""

                if not super(AcceptAllLongOptsDict, self).__contains__(item):
                    # create an option name
                    if item[:2] != '--':
                        return False
                    
                    name = item.lstrip('-')

                    # add the option to our command
                    click.option(item)(command)

                    # get the option instance from the command
                    option = command.params[-1]

                    # add the option instance to the parser
                    parser.add_option(
                        [item], name.replace('-', '_'), obj=option)
                return True

        # set the parser options to our dict
        parser._long_opt = AcceptAllLongOptsDict(parser._long_opt)

        return parser
