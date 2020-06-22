import tqdm
import os
import click
import tensorflow as tf
from pathlib import Path
from decouple import config
from contextlib import contextmanager
import gin

def get_data_path():
    return Path(config("DISENTANGLED_REPRESENTATIONS_DIRECTORY"))

@gin.configurable
def get_logs_path(suffix=None):
    base = Path(config("DISENTANGLED_REPRESENTATIONS_DIRECTORY")) / "logs"

    if suffix is not None:
        base /= suffix

    subdir = [b for b in base.iterdir()]
    if len(subdir) == 0:
        counter = 0
    else:
        counter = max([b.parts[-1] for b in subdir])
   
    return base / str(counter + 1)

def get_config_path():
    return Path(__file__).resolve().parent / "config"

@contextmanager
def config_path():
    old = Path('.')
    os.chdir(get_config_path())
    try:
        yield
    finally:
        os.chdir(old)

def parse_config_file(path):
    with config_path():
        gin.parse_config_file(path)


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

    def update(self, logs, interval=10):
        loss = logs.pop('loss')
        if self.n % interval == 0:
            self.postfix = 'Loss: {loss:.2f}, '.format(loss=loss) + ', '.join(key + ': ' + '{:.2f}'.format(logs[key]) for key in logs.keys())
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
