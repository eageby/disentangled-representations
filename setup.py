from setuptools import setup, find_packages

setup(name='disentangled', version='0.1', packages=find_packages(),
       entry_points='''[console_scripts]
        disentangled=main:cli
        disentangled-data=disentangled.dataset.serialize.prepare:cli
    ''',)

