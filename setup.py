from setuptools import setup, find_packages

setup(name='disentangled', version='0.1', packages=find_packages(),
       entry_points='''[console_scripts]
        disentangled-main=main:cli
        disentangled-training=bins.disentangled_training:main
        disentangled-prepare-data=bins.prepare:main
        disentangled-template=bins.disentangled_template:main
    ''',)

