# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from setuptools import setup, find_packages


setup(
    name='nn-meter',
    version='2.0',
    description='nn-Meter is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices.',
    long_description = open('README.md', encoding='utf-8').read(),
    long_description_content_type = 'text/markdown',
    author='Microsoft nn-Meter Team',
    author_email='nn-meter-dev@microsoft.com',
    url='https://github.com/microsoft/nn-Meter',
    project_urls={
        'Data of models': 'https://github.com/microsoft/nn-Meter/releases/tag/v1.0-data',
    },
    license = 'MIT',
    classifiers = [
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    packages=find_packages(),
    package_data={
        'nn_meter': [
            'configs/*.yaml',
            'configs/builder/backends/*.yaml',
            'configs/builder/fusion_rule_tester/*.yaml',
            'configs/builder/predictor_builder/*.yaml',
            'builder/kernel_predictor_builder/data_sampler/prior_config_lib/*.csv',
            'kernel_detector/fusion_lib/*.json'],
    },
    entry_points={
        'console_scripts': ['nn-meter=nn_meter.utils.nn_meter_cli.interface:nn_meter_cli'],
    },
    install_requires=[
        'numpy', 'pandas', 'tqdm', 'networkx', 'requests', 'protobuf', 'PyYAML', 'scikit_learn', 'packaging', 'jsonlines'
    ],
)
