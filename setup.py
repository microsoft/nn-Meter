from setuptools import setup, find_packages


setup(
    name='nn_meter',
    version='1.0',
    description='nn-Meter is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices.',
    author='',
    author_email='',
    url='https://github.com/microsoft/nn-Meter',
    packages=find_packages(),
    package_data={
        'nn_meter': ['configs/*.yaml', 'kerneldetection/fusionlib/*.json'],
    },
    entry_points={
        'console_scripts': ['nn-meter=nn_meter.nn_meter:nn_meter_cli'],
    },
    install_requires=[
        'numpy', 'tqdm', 'networkx', 'requests', 'protobuf', 'PyYAML', 'scikit_learn', 'packaging', 'logging'
    ],
)
