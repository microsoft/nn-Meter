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
        'numpy==1.18.5', 
        'tqdm==4.47.0', 
        'networkx==2.4', 
        'requests==2.22.0', 
        'protobuf==3.17.1', 
        'PyYAML==5.4.1', 
        'scikit_learn==0.24.2', 
        'packaging==21.0', 
        'logging'
    ],
)
