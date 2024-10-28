import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="fpgai_engine",
    version="0.0.1",
    author="Umut Can Altin",
    author_email="umut.altin@donders.ru.nl",
    description=("FPGAI HLS Engine"),
    license="MIT",
    keywords="fpga,hls,deep learning",
    url="",
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT License"
    ],
        install_requires=[
        'numpy',  
        'google.protobuf.json_format.MessageToDict',  
        'onnx',
        'torch'
    ]
)



