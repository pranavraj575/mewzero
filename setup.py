from distutils.core import setup
from setuptools import find_packages

setup(
    name='mewzero',
    version='6.9.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'torch',
                      'matplotlib',
                      ],
    license='Liscence to Krill',
)
