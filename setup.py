from setuptools import setup, find_packages

setup(
    name='ashvqa',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pillow',
        'numpy',
        'tqdm'
    ],
)
