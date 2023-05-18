from setuptools import setup, find_packages

setup(
    name='ashvqa',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'boto3',
        'torch',
        'transformers',
        'pillow',
        'matplotlib',
        'numpy',
        'tqdm'
    ],
)
