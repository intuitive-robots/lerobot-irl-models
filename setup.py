from setuptools import setup, find_packages

setup(
    name='lerobot-irl-models', # A simple name for your package
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*']),
    # You can add dependencies here, e.g., install_requires=['numpy']
    install_requires=[], 
    # Use 'package_dir' to tell setuptools where to find the source
    package_dir={'': '.'},
)