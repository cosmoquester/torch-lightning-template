from setuptools import find_packages, setup

setup(
    name="torch-lightning-template",
    version="0.0.1",
    description="This repository is template for my pytorch lightning project.",
    python_requires=">=3.7",
    install_requires=[],
    url="https://github.com/cosmoquester/torch-lightning-template.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
