from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='svcviz',
    version='0.0.1',
    author='Victor Irekponor, Taylor Oshan',
    author_email='vireks@umd.edu',
    description='A python visualization package for ensuring reproducibility and replicability in spatially varying coefficient models',
    url='https://github.com/marquisvictor/svc-visualizer',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
)