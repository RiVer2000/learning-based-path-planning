#!/usr/bin/env python
"""
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

from setuptools import setup, find_packages

setup(name="planning_datasets_utils",
      version="0.1.0",
      description="Path Planning using Neural A* Search",
      author="Ryo Yonetani",
      author_email="ryo.yonetani@sinicx.com",
      url="https://github.com/omron-sinicx/planning-datasets",
      install_requires=[
          "torch==2.2.0",
          "numpy==1.22.0",
          "tqdm==4.66.3",
          "natsort==7.0.1",
          "ipython==8.10.0",
          "scikit-image==0.17.2",
          "pandas==1.0.3",
      ],
      packages=find_packages())
