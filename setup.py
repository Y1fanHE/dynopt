import os
from setuptools import setup, find_packages


setup(name="dynopt",
      version="0.0.1",
      author="Yifan He",
      author_email="heyif@outlook.com",
      license="MIT",
      packages=find_packages(include=("dynopt")),
      install_requires=["numpy"],
      tests_require=["pytest"],)
