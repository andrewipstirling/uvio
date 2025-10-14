# py_vins/setup.py
from setuptools import setup, find_packages

setup(
    name="py_vins",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
)