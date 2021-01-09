from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    reqs = f.read()
with open("README.md") as f:
    readme = f.read()

setup(
    name="speech_processing",
    version="0.1",
    packages=find_packages(),
    license="LICENSE.txt",
    long_description=readme,
    install_requires=reqs.strip().split("\n"),
)
