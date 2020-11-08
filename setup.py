from setuptools import setup, find_packages

setup(
    name="mnist",
    version="0.0.1",
    author="Sounak Bhattacharya",
    author_email="sounak.bhattacharya@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "matplotlib"],
)
