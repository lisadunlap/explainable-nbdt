import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="explainable_nbdt", # Replace with your own username
    version="0.0.1",
    author="Lisa Dunlap",
    author_email="lisabdunlap@berkeley.edu",
    packages=setuptools.find_packages()
)