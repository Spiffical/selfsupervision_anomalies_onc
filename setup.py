from setuptools import setup, find_packages

setup(
    name="ssamba",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "h5py"
    ]
) 