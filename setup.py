from setuptools import setup, find_packages

setup(
    name="ssamba",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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