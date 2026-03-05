from setuptools import setup, find_packages

setup(
    name="equiml",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "equiml = equiml.cli:main",
        ],
    },
)
