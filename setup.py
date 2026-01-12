from setuptools import setup, find_packages

setup(
    name="learning-framework",
    version="0.1.0",
    description="Interactive DL/RL mastery framework",
    author="LH",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "paramiko>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "lf=learning_framework.cli:cli",
        ],
    },
)
