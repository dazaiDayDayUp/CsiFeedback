"""Package setup for csifeedback."""

from setuptools import find_packages, setup

setup(
    name="csifeedback",
    version="0.1.0",
    description="Unified PyTorch implementations of CLNet, STNet, and CsiNet for Massive MIMO CSI feedback",
    author="",
    author_email="",
    url="",
    packages=find_packages(exclude=["tests", "scripts", "archive", "experiments"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy<2.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "packaging>=21.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "thop>=0.1.1"],
    },
)
