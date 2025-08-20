#!/usr/bin/env python3
"""
Setup script for AI Inference Performance Optimization Tool
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-inference-optimizer",
    version="1.0.0",
    author="AI Performance Team",
    author_email="team@example.com",
    description="A Python-based tool to profile and optimize AI inference performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ambicuity/AI-Inference-Performance-Optimization-Tool",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "gpu": [
            "pynvml>=11.0",
            "gputil>=1.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-optimizer=ai_optimizer.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)