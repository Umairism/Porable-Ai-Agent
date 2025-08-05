#!/usr/bin/env python3
"""
Setup script for Portable AI Agent
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="portable-ai-agent",
    version="1.0.0",
    author="Muhammad Umair Hakeem",
    author_email="iamumair1124@gmail.com",
    description="A self-contained, offline-capable AI agent with self-learning capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/portable-ai-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Chat",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "portable-ai=main:main",
            "pai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "chatbot",
        "ai assistant",
        "offline ai",
        "self-learning",
        "privacy",
        "local ai",
        "personal assistant",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/portable-ai-agent/issues",
        "Source": "https://github.com/yourusername/portable-ai-agent",
        "Documentation": "https://github.com/yourusername/portable-ai-agent/wiki",
        "Changelog": "https://github.com/yourusername/portable-ai-agent/blob/main/CHANGELOG.md",
    },
)
