"""
Setup script for Viral Clip Extractor
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="clip-extract",
    version="1.0.0",
    author="Clip Extraction Project",
    description="AI-powered multimodal video clip extraction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clip-extract",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.8.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "tensorflow-hub>=0.15.0",
        "ultralytics>=8.0.0",
        "openai-whisper>=20231117",
        "praat-parselmouth>=0.4.3",
        "ruptures>=1.1.9",
        "dlib>=19.24.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clip-extract=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pt"],
    },
)
