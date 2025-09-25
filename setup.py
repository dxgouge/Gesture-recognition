from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gesture-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive gesture recognition system for Rock-Paper-Scissors using multiple ML approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gesture-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "gesture-recognition-classification=Classification.rps_classification_webcam:main",
            "gesture-recognition-detection=ObjectRecognition.rps_detection_webcam:main",
            "gesture-recognition-mediapipe=MediapipesGestureRecognition.rps_mediapipe_landmarks:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.task", "*.pt"],
    },
    keywords="gesture recognition, computer vision, machine learning, yolo, mediapipe, rock paper scissors, hand detection",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gesture-recognition/issues",
        "Source": "https://github.com/yourusername/gesture-recognition",
        "Documentation": "https://github.com/yourusername/gesture-recognition#readme",
    },
)
