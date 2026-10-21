from setuptools import find_packages, setup

setup(
    name="dysarthric_speech_transformer",
    version="0.1.0",
    author="Mridul Manhas",
    description="Sequence-to-Sequence Dysarthric Speech Recognition System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "pandas>=2.0.0",
        "librosa>=0.10.0",
        "fastapi>=0.104.0",
        "pyyaml>=6.0.1",
    ],
)