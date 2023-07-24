from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'An Efficient Scene Detector'
LONG_DESCRIPTION = 'A package that allows to detect scenes using audio features and save the scenes in mp4 format'

# Setting up
setup(
    name="EfficientSceneDetector",
    version=VERSION,
    author="Mayur Akewar",
    author_email="<mayurakewar87@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['tensorflow', 'soundfile', 'pandas', 'resampy', 'numpy', 'ffmpeg-python', 'moviepy'],
    keywords=['python', 'scene', 'audio', 'detect scenes'],
    classifiers=[
        "Development Status :: 1 - Development",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
