from setuptools import setup
import sys
version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

install_requires = [
        "blockfs",
        "numpy",
        "tifffile",
        "tqdm"
    ]

setup(
    name="spimstitch",
    version=version,
    description=
    "Chung Lab oblique SPIM stitcher",
    long_description=long_description,
    install_requires=install_requires,
    author="Kwanghun Chung Lab",
    packages=["spimstitch", "spimstitch.commands"],
    entry_points={ 'console_scripts': [
        "dcimg2tif=spimstitch.commands.dcimg2tif:main",
        "stack2oblique=spimstitch.commands.stack2oblique:main",
        "oblique2stitched=spimstitch.commands.stitch:main"
    ]},
    url="https://github.com/chunglabmit/spimstitch",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)