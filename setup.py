import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vnhtr",
    version="0.0.2",
    author="nguyenhoanganh2002",
    author_email="anh.nh204511@gmail.com",
    description="Encoder-Decoder base for Vietnamese handwriting recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nguyenhoanganh2002/vnhtr",
    packages=setuptools.find_packages(),
    install_requires=[
        'pillow'
    ],
    keywords=['ocr', 'vnocr', 'htr', 'vnhtr'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)