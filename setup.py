from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simmpst",
    version="0.1.1",
    author="Sa Phyo Thu Htet",
    author_email="phyothuhtet@simbolomm.com",
    description="Simbolo Multilingual Partial-syllable Tokenizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaPhyoThuHtet/multilingual-partial-syllable-tokenizer",
    download_url='https://github.com/SaPhyoThuHtet/multilingual-partial-syllable-tokenizer/archive/0.1.0.tar.gz',
    install_requires=[
        'tensorflow>=2.15.0',
        'keras>=2.15.0'
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license="MIT",
)
