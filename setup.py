from setuptools import setup, find_packages

DEPENDENCIES = [
    "APScheduler",
    "pynvml>=5.6.2",
    "psutil",
    "py-cpuinfo",
    "numpy",
    "pandas",
    "tzlocal",
    "requests",
    "setuptools>=45.0.0",
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eco2ai",
    version="0.3.13",
    description="Carbon emission tracker for AI/ML experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yinuo Yang",
    author_email="",
    url="https://github.com/YinuoYang327/eco2ai",
    packages=find_packages(),
    install_requires=DEPENDENCIES,
    package_data={
        "eco2ai": [
            "data/cpu_names.csv",
            "data/config.txt",
            "data/carbon_index.csv",
        ]
    },
    include_package_data=True,
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
    keywords="carbon emissions, CO2, sustainability, AI, ML, deep learning, environmental impact",
)
