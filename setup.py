import os
from setuptools import setup, find_packages

version = {}
with open(os.path.join("eco2ai", "__init__.py")) as f:
    exec(f.read(), version)

DEPENDENCIES = [
    "APScheduler",
    "pynvml>=5.6.2",
    "psutil",
    "py-cpuinfo",
    "numpy",
    "pandas",
    "tzlocal",
    "requests", 
]

setup(
    name="eco2ai_jsonbin",
    description="Carbon emission tracker for AI/ML experiments",
    long_description="Track CO2 emissions of AI/ML experiments and export results to JSON/JSONBin.",
    long_description_content_type="text/markdown",
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
    version=version["__version__"],
    python_requires=">=3.7",
)
