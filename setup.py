from setuptools import find_packages, setup
import os


def parse_requirements(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]


with open("README.md", "r", encoding="utf-8", errors="ignore") as fh:
    long_description = fh.read()

version = {}
with open("ai_data_science_team/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)


setup(
    name="ai-data-science-team",
    version=version["__version__"],
    description="Build and run an AI-powered data science team.",
    author="Matt Dancho",
    author_email="mdancho@business-science.io",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/business-science/ai-data-science-team",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "machine_learning": ["h2o", "mlflow"],
        "data_science": ["pytimetk", "missingno", "sweetviz"],
        "all": ["h2o", "mlflow", "pytimetk", "missingno", "sweetviz"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
