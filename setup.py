# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retail-customer-analytics",
    version="1.0.0",
    author="Prasanth",
    author_email="prasanthk3022@gmail.com",
    description="Enhanced retail customer experience through data analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/retail-customer-analytics",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/retail-customer-analytics/issues",
        "Documentation": "https://your-docs-link.com",
    },
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
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "retail-analytics=src.pipelines.data_pipeline:main",
            "train-models=src.pipelines.training_pipeline:main",
            "make-predictions=src.pipelines.prediction_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    zip_safe=False,
)