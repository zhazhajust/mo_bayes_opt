from setuptools import setup, find_packages

setup(
    name="mo_bayes_opt",
    version="0.1.0",
    author="Jie Cai",
    author_email="zhazhajust@gmail.com",
    description="Multi-Objective Bayesian Optimization with Gaussian Processes and adaptive noise",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhazhajust/mo_bayes_opt",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=1.10",
        "gpytorch>=1.8",
        "botorch>=0.8",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "tqdm>=4.60"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "jupyter"
        ]
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
