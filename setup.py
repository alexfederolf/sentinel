import os
from setuptools import find_packages, setup

requirements = []

if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        content = f.readlines()
    requirements.extend([x.strip() for x in content if "git+" not in x])

if os.path.isfile("requirements_dev.txt"):
    with open("requirements_dev.txt") as f:
        content = f.readlines()
    requirements.extend([x.strip() for x in content if "git+" not in x])


setup(
    name="sentinel",
    version="0.1.0",
    description="SENTINEL — ESA spacecraft anomaly detection",
    # src/-layout: packages live under src/, not at project root
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    test_suite="tests",
    include_package_data=True,
    zip_safe=False,
)
