from setuptools import find_packages, setup


def read_requirements(path):
    requirements = []
    with open(path) as file:
        for line in file:
            requirement = line.strip()
            if requirement and not requirement.startswith("#"):
                requirements.append(requirement)
    return requirements


requirements = read_requirements("requirements.txt")
setup(
    name="ctd",
    version="1.0.1",
    description="Computation-Through-Dynamics Toolkit",
    url="https://github.com/snel-repo/ComputationThroughDynamicsToolkit",
    license="BSD-3-Clause",
    license_files=(
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "ctd/data_modeling/LICENSE",
    ),
    install_requires=requirements,
    packages=find_packages(),
    py_modules=["utils"],
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)
