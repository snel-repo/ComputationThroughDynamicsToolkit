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
    version="1.0",
    install_requires=requirements,
    packages=find_packages(),
    py_modules=["utils"],
)
