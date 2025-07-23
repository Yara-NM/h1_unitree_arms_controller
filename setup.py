from setuptools import setup, find_packages

setup(
    name="arms_controller",
    version="0.1.0",
    description="Unitree H1 Robot Arm Controller and IK Package",
    author="Your Name",
    author_email="Yara.Mahmoud.MechaEng@gmail.com",
    packages=find_packages(),
    install_requires=[
    
        # add dependencies as needed
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "arms_controller": [
            "assets/h1/*.urdf",
            "assets/h1/*.obj",
            "assets/h1/*.stl",
            "assets/h1/*.dae",
            # Add more patterns if you use other file types!
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
