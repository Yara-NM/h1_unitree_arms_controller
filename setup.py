from setuptools import setup, find_packages

setup(
    name="arms_controller",
    version="0.1.0",
    description="Unitree G1 Robot Arm Controller and IK Package",
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
            "assets/g1/*.urdf",
            "assets/g1/*.obj",
            "assets/g1/*.stl",
            "assets/g1/*.dae",
            # Add more patterns if you use other file types!
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
