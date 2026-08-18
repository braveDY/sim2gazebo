import os
from glob import glob
from setuptools import find_packages, setup

package_name = "rl_policy_runtime"

policy_data = []
for root, _, files in os.walk("policy"):
    files_to_install = [
        os.path.join(root, name)
        for name in files
        if name.endswith((".yaml", ".py", ".pt", ".onnx", ".engine"))
    ]
    if files_to_install:
        policy_data.append((os.path.join("share", package_name, root), files_to_install))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        *policy_data,
    ],
    install_requires=["setuptools", "numpy", "PyYAML", "rich"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="ROS 2 Python runtime for deploying RL policies on the Go2 robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "deploy_node = rl_policy_runtime.deploy_node:main",
        ],
    },
)
