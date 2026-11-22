import os
from pathlib import Path
from setuptools import find_packages, setup

package_name = "telepathy_receptor"
soul_file = Path(__file__).resolve().parent / "persona" / "SOUL.md"
identity_file = Path(__file__).resolve().parent / "persona" / "identity.yaml"
package_root = Path(__file__).resolve().parent
soul_source = os.path.relpath(soul_file, package_root)
identity_source = os.path.relpath(identity_file, package_root)

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/persona", [soul_source, identity_source]),
    ],
    install_requires=["setuptools", "httpx", "PyYAML"],
    zip_safe=True,
    maintainer="OppaAI",
    maintainer_email="oppa.ai.org@gmail.com",
    description="ROS adapter for Grace's Interaction conversation engine",
    license="GPL-3.0-only",
    entry_points={"console_scripts": ["telepathy_receptor = telepathy_receptor.node:main"]},
)
