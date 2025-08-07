from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_USB'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'ultralytics', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='ares',
    maintainer_email='ares@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'video = visual_USB.video:main',
        'pid_map = visual_USB.pid_map:main',
        ],
    },
)
