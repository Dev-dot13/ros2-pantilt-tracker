from setuptools import find_packages, setup

package_name = 'pantilt_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='ROS2 pan-tilt target tracker',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node    = pantilt_tracker.detector_node:main',
            'controller_node  = pantilt_tracker.controller_node:main',
            'viz_node         = pantilt_tracker.viz_node:main',
            'camera_node      = pantilt_tracker.camera_node:main',
            'motor_driver_node = pantilt_tracker.motor_driver_node:main',
            'llm_node = pantilt_tracker.llm_node:main',
            'command_interface_node = pantilt_tracker.command_interface_node:main',
            'voice_input_node = pantilt_tracker.voice_input_node:main',
        ],
    },
)
