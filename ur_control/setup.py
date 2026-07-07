from setuptools import setup

package_name = 'ur_control'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'EAIK'],
    zip_safe=True,
    maintainer='Cristian Beltran',
    maintainer_email='cristian.beltran@sinicx.com',
    description='UR control library: motion, force/compliance control, grippers (ROS 2 port).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'active_gripper_publisher = ur_control.active_gripper_publisher:main',
        ],
    },
)
