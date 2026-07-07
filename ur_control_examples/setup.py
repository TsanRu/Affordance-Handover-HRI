from setuptools import setup

package_name = 'ur_control_examples'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cristian Beltran',
    maintainer_email='cristian.beltran@sinicx.com',
    description='Example scripts and demos for ur_control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_position_keyboard = ur_control_examples.joint_position_keyboard:main',
            'joint_position_mouse6d = ur_control_examples.joint_position_mouse6d:main',
            'controller_examples = ur_control_examples.controller_examples:main',
            'cartesian_compliance_controller_examples = ur_control_examples.cartesian_compliance_controller_examples:main',
            'ft_filter = ur_control_examples.ft_filter:main',
        ],
    },
)
