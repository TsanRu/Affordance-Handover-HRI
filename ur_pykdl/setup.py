from glob import glob
from setuptools import setup

package_name = 'ur_pykdl'

setup(
    name=package_name,
    version='0.2.0',
    packages=['ur_pykdl', 'ur_kdl'],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cristian Beltran',
    maintainer_email='cristian.beltran@sinicx.com',
    description='Simple implementation of PyKDL kinematics + kdl_parser_py (ROS 2 port).',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # standalone scripts (display_urdf, ur_kinematics) added here when ported.
        ],
    },
)
