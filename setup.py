from setuptools import setup, find_packages

setup(
    name='vggt_slam',
    version='2.0.0',
    description='VGGT-SLAM (SL(4) feed-forward SLAM), extended with GPS fusion and metric scaling for trajectory inference.',
    packages=find_packages(include=['vggt_slam', 'vggt_slam.*']),
)


