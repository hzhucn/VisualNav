from setuptools import setup


setup(
    name='visualnav',
    version='0.0.1',
    packages=[
        'visual_nav',
        'visual_nav.utils',
        'visual_sim',
        'visual_sim.envs',
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'imutils',
        'opencv-python',
        'airsim'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)