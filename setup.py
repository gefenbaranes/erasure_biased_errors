from setuptools import setup, find_packages

setup(
    name='BiasedErasure',
    version='0.1.2',
    author='Gefen Baranes and Pablo Bonilla',
    author_email='gefenbaranes123@gmail.com',
    packages=find_packages(include=['BiasedErasure', 'BiasedErasure.*']),
    license='LICENSE',
    description='Simulate logical circuits with loss errors with efficient smart decoding',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.21',
        'networkx',
        'matplotlib',
        'stim==1.12.0',
        'pymatching==2.1.0',
        'sinter',
        'scipy==1.9.1'
    ],
)