from setuptools import setup, find_packages

setup(
    name='BiasedErasure',
    version='0.1.5',
    author='Gefen Baranes and Pablo Bonilla',
    author_email='gefenbaranes123@gmail.com',
    packages=find_packages(include=['BiasedErasure', 'BiasedErasure.*']),
    license='LICENSE',
    description='Simulate logical circuits with loss errors with efficient smart decoding',
    long_description=open('README.md').read(),
    install_requires=[
        'networkx==2.8.4',
        'matplotlib==3.5.2',
        'stim==1.12.0',
        'pymatching==2.1.0',
        'sinter==1.11.0',
        'scipy==1.9.1',
        'stim==1.12.0',
        'numpy==1.21.6',
    ],
)