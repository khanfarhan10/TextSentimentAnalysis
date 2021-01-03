from setuptools import setup
setup(
    name = 'TxT-CLI',
    version = '0.1.0',
    packages = ['TxT'],
    entry_points = {
        'console_scripts': [
            'TxT = TxT.__main__:main'
        ]
    })