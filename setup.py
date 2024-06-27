from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="crystalformerapp",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    package_data={
        '': ['data/*', 'model/*'],
    },
    entry_points={
        'console_scripts': [
            'crystalgpt-app=crystalformerapp.gr_frontend:main'
        ]
    },    
    author="Schrodinger's Cat",
    author_email="gmy721212@163.com",
    description="CrystalFormer APP",
    keywords="Crystal Generation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

