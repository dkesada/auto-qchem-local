from setuptools import setup

setup(
    name='auto-qchem-local',
    version='2.0.0dev',
    packages=['autoqchem', 'morfeus_ml', 'api'],
    url='https://github.com/dkesada/auto-qchem_exp',
    license='GPL',
    author='Andrzej Zuranski, Benjamin Shields, Jason Wang, Winston Gee, David Quesada',
    description='auto-qchem local version',
    long_description='automated dft calculation management software without slurm job structure',
    install_requires=['numpy>=1.22',
                      'pandas>=1.3',
                      'pyyaml>=6.0',
                      'scipy>=1.7',
                      'pymongo>=3.10',
                      'appdirs>=1.4',
                      'ipywidgets>=8.0',
                      'py3Dmol>=1.8',
                      'jupyterlab>=3.4',
                      'notebook>=6.4',
                      'ipywidgets>=8.0',
                      'xlrd>=2.0',
                      'openpyxl>=3.0',
                      'rdkit',
                      'matplotlib>=3.5',
                      'tqdm>=4.66.1'
                      ],
    python_requires='>=3.8'
)
