from setuptools import setup

setup(
    name='auto-qchem-local',
    version='2.0.9dev',
    packages=['autoqchem_local', 'autoqchem_local.api', 'autoqchem_local.autoqchem', 'autoqchem_local.morfeus_ml'],
    url='https://github.com/dkesada/auto-qchem_exp',
    exclude_package_data={'': ['media', 'markdowns']},
    license='GPL',
    author='Andrzej Zuranski, Benjamin Shields, Jason Wang, Winston Gee, David Quesada',
    description='auto-qchem local version',
    long_description='automated dft calculation management software without slurm job structure',
    install_requires=['numpy>=1.22',
                      'pandas>=1.3',
                      'pyyaml>=6.0',
                      'scipy>=1.7',
                      'xlrd>=2.0',
                      'openpyxl>=3.0',
                      'rdkit',
                      'matplotlib>=3.5',
                      'tqdm>=4.66.1',
                      'func-timeout>=4.3.5',
                      'morfeus-ml>=0.7.2',
                      'qcengine>=0.29.0'
                      ],
    python_requires='>=3.8'
)
