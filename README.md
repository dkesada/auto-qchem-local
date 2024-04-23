<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/autoqchem.png" alt="logo" width="600">
</p>

## Introduction

❗ This is an experimental, modified version of [Auto-QChem](https://github.com/doyle-lab-ucla/auto-qchem) where I have taken down all components related with the Slurm scheduler, MongoDB and cloud management so that it can be run locally only by installing the Python library from GitHub. I've simplified the package as much as possible so that the relevant functions that create .gjf files, extract Gaussian .log properties, calculate Morfeus descriptors and generate .csv datasets can be accessed from a simple api defined as an object. Once the user instantiates this object, all functionality can be accessed easily. 

Keep in mind that this also means that all concurrent computations are now performed sequentially locally unless you define them otherwise. With that said, the only functions that are time consuming are the Morfeus calculations. Depending on the molecule and the number of conformers, this can take quite a while. For example, calculating the descriptors of a molecule with 90 atoms and a metal component with 5 conformers can take around 15-20 minutes on an average machine, while calculating this same molecule on 1 or 2 conformers can take up to 3-4 minutes. Simpler molecules without metal atoms will take less time and will have less failures in the conformer optimization of Morfeus.

The idea of this fork is to provide the possibility to perform their own calculations to any lab that wants to do them, even if they do not have a cluster infrastructure.

## Quick links

### Installation instructions

You will need to install this autoq-chem version from GitHub with the following command:

```
pip install git+https://github.com/dkesada/auto-qchem_exp.git
```

Also, beware that Morfeus calculations **only** work on Linux (and maybe MacOS, but I haven't tried it) machines, because they need the xtb handler for Python, and that is only available in Linux. This will not work on Windows.

### Code structure

The main functionality of the package is separated into different objects inside the [api](https://github.com/dkesada/auto-qchem_exp/tree/master/autoqchem_local/api) module: the `GjfGenerator` class that controls the generation of Gaussian input files, the `MorfeusGenerator` class that controls the morfeus calculations, the `LogExtractor` class that controls the extraction of information out of Gaussian .log files and the `AutoChem` controller class that serves as the main entry point to the package.

### Usage examples

In the [markdowns](https://github.com/dkesada/auto-qchem_exp/tree/master/markdowns) folder there are some examples on how to use the package as a script tool with `argparse`. The [main_api.py](https://github.com/dkesada/auto-qchem_exp/blob/master/markdowns/main_api.py) shows how to use the `AutoChem` class as the main entry point to the api of the package. Each of these components can be used independently, but I would say that using only the `AutoChem` class is the easiest way to use the package.

As for a full example, we have prepared some files and folders to showcase how to use this package. Inside the [markdowns](https://github.com/dkesada/auto-qchem_exp/tree/master/markdowns) folder, there is the [example](https://github.com/dkesada/auto-qchem_exp/tree/master/markdowns/example) directory with some example files and folders that we will use in the following sections.

#### Input .gjf files generation

Let's start with the generation of .gjf files. For this, the only thing we require is a single .smi file with a SMILES code per line for each of the compounds we want to analyze with Gaussian. An example .smi file is stored [here](https://github.com/dkesada/auto-qchem_exp/tree/master/markdowns/example/input_gaussian_files). In this case, we would have a folder like this:

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/gjf_1.png" alt="gjf_1" width="600">
</p>

Then, we only need to use the `AutoChem` class to generate the .gjf files from this .smi file:

```python
from autoqchem_local.api.api import AutoChem

# Instantiate the AutoChem object
controller = AutoChem(log_to_file=True)

controller.generate_gjf_files(path_to_smi_file)
```

This will generate a separate .gjf and .smi file for each SMILES in the original .smi file inside a new ./output_gjf/ directory:

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/gjf_2.png" alt="gjf_1" width="600">
</p>

This generated .gjf files will be named after the InChI code of each molecule, because SMILES codes have characters that cannot be in file names. This files can now be inputed into Gaussian for calculation. The other .smi files with the same names as the .gjf files contain the SMILES code of each molecule and will be needed for the morfeus calculation part. Additionally, if the `log_to_file` parameter is set to `True`, a log file will be generated with all relevant execution information of the controller object. 

#### Full dataset generation

The dataset generation can be done all in one single function call or each part can be done individually. To begin this process, we need all .smi files and .log files if available in the same folder and with the same names. Additionally, a .xyz file with the coordinates of a conformer can also be present for each molecule. All files for the same molecule need to have the same name so that they can be joined in the same row in the final dataset. 

To run the full pipeline in one call, we use the following code:

```python
controller.generate_dataset(data_dir=path_to_folder, gaussian=True)
```

This generates intermediate files and eventually returns the full_dataset.csv file with all information of both the .log files if available and the morfeus properties of the molecules

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/full.png" alt="full" width="600">
</p>

All intermediate steps can be performed independently if so desired with the other functions of the `AutoChem` class.

#### Morfeus calculation

To calculate the morfeus properties of some molecules, we need the individual .smi files for each of the molecules inside a directory (they can be stored in further subdirectories inside, the controller will look for .smi files recursively through the dir tree) and optionally the .log and .xyz files **with the same names as the .smi files**.

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/morf_1.png" alt="morf_1" width="600">
</p>

With this, we can use the `AutoChem` object to calculate the morfeus properties for each molecule first and then we join all intermediate .csv files into a single one. Please, bear in mind that this is the most computationally expensive process (other than using Gaussian for calculations, but that is outside the scope of this package), and so it can take quite a while:

```python
# Calculate morfeus properties
controller.process_morfeus(data_dir=path_to_folder)
```

This method creates a .csv file for each processed molecule with its morfeus properties. Afterwards, we can join all of them together into a single table. This table can be your last step if you do not want to process .log files:

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/morf_2.png" alt="morf_2" width="600">
</p>


```python
# Join all morfeus .csv separate files into a single one
controller.join_morfeus_csv_files(data_dir=path_to_folder)
```

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/morf_3.png" alt="morf_3" width="600">
</p>

#### .log file extraction 

We can join all the extracted information from different .log files into a single .csv with a single function call:

```python
controller.process_log_files(data_dir=path_to_folder, output_path=path_to_folder)
```

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/log.png" alt="log" width="600">
</p>

#### Merging all files

In this last step, we merge both the morfeus calculations and the log extractions into a single pandas dataframe and save it to a .csv file, obtaining the same result as with the `generate_dataset()` function:

```python
# Join both morfeus and log files
res = controller.join_log_and_morfeus(log_dir=f'{path_to_folder}log_values.csv',
                                      morfeus_dir=f'{self._format_path(path_to_folder)}morfeus_values.csv')
                                
# Store the dataframe as the final .csv file
res.reset_index(drop=True, inplace=True)
res.to_csv(f'{path_to_folder}full_dataset.csv', index=False)                           
```

<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/master/media/full.png" alt="full" width="600">
</p>

#### Standalone script

If, rather than using the package as a Python module, one prefers using this functionality as a standalone bash script, there is an example on the [main_api.py](https://github.com/dkesada/auto-qchem_exp/blob/master/markdowns/main_api.py) file on how to define it. This file could be used as an entry point to the package functionality through simple command prompt calls using the `argparse` module.


