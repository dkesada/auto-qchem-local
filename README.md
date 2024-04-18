<p align="center">
  <img src="https://github.com/dkesada/auto-qchem_exp/blob/simple/media/autoqchem.png" alt="logo" width="600">
</p>

## Introduction

❗ This is an experimental, modified version of [Auto-QChem](https://github.com/doyle-lab-ucla/auto-qchem) where I have taken down all components related with the Slurm scheduler, MongoDB and cloud management so that it can be run locally only by installing the Python library from GitHub. I've simplified the package as much as possible so that the relevant functions that create .gjf files, extract Gaussian .log properties, calculate Morfeus descriptors and generate .csv datasets can be accessed from a simple api defined as an object. Once the user instantiates this object, all functionality can be accessed easily. 

Keep in mind that this also means that all concurrent computations are now performed sequentially locally unless you define them otherwise. With that said, the only functions that are time consuming are the Morfeus calculations. Depending on the molecule and the number of conformers, this can take quite a while. For example, calculating the descriptors of a molecule with 90 atoms and a metal component with 5 conformers can take around 15~20 minutes on an average machine, while calculating this same molecule on 1 or 2 conformers can take up to 3~4 minutes. Simpler molecules without metal atoms will take less time and will have less failures in the conformer optimization of Morfeus.

The idea of this fork is to provide the possibility to perform their own calculations to any lab that wants to do them, even if they do not have a cluster infrastructure.

## Quick links

### Installation instructions

You will need to install this autoq-chem version from GitHub with the following command:

(put the command here)

Also, beware that Morfeus calculations *only* work on Linux (and maybe MacOS, but I haven't tried it) machines, because they need the xtb handler for Python, and that is only available in Linux. This will not work on Windows.

### Usage examples

In the future I'll add some examples on how to use this package locally. It is simplified enough so that in a few function calls one can have a full dataset with the descriptors of the desired molecules.


