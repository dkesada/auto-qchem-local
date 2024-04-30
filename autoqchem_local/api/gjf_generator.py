import os
from autoqchem_local.autoqchem.molecule import molecule
from autoqchem_local.autoqchem.gaussian_input_generator import gaussian_input_generator
import func_timeout
import logging


logger = logging.getLogger(__name__)


class GjfGenerator:
    def __init__(self, log=None, conv_timeout=40, workflow_type="custom", workdir_gjf='./output_gjf', theory="B3LYP",
                 solvent="none", light_basis_set="6-31+G(d,p)", heavy_basis_set="SDD", generic_basis_set="genecp",
                 max_light_atomic_number=25):
        if log:
            self.logger = log
        else:
            self.logger = logger
        self.conv_timeout = conv_timeout
        self.workflow_type = workflow_type
        self.workdir_gjf = workdir_gjf
        self.theory = theory
        self.solvent = solvent
        self.light_basis_set = light_basis_set
        self.heavy_basis_set = heavy_basis_set
        self.generic_basis_set = generic_basis_set
        self.max_light_atomic_number = max_light_atomic_number

    def create_gjf_for_molecule(self, smiles):
        """
        Generate Gaussian input files for a molecule

        :param smiles: a SMILES code from a molecule
        """

        # create Gaussian files
        try:
            self.logger.info(f'Now processing SMILES {smiles}')
            m = func_timeout.func_timeout(self.conv_timeout, lambda: molecule(smiles, num_conf=1))
            simplified_name = ''.join(e for e in smiles if e.isalnum())
            molecule_workdir = os.path.join(f"{self.workdir_gjf}", simplified_name)
            gig = gaussian_input_generator(m, self.workflow_type, self.workdir_gjf, self.theory, self.solvent,
                                           self.light_basis_set, self.heavy_basis_set, self.generic_basis_set,
                                           self.max_light_atomic_number)
            gig.create_gaussian_files()
            with open(f'output_gjf/{m.inchikey}_conf_0.gjf.smi', "w") as f:
                f.write(smiles)

        except func_timeout.FunctionTimedOut:
            self.logger.error(f"Timed out! Possible bad conformer Id for molecule {smiles}")
        except ValueError as e:
            self.logger.error(f"Bad conformer Id for molecule {smiles}. Error: {e}")
        except Exception as e:
            self.logger.error(f"Could not convert molecule {smiles}. Error: {e}")
