import os

from .calculator import Calculator, CalculatorError
from elementary_step_om.io import io_gaussian

class GaussianCalculator(Calculator):
    """ """
    allowd_properties = ["energy", "structure", "frequencies", "irc_structure"]

    def __init__(
        self,
        kwds: str = "opt",
        external_script = None,
        charge: int = 0,
        spin: int = 1,
        properties: list = ["energy", "structure"],
        nprocs: int = 1,
        memory: int = 2,
        overwrite: bool = True,
        location: str = ".",
    ):
        # Check that the property reader is implemeted.
        for property in properties:
            if property not in self.allowd_properties:
                raise CalculatorError("Property not implemented.")

        self._charge = charge
        self._spin = spin
        self._properties = properties
        self._kwds = kwds
        self._external_script = external_script
        self._memory = memory

        super().__init__(nprocs=nprocs, location=location, overwrite=overwrite)

        self._setup_gaussian_enviroment()

    def _setup_gaussian_enviroment(self) -> None:
        """ """
        if "G16_CMD" not in os.environ:
            raise CalculatorError('No G16_CMD command. export G16_CMD="path to G16"')
        self._G16_CMD = os.environ["G16_CMD"] 

    def _make_cmd(self, input_filename: str) -> None:
        """ """
        g16_cmd = f"{self._G16_CMD} {input_filename}"
        return g16_cmd

    def _write_input(self, atoms: list, coords: list, namespace: str) -> str:
        """ """
        if len(atoms) != len(coords):
            raise CalculatorError("Length of atoms and coords doesn't match.")
        
        input_filename = namespace + ".com"
        with open(input_filename, "w") as inputfile:
            inputfile.write(f"%nprocs={self._nprocs} \n")
            inputfile.write(f"%mem={self._memory}GB \n")
            
            # update self._kwds because external gaussian needs nomirco kwd. 
            if self._external_script is not None:
                if "opt=(" in self._kwds:
                    tmp_kwds = self._kwds.split(',')
                    tmp_kwds.insert(-1, 'nomicro')
                    self._kwds = ",".join(tmp_kwds)
                elif "opt" == self._kwds:
                    self._kwds += "=nomicro"
            
            inputfile.write(f"# {self._kwds} \n")
            if self._external_script is not None:
                inputfile.write(f'# external="{self._external_script}" \n')

            inputfile.write(f"\n header text \n \n")
            inputfile.write(f"{self._charge}  {self._spin} \n")
            for atom, coord in zip(atoms, coords):
                inputfile.write(f"{atom}  {coord[0]} {coord[1]} {coord[2]} \n")
            inputfile.write("\n")
        
        return input_filename
    
    def _clean(self, working_dir: str) -> None:
        """ """
        self._remove_working_dir(working_dir)
    
    def __call__(self, atoms, coords, label):
        """ """
        results = {}
        
        working_dir = self._make_working_directory(namespace=label, overwrite=self._overwrite)
        input_filename = self._write_input(atoms, coords, label)
        cmd = self._make_cmd(input_filename)

        output, _ = Calculator.run_cmd(cmd)
        if io_gaussian.read_gaussian_out(output, property="converged"):
            results['normal_termination'] = True
            results['converged'] = True

            for property in self._properties:
                if (self._external_script is not None) and (property == "energy"):
                    property += "_external"
                property_value = io_gaussian.read_gaussian_out(output, property=property)

                if property_value is None:
                    results = {}
                    results["normal_termination"] = False
                    results['converged'] = False
                    break
                
                if property == "energy_external": # rename external energy
                    property = property.split('_')[0]

                results[property] = property_value
        else:
            results['normal_termination'] = False
            results['converged'] = False

        os.chdir(self._root_dir)
        self._clean(working_dir)
        return results