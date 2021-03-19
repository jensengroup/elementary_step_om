import os
import shutil
import subprocess

class CalculatorError(Exception):
    """ Exemption for calcualtor errors """


class Calculator:
    def __init__(self, nprocs: int = 1, location: str = ".", overwrite: bool = True):

        self._nprocs = nprocs
        self._location = location
        self._overwrite = overwrite
        self._root_dir = os.getcwd()

    def _make_working_directory(
        self, namespace: str = None, overwrite: bool = False
    ) -> str:
        """ Make directory to to run calculations in. """
        working_dir = os.path.join(self._location, namespace)
        if os.path.isdir(working_dir) and self._overwrite:
            shutil.rmtree(working_dir, ignore_errors=True)

        elif os.path.isdir(working_dir):
            raise CalculatorError("Working directory allready exists")

        os.makedirs(working_dir)
        os.chdir(working_dir)

        return working_dir

    def _remove_working_dir(self, namespace: str) -> None:
        """ Removes working directory all and files within """
        working_dir = os.path.join(self._location, namespace)
        os.chdir(self._root_dir)
        shutil.rmtree(working_dir)

    @staticmethod
    def run_cmd(cmd):
        """ Run cmd to run QM program """
        cmd = cmd.split()
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, err = p.communicate()
        return output.decode("utf-8"), err.decode("utf-8")