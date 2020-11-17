import os
import subprocess

from .io_xtb import read_xtb_out


def run_cmd(cmd):
    """ Run cmd to run QM program """
    cmd = cmd.split()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8')


class Calculator:

    def __init__(self, sdf=None, cpus=1):

        self.conformer = sdf
        self._ncores = cpus

        self._inp_filename = None
        self._working_dir = None

    def _make_working_dir(self):
        """ """
        wdir_name = self.conformer.label
        if os.chdir != wdir_name and not os.path.isdir(wdir_name):
            os.makedirs(wdir_name)

        self._make_working_dir = wdir_name
        os.chdir(self._make_working_dir)

    def add_structure(self, conf):
        self.conformer = conf


class xTB(Calculator):

    def __init__(self, *args, **kwds):
        self._xtb_kwds = self._make_cmd(kwds.pop('xtb_args', {'opt': 'loose'}))
        super().__init__(*args, **kwds)

    def _make_cmd(self, xtb_args):
        xtb_kwds = ''
        for arg, value in xtb_args.items():
            if value in [None, '']:
                xtb_kwds += f"--{arg.lower()} "
            else:
                xtb_kwds += f"--{arg.lower()} {value.lower()} "
        return xtb_kwds

    def write_input(self):
        """ """
        self._inp_filename = f"{self.conformer.label}.xyz"

        with open(f"{ self._inp_filename}", 'w') as inp:
            inp.write(self.conformer.write_xyz())

    def calculate(self, cpus=1, properties=['energy', 'structure']):
        """ """
        os.environ['OMP_NUM_THREADS'] = str(cpus)
        os.environ['MKL_NUM_THREADS'] = str(cpus)

        self._make_working_dir()
        self.write_input()

        output = run_cmd(f'xtb {self._inp_filename} {self._xtb_kwds}')
        with open('test.log', 'w') as f:
            f.write(output)
        os.chdir('..')

        results = dict()
        for prop in properties:
            results[prop] = read_xtb_out(output, prop)

        return results
