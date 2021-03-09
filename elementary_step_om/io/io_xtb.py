import numpy as np

def read_xtb_out(content, property: str = 'energy' ):
    """Reads gaussian output file

    - quantity = 'structure' - final structure form output.
    - quantity = 'energy' - final energy from output.

    Works with xTB version 6.1.
    """
    if property == 'structure':
        return read_structure(content)
    elif property == 'energy':
        return read_energy(content)
    elif property == 'converged':
        return read_converged(content)
    else:
        raise NotImplementedError(f"Reader for {property} not implemented.")

def read_converged(content):
    """Check if program terminated normally"""
    
    error_msg = ['[ERROR] Program stopped due to fatal error',
                 '[WARNING] Runtime exception occurred',
                 'Error in Broyden matrix inversion!']

    for line in reversed(content.split('\n')):
        if any([msg in line for msg in error_msg]):
            return False
    return True


def read_energy(content):
    """Read total electronic energy """
    for line in content.split('\n'):
        if 'TOTAL ENERGY' in line:
            try:
                energy = float(line.strip().split()[3])
            except ValueError:
                raise ValueError('xTB energy not a float. Made for v6.3.3 output.')
    try: 
        return energy
    except:
        print(content)


def read_structure(content):
    """
    Read structure from output file 
    Works with xTB v. 6.1.
    """
    try:
        structure_block = content.split('final structure:')[1]
    except:
        return None

    atom_positions = []
    for line in structure_block.split('\n')[3:]:
        if line == "$end": 
            break

        line = line.strip().split()
        if len(line) != 4:
            raise RuntimeError('Length of line does not match structure!')
        
        try:
            atom_position = list(map(float, line[:3]))
            atom_positions.append(atom_position)
        except ValueError:
            raise ValueError('Expected a line with one string and three floats.')
    
    return np.asarray(atom_positions, dtype=float) * 0.529177249 # 6.1.3 is in bohr. :(

