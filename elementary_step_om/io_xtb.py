
def read_xtb_out(content, quantity='energy'):
    """Reads gaussian output file

    - quantity = 'structure' - final structure form output.
    - quantity = 'atomic_numbers' - atmoic numbers
    - quantity = 'energy' - final energy from output.
    """

    if quantity == 'structure':
        return read_structure(content)

    elif quantity == 'atomic_symbols':
        return read_atomic_symbols(content)

    elif quantity == 'energy':
        return read_energy(content)

    elif quantity == 'converged':
        return read_converged(content)


def read_converged(content):
    """Check if program terminated normally"""
    for line in content.split('\n'):
        if '[ERROR] Program stopped due to fatal error' in line:
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
    return energy


def read_structure(content):
    """Read structure from output file """
    
    structure_block = content.split('final structure:')[1]

    atom_positions = []
    for line in structure_block.split('\n')[4:]:
        if len(line) == 0: # empty lines ends structure block
            break

        line = line.strip().split()
        if len(line) != 4:
            raise RuntimeError('Length of line does not match structure!')
        
        try:
            atom_position = list(map(float, line[1:]))
            atom_positions.append(atom_position)
        except ValueError:
            raise ValueError('Expected a line with one string and three floats.')

    return atom_positions

def read_atomic_symbols(content):
    """Read stmbols from output file """
    
    structure_block = content.split('final structure:')[1]

    atom_symbols = []
    for line in structure_block.split('\n')[4:]:
        if len(line) == 0: # empty lines ends structure block
            break

        line = line.strip().split()
        if len(line) != 4:
            raise RuntimeError('Length of line does not match structure!')
        
        try:
            atom_symbol = str(line[0])
            atom_symbols.append(atom_symbol)
        except ValueError:
            raise ValueError('Expected a line with one string and three floats.')

    return atom_symbols
