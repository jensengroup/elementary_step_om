from rdkit import Chem

def reassign_atom_idx(mol):
    """ Assigns RDKit mol atomid to atom mapped id """
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key = lambda x: x[1])]
    
    return Chem.RenumberAtoms(mol, new_idx)


reactant = "[N+:1](=[B-:2](/[H:6])[H:7])(\[H:8])[H:9].[N+:3](=[B-:4](/[H:11])[H:12])(\[H:5])[H:10]"
product = "[N+:1](=[B-:2](/[H:6])[H:7])(\[H:9])[H:12].[N+:3](=[B-:4](/[H:8])[H:11])(\[H:5])[H:10]"

reac_mol = Chem.MolFromSmiles(reactant, sanitize=False)
Chem.SanitizeMol(reac_mol)
#reac_mol = reassign_atom_idx(reac_mol)

[atom.SetAtomMapNum(0) for atom in reac_mol.GetAtoms()]
Chem.MolToMolFile(reac_mol, 'reactant4_molblock.mol')



input_mol = Chem.MolFromMolFile('reactant4_molblock.mol', sanitize=False)
Chem.SanitizeMol(reac_mol)

for atom in input_mol.GetAtoms():
    print(atom.GetAtomMapNum())

print()
print(input_mol.GetNumConformers())
if all(input_mol.GetConformer().GetPositions()[:,2] == 0):
    input_mol.RemoveAllConformers()
print(input_mol.GetNumConformers())



