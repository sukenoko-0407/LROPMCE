from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def measure_mol_props(smiles: str) -> dict:
    """
    Measure molecular properties for constraints.
    Returns: {"HAC": int, "MW": float, "cnt_hetero": int, "cnt_chiral": int}
    
    For BranchNodes with dummy atoms, the dummy is usually excluded if we want "real molecule" props.
    However, RDKit treats * as atom #0.
    The spec says: "計数対象は「ダミー原子を除いた実分子部分」"
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"HAC": 0, "MW": 0.0, "cnt_hetero": 0, "cnt_chiral": 0}

    # Count atoms excluding dummy
    hac = 0
    hetero = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0: # Dummy
            continue
        if atom.GetAtomicNum() != 1: # Not H (implicitly handled by GetAtoms usually, unless AddHs)
            hac += 1
            if atom.GetAtomicNum() != 6: # Not C
                hetero += 1
                
    # MW
    # Descriptors.MolWt includes dummy? Dummy MW is 0 usually.
    mw = Descriptors.MolWt(mol)
    
    # Chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    cnt_chiral = len(chiral_centers)

    return {
        "HAC": hac,
        "MW": mw,
        "cnt_hetero": hetero,
        "cnt_chiral": cnt_chiral
    }
