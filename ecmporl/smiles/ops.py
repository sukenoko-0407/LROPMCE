from rdkit import Chem
from typing import List, Optional

def hydrogen_replace(smiles_with_D: str, *, canonical: bool = True) -> List[str]:
    """
    入力SMILES中の各重水素([2H])を「1個だけ」ダミーアトム(*)に置換したSMILESを全列挙し、
    重複を除いたリストを返す。
    """
    mol = Chem.MolFromSmiles(smiles_with_D)
    if mol is None:
        raise ValueError(f"SMILES parse error: {smiles_with_D}")

    d_indices = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetIsotope() == 2
    ]
    if not d_indices:
        return []

    out = set()

    for idx in d_indices:
        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(idx)

        # [2H] -> * (AtomicNum 0)
        a.SetAtomicNum(0)
        a.SetIsotope(0)
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)

        m2 = rw.GetMol()
        Chem.SanitizeMol(m2) 

        s2 = Chem.MolToSmiles(m2, isomericSmiles=True, canonical=canonical)
        out.add(s2)

    return sorted(out)


def smiles_addH_then_deuterate(smiles_with_dummy: str, *, canonical: bool = True) -> str:
    """
    ダミーアトム(*)を含むSMILESを受け取り、
    1) RDKitで分子化
    2) 明示的に水素付加 (AddHs)
    3) 付加された水素のみを重水素(同位体=2)に置換
    したSMILESを返す。
    """
    mol = Chem.MolFromSmiles(smiles_with_dummy)
    if mol is None:
        raise ValueError(f"SMILES parse error: {smiles_with_dummy}")

    n0 = mol.GetNumAtoms()
    molH = Chem.AddHs(mol)

    for idx in range(n0, molH.GetNumAtoms()):
        atom = molH.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:    # H
            atom.SetIsotope(2)          # D

    return Chem.MolToSmiles(molH, isomericSmiles=True, canonical=canonical)


def combine_smiles(branch_smiles_with_dummy: str, frag_smiles: str) -> str:
    """
    Combines a branch smiles (with dummy '*') and a fragment smiles.
    Fragment is expected to have a dummy '*' indicating attachment point.
    Returns canonical leaf smiles (no dummy).
    """
    mol_branch = Chem.MolFromSmiles(branch_smiles_with_dummy)
    
    # Process fragment: add H then deuterate (though this function name implies branch prep?)
    # Wait, the requirement says "fragmentについては、先にcombine_smiles関数内でsmiles_addH_then_deuterateを処理する"
    # Actually, usually we deuterate the *part* that we want to expand NEXT.
    # But let's follow given example/requirement logic.
    # The requirement 6.1 says: "fragmentについては、先にcombine_smiles関数内でsmiles_addH_then_deuterateを処理する。この関数はFragmentに存在するH原子を重水素に変換する。"
    # This implies the fragment adds capability to be further extended?
    # NO. Leaf is evaluated. Branch generation happens later via hydrogen_replace.
    # If we deuterate the fragment NOW, it becomes part of the Leaf.
    # Then hydrogen_replace on Leaf will turn those D's into * for next steps.
    # YES. That seems to be the flow.
    
    frag_smiles_d = smiles_addH_then_deuterate(frag_smiles)
    mol_frag = Chem.MolFromSmiles(frag_smiles_d)
    
    if mol_branch is None:
        raise ValueError(f"Invalid branch SMILES: {branch_smiles_with_dummy}")
    if mol_frag is None:
        raise ValueError(f"Invalid fragment SMILES: {frag_smiles_d}")

    def find_dummy_neighbor(mol):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0: # Dummy *
                neighbors = atom.GetNeighbors()
                if neighbors:
                    return atom.GetIdx(), neighbors[0].GetIdx()
        return None, None

    pad_b_dummy, pad_b_neighbor = find_dummy_neighbor(mol_branch)
    if pad_b_dummy is None:
        raise ValueError("Branch SMILES must have a dummy atom (*)")
        
    pad_f_dummy, pad_f_neighbor = find_dummy_neighbor(mol_frag)
    if pad_f_dummy is None:
        raise ValueError("Fragment SMILES must have a dummy atom (*)")
        
    combined = Chem.CombineMols(mol_branch, mol_frag)
    rwmol = Chem.RWMol(combined)
    
    num_b_atoms = mol_branch.GetNumAtoms()
    
    idx_b_dummy = pad_b_dummy
    idx_b_att = pad_b_neighbor
    
    idx_f_dummy = pad_f_dummy + num_b_atoms
    idx_f_att = pad_f_neighbor + num_b_atoms
    
    rwmol.AddBond(idx_b_att, idx_f_att, order=Chem.rdchem.BondType.SINGLE)
    
    if idx_f_dummy > idx_b_dummy:
        rwmol.RemoveAtom(idx_f_dummy)
        rwmol.RemoveAtom(idx_b_dummy)
    else:
        rwmol.RemoveAtom(idx_b_dummy)
        rwmol.RemoveAtom(idx_f_dummy)
        
    try:
        Chem.SanitizeMol(rwmol)
        return Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True)
    except Exception as e:
        raise ValueError(f"Failed to combine smiles: {e}")

def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
