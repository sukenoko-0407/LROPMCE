from rdkit import Chem
from rdkit.Chem import AllChem

from typing import List

def hydrogen_replace(smiles_with_D: str, *, canonical: bool = True) -> List[str]:
    """
    入力SMILES中の各重水素([2H])を「1個だけ」ダミーアトム(*)に置換したSMILESを全列挙し、
    重複を除いたリストを返す。

    - [2H] が無い場合は [] を返す
    - 対称性などにより、Dが複数あってもユニークSMILES数がD数より少ない場合がある
    """
    mol = Chem.MolFromSmiles(smiles_with_D)
    if mol is None:
        raise ValueError(f"SMILESのパースに失敗しました: {smiles_with_D}")

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

        # [2H] -> *（原子番号0がダミーアトム）
        a.SetAtomicNum(0)
        a.SetIsotope(0)
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)

        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)  # ダミー原子を含んだままでも基本OK

        s2 = Chem.MolToSmiles(m2, isomericSmiles=True, canonical=canonical)
        out.add(s2)

    return sorted(out)


# 例:
# smiles = "CC([2H])([2H])O"
# print(deuterium_sites_to_dummy_smiles(smiles))


def combine_smiles(branch_smiles_with_dummy: str, frag_smiles: str) -> str:
    """
    Combines a branch smiles (with dummy '*') and a fragment smiles.
    Fragment is expected to have a dummy '*' indicating attachment point.
    Returns canonical leaf smiles (no dummy).
    """
    mol_branch = Chem.MolFromSmiles(branch_smiles_with_dummy)
    frag_smiles = smiles_addH_then_deuterate(frag_smiles)
    mol_frag = Chem.MolFromSmiles(frag_smiles)
    
    if mol_branch is None:
        raise ValueError(f"Invalid branch SMILES: {branch_smiles_with_dummy}")
    if mol_frag is None:
        raise ValueError(f"Invalid fragment SMILES: {frag_smiles}")

    # Helper to find dummy and its neighbor
    def find_dummy_neighbor(mol):
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0: # Dummy *
                neighbors = atom.GetNeighbors()
                if neighbors:
                    return atom.GetIdx(), neighbors[0].GetIdx()
        return None, None

    # Find attachment points
    pad_b_dummy, pad_b_neighbor = find_dummy_neighbor(mol_branch)
    if pad_b_dummy is None:
        raise ValueError("Branch SMILES must have a dummy atom (*)")
        
    pad_f_dummy, pad_f_neighbor = find_dummy_neighbor(mol_frag)
    if pad_f_dummy is None:
        # If fragment has no dummy, maybe it's just an atom? 
        # But library has *C. If input is 'C', we can't strict attach?
        # Requirement says fragment replaces dummy.
        # If no dummy in frag, we assume we attach it?
        # But let's enforce * for now as per library format.
        raise ValueError("Fragment SMILES must have a dummy atom (*)")
        
    # Combine Mols
    # Note: CombineMols returns a fixed molecule. We need RW for editing.
    combined = Chem.CombineMols(mol_branch, mol_frag)
    rwmol = Chem.RWMol(combined)
    
    # Indices in combined mol:
    # Branch atoms are 0..N-1
    # Frag atoms are N..N+M-1
    # We need to shift frag indices
    num_b_atoms = mol_branch.GetNumAtoms()
    
    # Branch Indices
    idx_b_dummy = pad_b_dummy
    idx_b_att = pad_b_neighbor
    
    # Frag Indices (shifted)
    idx_f_dummy = pad_f_dummy + num_b_atoms
    idx_f_att = pad_f_neighbor + num_b_atoms
    
    # Add Bond between attachment points
    # Get bond type from one of the dummy bonds?
    # Usually single.
    rwmol.AddBond(idx_b_att, idx_f_att, order=Chem.rdchem.BondType.SINGLE)
    
    # Remove Dummies
    # Remove highest index first to avoid shifting problems
    if idx_f_dummy > idx_b_dummy:
        rwmol.RemoveAtom(idx_f_dummy)
        rwmol.RemoveAtom(idx_b_dummy)
    else:
        rwmol.RemoveAtom(idx_b_dummy)
        rwmol.RemoveAtom(idx_f_dummy)
        
    # Sanitize and Return
    try:
        Chem.SanitizeMol(rwmol)
        return Chem.MolToSmiles(rwmol, canonical=True, isomericSmiles=True)
    except Exception as e:
        raise ValueError(f"Failed to combine smiles: {e}")


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
        raise ValueError(f"SMILESのパースに失敗しました: {smiles_with_dummy}")

    n0 = mol.GetNumAtoms()              # 付加前の原子数
    molH = Chem.AddHs(mol)              # 明示的Hを付加

    # 付加されたH(=新規に増えた原子)だけを [2H] にする
    for idx in range(n0, molH.GetNumAtoms()):
        atom = molH.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:    # H
            atom.SetIsotope(2)          # D として表現される

    return Chem.MolToSmiles(molH, isomericSmiles=True, canonical=canonical)


# 例:
# print(smiles_addH_then_deuterate("c1cc(*)ccc1"))



def hydrogen_replace_old(leaf_smiles_no_dummy: str) -> list[str]:
    """
    Generates branch candidates by replacing a Hydrogen with a Dummy atom (*).
    Returns list of canonical SMILES (each has exactly 1 dummy).
    """
    mol = Chem.MolFromSmiles(leaf_smiles_no_dummy)
    if mol is None:
        return []
    
    # Add explicit hydrogens so we can replace them
    mol = Chem.AddHs(mol)
    
    candidates = []
    
    # Iterate over all atoms, check if H, try to replace with *
    # Actually, usually we iterate over heavy atoms and replace one of their implicit/explicit Hs.
    # But since we added Hs, they are now atoms.
    
    # Strategy:
    # 1. Identify H atoms.
    # 2. For each H atom, create a copy of Mol, replace H with *.
    # 3. Sanitize, Canonicalize.
    # 4. Remove duplicates.
    
    # Improvement: Identify unique positions first using symmetry?
    # For now, brute force and set to unique.
    
    indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'H']
    
    seen = set()
    results = []
    
    for idx_to_replace in indices:
        # Clone
        rwmol = Chem.RWMol(mol)
        
        # Replace H with * (atomic num 0)
        rwmol.GetAtomWithIdx(idx_to_replace).SetAtomicNum(0)
        # rwmol.GetAtomWithIdx(idx_to_replace).SetIsotope(0) # ensure it's *
        
        # We need to remove other Hs to get back to implicit H representation for SMILES
        # But simply calling RemoveHs might remove our * if we are not careful?
        # No, RemoveHs removes H. * is not H.
        
        try:
            # Convert back to Mol
            new_mol = rwmol.GetMol()
            Chem.SanitizeMol(new_mol)
            # Remove remaining hydrogens
            new_mol = Chem.RemoveHs(new_mol)
            
            smi = Chem.MolToSmiles(new_mol, canonical=True, isomericSmiles=True)
            
            if smi not in seen:
                seen.add(smi)
                results.append(smi)
        except Exception:
            continue
            
    return results

def canonicalize(smiles: str) -> str:
    """Canonicalize a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
