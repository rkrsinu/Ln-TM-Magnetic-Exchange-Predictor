import math
import numpy as np
import pandas as pd


# ---------------- Geometry utilities ----------------
def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def angle(a, b, c):
    ba = a - b
    bc = c - b
    return np.degrees(
        np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    )


def midpoint(a, b):
    return (a + b) / 2


def dihedral(p0, p1, p2, p3):

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    return np.degrees(
        np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))
    )


# ---------------- Periodic table ----------------
symbol_to_Z = {
'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,
'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,
'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80
}


# ---------------- Constants ----------------
lanthanides = set(range(57, 72))

transition_metals = (
    set(range(21, 31)) |
    set(range(39, 49)) |
    set(range(72, 81))
)

spin_map = {
    23: 1.5,
    24: 2.0,
    25: 2.5,
    26: 2.0,
    27: 1.5,
    28: 1.0,
    29: 0.5,
    30: 0.0
}


# ---------------- XYZ reader ----------------
def read_xyz(xyz_file):

    atoms = []

    with open(xyz_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    # detect standard XYZ header
    start = 0
    try:
        int(lines[0])
        start = 2
    except:
        start = 0

    idx = 1

    for line in lines[start:]:

        parts = line.split()

        if len(parts) < 4:
            continue

        atom = parts[0]

        # atomic number
        if atom.isdigit():
            Z = int(atom)

        # atomic symbol
        else:
            atom = atom.capitalize()

            if atom not in symbol_to_Z:
                raise ValueError(f"Unknown element symbol: {atom}")

            Z = symbol_to_Z[atom]

        x, y, z = map(float, parts[1:4])

        atoms.append((idx, Z, np.array([x, y, z])))

        idx += 1

    return atoms


# ---------------- Feature extraction ----------------
def extract_features(xyz_file, ln_index=None, tm_index=None):

    atoms = read_xyz(xyz_file)

    # Identify Ln and TM atoms
    Ln_atoms = [a for a in atoms if a[1] in lanthanides]
    Tm_atoms = [a for a in atoms if a[1] in transition_metals]

    if len(Ln_atoms) == 0:
        raise ValueError("No lanthanide atom found in the structure.")

    if len(Tm_atoms) == 0:
        raise ValueError("No transition metal atom found in the structure.")

    if len(Ln_atoms) > 1 and ln_index is None:
        raise ValueError(
            f"Multiple lanthanides detected (indices: {[a[0] for a in Ln_atoms]}). "
            "Please specify the Ln atom index."
        )

    if len(Tm_atoms) > 1 and tm_index is None:
        raise ValueError(
            f"Multiple transition metals detected (indices: {[a[0] for a in Tm_atoms]}). "
            "Please specify the TM atom index."
        )

    Ln = next(a for a in Ln_atoms if a[0] == ln_index) if ln_index else Ln_atoms[0]
    Tm = next(a for a in Tm_atoms if a[0] == tm_index) if tm_index else Tm_atoms[0]

    # Zn safeguard
    if Tm[1] == 30:
        raise ValueError(
            "Zn(II) is diamagnetic (S = 0). No magnetic exchange coupling expected."
        )

    # ---------------- Oxygen bridge detection ----------------
    O_atoms = [a for a in atoms if a[1] == 8]

    if len(O_atoms) < 2:
        raise ValueError(
            "No oxygen bridge detected. This model applies only to Ln–O–Tm bridged systems."
        )

    bridging = []

    for o in O_atoms:

        d_Ln = dist(Ln[2], o[2])
        d_Tm = dist(Tm[2], o[2])

        if d_Ln < 4.0 and d_Tm < 3.0:
            bridging.append((d_Ln + d_Tm, o))

    if len(bridging) < 2:
        raise ValueError(
            "Structure does not contain two bridging oxygen atoms forming a Ln–O–Tm pathway."
        )

    bridging = sorted(bridging)[:2]

    O1 = bridging[0][1][2]
    O2 = bridging[1][1][2]

    # Ln–O sanity check
    LnO1 = dist(Ln[2], O1)
    LnO2 = dist(Ln[2], O2)

    if LnO1 > 4.0 or LnO2 > 4.0:
        raise ValueError(
            f"Unphysical structure detected: Ln–O distance ({max(LnO1, LnO2):.2f} Å) > 4 Å."
        )

    # ---------------- Find heavy atoms for torsion ----------------
    def nearest_heavy(Ocoord):

        candidates = []

        for _, Z, c in atoms:

            if Z not in lanthanides and Z not in transition_metals and Z != 8:

                candidates.append((dist(Ocoord, c), c))

        if not candidates:
            raise ValueError("No suitable atoms found for torsion calculation.")

        return min(candidates)[1]

    C1 = nearest_heavy(O1)
    C2 = nearest_heavy(O2)

    # ---------------- Geometry calculations ----------------
    LnTm = dist(Ln[2], Tm[2])

    LnO = sorted([dist(Ln[2], O1), dist(Ln[2], O2)])
    TmO = sorted([dist(Tm[2], O1), dist(Tm[2], O2)])

    LnOTm = sorted([
        angle(Ln[2], O1, Tm[2]),
        angle(Ln[2], O2, Tm[2])
    ])

    X = midpoint(O1, O2)

    LnXTm = angle(Ln[2], X, Tm[2])

    torsion = abs(dihedral(C1, O1, O2, C2))

    # ---------------- Feature vector ----------------
    return pd.DataFrame([{
        "Spin": spin_map[Tm[1]],
        "TmZ": Tm[1],
        "Ln-Tm": LnTm,
        "(Ln-O-Tm)1": LnOTm[0],
        "(Ln-O-Tm)2": LnOTm[1],
        "(Ln-X-Tm)": LnXTm,
        "tio": torsion,
        "Ln-O1": LnO[0],
        "Ln-O2": LnO[1],
        "Tm-O1": TmO[0],
        "Tm-O2": TmO[1]
    }])
