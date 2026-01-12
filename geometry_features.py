import math
import numpy as np
import pandas as pd

# ---------- geometry utilities ----------
def dist(a,b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def angle(a,b,c):
    ba = a-b
    bc = c-b
    return np.degrees(
        np.arccos(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)))
    )

def midpoint(a,b):
    return (a+b)/2

def dihedral(p0,p1,p2,p3):
    b0 = p0-p1
    b1 = p2-p1
    b2 = p3-p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0,b1)*b1
    w = b2 - np.dot(b2,b1)*b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1,v),w), np.dot(v,w)))

# ---------- constants ----------
lanthanides = set(range(57,72))
transition_metals = set(range(21,31)) | set(range(39,49)) | set(range(72,81))

spin_map = {
    23:1.5, 24:1.5, 25:2.5, 26:2.0,
    27:1.5, 28:1.0, 29:0.5
}

# ---------- feature extractor ----------
def extract_features(xyz_file):

    atoms = []
    with open(xyz_file) as f:
        for i,line in enumerate(f):
            Z,x,y,z = line.split()
            atoms.append((i+1,int(Z),np.array([float(x),float(y),float(z)])))

    Ln = [a for a in atoms if a[1] in lanthanides][0]
    Tm = [a for a in atoms if a[1] in transition_metals][0]
    O_atoms = [a for a in atoms if a[1]==8]

    # two closest oxygens
    oxy = sorted(
        [(dist(Ln[2],o[2])+dist(Tm[2],o[2]),o) for o in O_atoms]
    )[:2]

    O1, O2 = oxy[0][1][2], oxy[1][1][2]

    def nearest_heavy(Ocoord):
        cand=[]
        for _,Z,c in atoms:
            if Z not in lanthanides and Z not in transition_metals and Z!=8:
                cand.append((dist(Ocoord,c),c))
        return min(cand)[1]

    C1 = nearest_heavy(O1)
    C2 = nearest_heavy(O2)

    LnTm = dist(Ln[2],Tm[2])
    LnO = sorted([dist(Ln[2],O1),dist(Ln[2],O2)])
    TmO = sorted([dist(Tm[2],O1),dist(Tm[2],O2)])
    LnOTm = sorted([angle(Ln[2],O1,Tm[2]),angle(Ln[2],O2,Tm[2])])

    X = midpoint(O1,O2)
    LnXTm = angle(Ln[2],X,Tm[2])
    torsion = abs(dihedral(C1,O1,O2,C2))

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
