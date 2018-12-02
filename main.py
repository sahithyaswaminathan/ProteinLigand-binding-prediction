import numpy as np

def main(X_pro, Y_pro, Z_pro, atomtype_pro, X_lig, Y_lig, Z_lig, atomtype_lig):

    ''' This function contains pre-processing of the data.
    The X_pro, Y_pro and Z_pro are subjected to a cube along with the ligand and then preporocessed to preserve the structure of the pro-lig
     Preprocessing steps:
        1. Normalize the protein and ligand
            protein = (protein - min )/ (max-min)
            ligand = (ligand - min) / (max-min)
        2. state the number of bins for the cube and the dimension of the cube
            A cube of 1,1,1 with bins =20 is considered
        3. Find the placements of the protein and ligand inside the cube
        4. Find the count of number of atoms in each cell.
        5. If the number of polar is more = no pf polar * +1
            elif the number of non polar is more = no of non polar * -1
            else number of atoms *0'''
    protein_list = []
    for i in range(len(atomtype_pro)):
        protein_list.append((X_pro[i], Y_pro[i], Z_pro[i]))

    ligand_list = []
    for i in range(len(atomtype_lig)):
        ligand_list.append((X_lig[i], Y_lig[i], Z_lig[i]))

    pro_lig_list = protein_list + ligand_list
    # Finding the min and max: for normalization
    X_pro_lig = X_pro + X_lig
    Y_pro_lig = Y_pro + Y_lig
    Z_pro_lig = Z_pro + Z_lig
    atomtype = atomtype_pro + atomtype_lig

    min_X = min(X_pro_lig)
    min_Y = min(Y_pro_lig)
    min_Z = min(Z_pro_lig)

    max_X = max(X_pro_lig)
    max_Y = max(Y_pro_lig)
    max_Z = max(Z_pro_lig)
    # protein ligand normalization
    X_pro_lig_min_max = []
    Y_pro_lig_min_max = []
    Z_pro_lig_min_max = []
    for i in range(len(pro_lig_list)):
        X_pro_lig_min_max.append((X_pro_lig[i] - min_X) / (max_X - min_X))
        Y_pro_lig_min_max.append((Y_pro_lig[i] - min_Y) / (max_Y - min_Y))
        Z_pro_lig_min_max.append((Z_pro_lig[i] - min_Z) / (max_Z - min_Z))
    # Find the bin count
    bin_dict = {}
    i = 0
    ss = np.linspace(0, 1, 11)
    while i < len(ss) - 1:
        bin_dict[i] = [round(ss[i], 2), round(ss[i + 1], 2)]
        i += 1
    # placment of cells in each bins
    X_placement = []
    Y_placement = []
    Z_placement = []
    pro_lig_placement = []
    for i in range(len(pro_lig_list)):
        for key, value in bin_dict.items():
            if X_pro_lig_min_max[i] >= float(value[0]) and X_pro_lig_min_max[i] <= float(value[1]):
                X_placement.append(key)
            if Y_pro_lig_min_max[i] >= float(value[0]) and Y_pro_lig_min_max[i] <= float(value[1]):
                Y_placement.append(key)
            if Z_pro_lig_min_max[i] >= float(value[0]) and Z_pro_lig_min_max[i] <= float(value[1]):
                Z_placement.append(key)
        pro_lig_placement.append((X_placement[i], Y_placement[i], Z_placement[i]))

    pro_lig_dict = {}
    index = {}

    # count the number of atoms:
    for pro_lig_index in range(len(pro_lig_placement)):
        i = pro_lig_placement[pro_lig_index]
        if i in pro_lig_dict:
            pro_lig_dict[i] = pro_lig_dict[i] + 1
        else:
            pro_lig_dict[i] = 1

        if i in index:
            (index[i]).append(pro_lig_index)
        else:
            _dict = []
            _dict.append(pro_lig_index)
            index[i] = _dict

    matrix_pro_lig = np.zeros((10, 10, 10))
    for key, value in pro_lig_dict.items():
        matrix_pro_lig[key[0]][key[1]][key[2]] = value
    # give values to each bins of the cube
    atomIndexMap = {}
    for key, value in index.items():
        keytuple = key
        indexList = value
        h_count = 0
        p_count = 0
        for indx in indexList:
            if atomtype[indx] == 'h':
                h_count = h_count + 1
            elif atomtype[indx] == 'p':
                p_count = p_count + 1
        if h_count > p_count:
            atomIndexMap[keytuple] = -1
        elif p_count > h_count:
            atomIndexMap[keytuple] = 1
        else:
            atomIndexMap[keytuple] = 0

    for key, value in atomIndexMap.items():
        matrix_pro_lig[key[0]][key[1]][key[2]] = matrix_pro_lig[key[0]][key[1]][key[2]] * value
    return matrix_pro_lig