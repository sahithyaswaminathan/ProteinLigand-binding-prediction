import numpy as np
import random
from read_pdb_train import read_pdb
from main import main
''' This file contains the call for main files and the read_pdb files to read the protein and ligand'''
def protein_ligand():
    matrices = []
    y_vector = []
    for i in range(2699):
        X_pro, Y_pro, Z_pro, atomtype_pro = read_pdb(
            "/content/drive/My Drive/app/training_data/" + str(i + 1).zfill(4) + "_pro_cg.pdb")
        X_lig, Y_lig, Z_lig, atomtype_lig = read_pdb(
            "/content/drive/My Drive/app/training_data/" + str(i + 1).zfill(4) + "_lig_cg.pdb")
        matrix = main(X_pro, Y_pro, Z_pro, atomtype_pro, X_lig, Y_lig, Z_lig, atomtype_lig)
        # print(i)
        matrices.append(matrix)
        y_vector.append(1)

    num = []
    for i in range(2):
        a = random.randint(1, 3000)
        num.append(a)

    matrices_1 = []
    y_vector_1 = []
    for j in range(len(num)):
        for i in range(2699):
            if i != num[j]:
                X_pro, Y_pro, Z_pro, atomtype_pro = read_pdb(
                    "/content/drive/My Drive/app/training_data/" + str(i + 1).zfill(4) + "_pro_cg.pdb")
                X_lig, Y_lig, Z_lig, atomtype_lig = read_pdb(
                    "/content/drive/My Drive/app/training_data/" + str(num[j]).zfill(4) + "_lig_cg.pdb")
                #      print(i)
                matrix_pro_lig = main(X_pro, Y_pro, Z_pro, atomtype_pro, X_lig, Y_lig, Z_lig, atomtype_lig)

                matrices_1.append(matrix_pro_lig)
                y_vector_1.append(0)

    new_matrix = matrices + matrices_1
    matrix_output = np.concatenate([arr[np.newaxis] for arr in new_matrix])
    new_y = y_vector + y_vector_1
    return matrix_output, new_y, num
