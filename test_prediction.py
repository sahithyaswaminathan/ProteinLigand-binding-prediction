# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:16:24 2018

@author: ssahi
"""

# Data Preprocessing - Testdata

from read_pdb_test import read_pdb
from main import main
import numpy as np
import pandas as pd

top_10 = pd.DataFrame(
    columns=['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id', 'lig8_id',
             'lig9_id', 'lig10_id'])

from keras.models import load_model

model_test = load_model("weights.h5")
# adm = optimizers.Adam(lr=0.1)
model_test.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

for pro in range(1, 825):
    matrices = []
    for i in range(824):
        # data preporcoessing
        X_pro, Y_pro, Z_pro, atomtype_pro = read_pdb(
            "/testing_data_release/testing_data/" + str(
                pro).zfill(4) + "_pro_cg.pdb")
        X_lig, Y_lig, Z_lig, atomtype_lig = read_pdb(
            "/testing_data_release/testing_data/" + str(
                i + 1).zfill(4) + "_lig_cg.pdb")
        matrix = main(X_pro, Y_pro, Z_pro, atomtype_pro, X_lig, Y_lig, Z_lig, atomtype_lig)
        # print(i)
        matrices.append(matrix)
        # y_vector.append(1)

    new_matrix = matrices
    matrix_output = np.concatenate([arr[np.newaxis] for arr in new_matrix])

    X_pro_1 = matrix_output

    X_pro_1 = X_pro_1.reshape((-1, 20, 20, 20, 1))

    listy = []

    print("=======LOADED MODEL FROM WEIGHT.BEST FILE============")

    pred = model_test.predict(X_pro_1)
    # Copy column-1 in list
    class1 = []
    for j in range(len(pred)):
        class1.append(pred[j, 1])

    sortlist = sorted(((e, i) for i, e in enumerate(class1)), reverse=True)

    pro_name = pro
    # get 10 values
    result = []
    result.append(pro_name)
    for val in range(10):
        result.append((sortlist[val][1] + 1))

    _result = pd.Series(result,
                        index=['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id',
                               'lig8_id', 'lig9_id', 'lig10_id'])
    top_10 = top_10.append([_result], ignore_index=True)

np.savetxt('test_predictions.txt', top_10.values, fmt='%d', delimiter="\t",
           header="pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id")