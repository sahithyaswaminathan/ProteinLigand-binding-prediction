
from keras.models import Sequential
from keras.layers import Dense,  Dropout, Flatten,BatchNormalization
from keras import optimizers
from keras.layers import Conv3D, MaxPooling3D
from matplotlib import pyplot as plt

from keras import backend as K
from keras.utils import to_categorical
from protein_ligand_train import protein_ligand

Xtrain, Ytrain, Xtest, Ytest = protein_ligand()

y_train = to_categorical(Ytrain, num_classes = 2)
y_test = to_categorical(Ytest, num_classes=2)

num_classes = 2

if K.image_data_format() == 'channels_first':
    x_train1 = Xtrain.reshape(Xtrain.shape[0], 1, 20, 20, 20)
    x_test1  = Xtest.reshape(Xtest.shape[0]  , 1, 20, 20, 20)
    input_shape = (1, 20, 20, 20)
else:
    x_train1 = Xtrain.reshape(Xtrain.shape[0], 20, 20, 20, 1)
    x_test1 = Xtest.reshape(Xtest.shape[0], 20, 20, 20, 1)
    input_shape = (20, 20, 20, 1)

model = Sequential()

# 1st layer group
model.add(Conv3D(16, kernel_size=(5, 5, 5), activation='relu', padding='same',
                 input_shape=input_shape, name='conv1_1'))
model.add(Dropout(0.25))

# 2nd layer group
model.add(Conv3D(32, kernel_size=(5, 5, 5), activation='relu', padding='same',
                 input_shape=input_shape, name='conv1_2'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool1_2'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# FC Layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

#output layer
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

batch_size = 32
num_classes = 2
epochs = 50
learning_rate = 0.1

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


model.summary()


history = model.fit(x_train1, y_train, batch_size=batch_size,
          epochs=epochs, verbose=1, validation_split=0.2)

model.save('weights.hdf5')


print("=======LOADED MODEL FROM WEIGHT.BEST FILE============")
from keras.models import load_model

model_test = load_model("weights.hdf5")
adm = optimizers.Adam(lr=0.01)
model_test.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])

pred_class = model_test.predict_classes(Xtest)

print("===========CALCULATING F1 SCORE============")
from sklearn import metrics

Test_f1= metrics.f1_score(Ytest, pred_class, average=None)
prec, rec, fbeta_test, support = metrics.precision_recall_fscore_support(Ytest, pred_class, average = None)
metrics.precision_recall_fscore_support(Ytest, pred_class, average = None)

print ("Test Evaluation: ")
print ("Average F1 Score: {:4f}".format((fbeta_test[0]+fbeta_test[1])/2))
print("F1 score of Class-0: {:4f}".format(fbeta_test[0]))
print("F1 score of Class-1: {:4f}".format(fbeta_test[1]))
print ("Average Precision Score: {:4f}".format((prec[0]+prec[1])/2))
print("Average Sensitivity/TPR Score: {:4f}".format((rec[0]+rec[1])/2))

#AUC AND ROC CURVE
print("===========PLOTTING THE ROC CURVE===========")

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(pred_class, Ytest)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()