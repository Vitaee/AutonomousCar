# 1-) path
# 2-) data balance
# 3-) load data
# 4-)  train test split
# 5-) get model
# 6-) fit
# 7-) model save
from utils import *
# path = 'datas'
path = '/home/vitae/projects/beta_simulator_linux/'
data = importDataInfo(path)

# balance data
data = balanceData(data, True)
images, steering = loadData(data, path)

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(images, steering, test_size=0.2, random_state=5)
model = getModel()
model.summary()

model.fit( batchGen(x_train, y_train, 75, 1),
         steps_per_epoch=375, epochs=15,
         validation_data=batchGen(x_test, y_test, 75, 0),
         validation_steps=375,
         verbose=1)


model.save('bestmodel.h5')