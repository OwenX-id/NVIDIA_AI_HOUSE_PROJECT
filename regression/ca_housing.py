import os

import numpy as np
import pandas as pd

import cv2

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

df = pd.read_csv('/kaggle/input/house-prices-and-images-socal/socal2.csv')
df.head()
df['street'] = df['street'].str.split(' ', 1).str[1]
df['street'].nunique()
df['street'].value_counts()

street_frequency = df['street'].value_counts().to_dict()

df = df.replace({'street': street_frequency})
df['citi'].nunique()
df['citi'].value_counts()
df['citi'].value_counts()

# city_frequency = df['citi'].value_counts().to_dict()

# df = df.replace({'citi': city_frequency})

df = pd.get_dummies(df, columns=['citi'])

# df = df.drop('citi', axis=1)
df = df.drop('n_citi', axis=1)
# df = df.drop('street', axis=1)

df['price'] = df.pop('price')

df.head()

df_features = df.iloc[:, :-1]
df_labels = df.iloc[:, [-1]]

tab_features = df_features.to_numpy()
labels = df_labels.to_numpy()

img_dir = '/kaggle/input/house-prices-and-images-socal/socal2/socal_pics'

img_height = 128
img_width = 128

files = os.listdir(img_dir)
files.sort(key=lambda x: int(x.split('.')[0]))

img_list = []

for file in files:
    img = cv2.imread(os.path.join(img_dir, file))
    img = cv2.resize(img, (img_height, img_width))
    img_list.append(img)

img_features = np.asarray(img_list)

rng = np.random.default_rng(seed=13)

size = len(tab_features)

p = rng.permutation(size)

tab_features = tab_features[p]
img_features = img_features[p]
labels = labels[p]

val_idx = int(size * 0.6)
test_idx = int(size * 0.8)

X_tab_train, X_tab_val, X_tab_test = np.split(tab_features, [val_idx, test_idx])
X_img_train, X_img_val, X_img_test = np.split(img_features, [val_idx, test_idx])
y_train, y_val, y_test = np.split(labels, [val_idx, test_idx])

input_tab = Input(X_tab_train.shape[1])
x_tab = Dense(1024, activation='relu')(input_tab)
x_tab = Dense(1024, activation='relu')(x_tab)

model_tab = Model(input_tab, x_tab)

input_img = Input((img_height, img_width, 3))
x_img = Rescaling(1./255)(input_img)
x_img = Conv2D(64, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(64, 3, padding='same', activation='relu')(x_img)
x_img = MaxPooling2D(padding='same')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Conv2D(128, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(128, 3, padding='same', activation='relu')(x_img)
x_img = MaxPooling2D(padding='same')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Conv2D(256, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(256, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(256, 3, padding='same', activation='relu')(x_img)
x_img = MaxPooling2D(padding='same')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Conv2D(512, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(512, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(512, 3, padding='same', activation='relu')(x_img)
x_img = MaxPooling2D(padding='same')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Conv2D(1024, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(1024, 3, padding='same', activation='relu')(x_img)
x_img = Conv2D(1024, 3, padding='same', activation='relu')(x_img)
x_img = MaxPooling2D(padding='same')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Dropout(0.25)(x_img)
x_img = Flatten()(x_img)
x_img = Dense(1024, activation='relu')(x_img)
x_img = Dense(1024, activation='relu')(x_img)

model_img = Model(input_img, x_img)

input = Concatenate()([model_tab.output, model_img.output])
x = Dense(4096, activation='relu')(input)
x = Dropout(0.1)(x)
x = Dense(4096, activation='relu')(x)
x = Dense(1, activation='linear')(x)

model = Model([model_tab.input, model_img.input], x)

model.compile(optimizer='adam', loss='mse',metrics=['mae', 'mape'])

model.summary()

epochs = 100
batch_size = 64

early_stopping = EarlyStopping(patience=20, verbose=2, restore_best_weights=True)

history = model.fit(
    [X_tab_train, X_img_train],
    y_train,
    validation_data=([X_tab_val, X_img_val], y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=early_stopping
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mean absolute error')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('model mean absolute percentage error')
plt.ylabel('mape')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_pred_nn = model.predict([X_tab_test, X_img_test])

metrics_nn = model.evaluate([X_tab_test, X_img_test], y_test)
metrics_nn.append(r2_score(y_test, y_pred_nn))

def scatterplot_true_and_pred(y_true, y_pred, title) -> None:

    ax = sns.scatterplot(x=y_true.reshape(-1), y=y_pred.reshape(-1))
    ax.set_title(title)
    ax.set(xlabel='True Value', ylabel='Predicted Value')

    max_elem = max([y_true.max(), y_pred.max()])
    diag_values = np.linspace(0, max_elem, 1000)
    plt.plot(diag_values, diag_values, color='magenta', linestyle='--')
    plt.show()
scatterplot_true_and_pred(y_true=y_test, y_pred=y_pred_nn, title='Performance of NN')

xgbr = XGBRegressor(n_estimators=500,
                    early_stopping_rounds=10)
_ = xgbr.fit(X_tab_train,
             y_train,
             eval_set=[(X_tab_val, y_val)])

y_pred_xgb = xgbr.predict(X_tab_test)
metrics_xgb = []
metrics_xgb.append(mean_squared_error(y_test, y_pred_xgb))
metrics_xgb.append(mean_absolute_error(y_test, y_pred_xgb))
metrics_xgb.append(mean_absolute_percentage_error(y_test, y_pred_xgb))
metrics_xgb.append(r2_score(y_test, y_pred_xgb))
scatterplot_true_and_pred(y_true=y_test, y_pred=y_pred_xgb, title='Performance of XGBoost')

print('\tNN\t\t\tXGBoost')
print(f'MSE:\t{metrics_nn[0]}\t\t{metrics_xgb[0]}')
print(f'MAE:\t{metrics_nn[1]}\t\t{metrics_xgb[1]}')
print(f'MAPE:\t{metrics_nn[2]}\t{metrics_xgb[2]}')
print(f'R2:\t{metrics_nn[3]}\t{metrics_xgb[3]}')

