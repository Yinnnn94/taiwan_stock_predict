from keras.models import load_model
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

lstm = load_model("ancestor_model.h5", compile = False)
lstm_ga = load_model("child_model.h5" , compile = False)

test = r'C:\Users\user\OneDrive\文件\Python Scripts\taiwan_stock_predict\test_data.csv'


data = pd.read_csv(test)

data = data["開市"]
data = data.values
data = data.reshape(-1, 1)
sc = MinMaxScaler(feature_range=(0, 1)) # Feature Scaling
test_set_scaled = sc.fit_transform(data)

# 創建X_train和y_train
X_test = []
y_test = []
for i in range(5, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-5:i, :])  # 使用五個column的資料
    y_test.append(test_set_scaled[i, :])      # 預測下一筆五個column的資料

X_test, y_test = np.array(X_test), np.array(y_test)
# Reshape 成3-dimension
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_ancestor = lstm.predict(X_test)
pred_child = lstm_ga.predict(X_test)

# 轉回原始scale
y_test_inverse = sc.inverse_transform(y_test)
pred_ancestor_inverse = sc.inverse_transform(pred_ancestor)
pred_child_inverse = sc.inverse_transform(pred_child)

plt.plot(y_test_inverse, color = 'red', label = 'Real Stock Price')
plt.plot(pred_ancestor_inverse, color = 'blue', label = 'Predicted Stock Price By lstm')
plt.plot(pred_child_inverse, color = 'green', label = 'Predicted Stock Price By lstm_ga')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()