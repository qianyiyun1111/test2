import pandas as pd
import numpy as np
import joblib
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# 可调整参数
data_path = "data/processed_test_data.json"

# 加载模型
model = load_model('data/news_category_model.h5')
# 读取数据集
test_data = pd.read_json(data_path, lines=True)
label_encoder = joblib.load('data/label_encoder.joblib')
tokenizer = joblib.load('data/tokenizer.joblib')

y_test = label_encoder.transform(test_data['category'])
X_test_seq = tokenizer.texts_to_sequences(test_data['processed_text'])
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post', truncating='post')

# 评估模型
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)