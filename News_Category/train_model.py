import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report

# 可调整参数
num_word = 30000
embedding_dim = 100
max_sequence_length = 100
data_path = "data/processed_data.json"


# 读取数据集
data = pd.read_json(data_path, lines=True)

# 1.对文本进行处理
print("Step1:Start processing...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)
# 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['category'])
y_test = label_encoder.transform(test_data['category'])
# 使用 Tokenizer 对文本进行处理
tokenizer = Tokenizer(num_words=num_word, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['processed_text'])
# 对文本序列进行填充
X_train_seq = tokenizer.texts_to_sequences(train_data['processed_text'])
X_test_seq = tokenizer.texts_to_sequences(test_data['processed_text'])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')
print("Processing finish.")


# 2.构建卷积神经网络模型
print("Step2:Start building...")
model = Sequential()
model.add(Embedding(input_dim=num_word, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(42, activation='softmax'))
# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 设置早停
# tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("Building finish.")

# 3.训练模型
print("Step3:Start training...")
model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])
print("Training finish.")

# 4.评估模型
print("Step4:Start testing...")
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Testing finish.")

# 5.保存结果模型和经处理数据
print("Step5:Saving...")
joblib.dump(label_encoder,'data/label_encoder.joblib')
joblib.dump(tokenizer,'data/tokenizer.joblib')
model.save("data/news_category_model.h5")
print("Saving finish.")