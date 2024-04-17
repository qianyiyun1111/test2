import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

# 可调整参数
input_path = "data/processed_data.json"
output_path = "data/output.json"

# Load the saved model
model = load_model('data/news_category_model.h5')
# Read the JSON file
input_data = pd.read_json(input_path, lines=True)
label_encoder = joblib.load('data/label_encoder.joblib')
tokenizer = joblib.load('data/tokenizer.joblib')
X_input_seq = tokenizer.texts_to_sequences(input_data['processed_text'])
X_input_pad = pad_sequences(X_input_seq, maxlen=100, padding='post', truncating='post')

# Make predictions
y_pred_prob = model.predict(X_input_pad)
y_pred = label_encoder.inverse_transform(y_pred_prob.argmax(axis=1))
output_data = pd.DataFrame({'category': y_pred})
# Save the output to a new JSON file
output_data.to_json(output_path, orient='records', lines=True)