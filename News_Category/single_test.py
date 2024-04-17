import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from process_text import process_text

def predict_category(headline, authors, link, des, date):
    new_data = process_text(headline + ' ' + authors + ' ' + des + ' ' + link)
    new_data_seq = tokenizer.texts_to_sequences([new_data])
    new_data_pad = pad_sequences(new_data_seq, maxlen=100)
    prediction = model.predict(new_data_pad)
    predicted_category_index = np.argmax(prediction, axis=1)
    predicted_category = label_encoder.inverse_transform(predicted_category_index)
    return predicted_category

# 加载模型
model = load_model('data/news_category_model.h5')
# 读取数据集
label_encoder = joblib.load('data/label_encoder.joblib')
tokenizer = joblib.load('data/tokenizer.joblib')

# 测试输入（headline, authors, link, des, date）输出category
test_headline = "Jeffrey Katzenberg's Email To Weinstein: 'There Appear To Be Two Harvey Weinsteins'"
test_authors = "Jenna Amatulli"
test_link = "https://www.huffingtonpost.com/entry/jeffrey-katzenbergs-email-to-weinstein-there-appear-to-be-two-harvey-weinsteins_us_59e10a08e4b0a52aca17c670"
test_description = "\"You have done terrible things to a number of women,\" the former Walt Disney Studios chairman \u2014 and Weinstein's friend more than\n 30 years \u2014\u00a0wrote."
test_date = "2017-10-13"
print(test_date)
#predicted_category = predict_category(test_headline, test_authors, test_link, test_description, test_date)
#print(f"预测的类别为：{predicted_category}")
