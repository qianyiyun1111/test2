import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# 文本处理函数
def process_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 词干提取和移除停用词
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    # 返回处理后的文本
    return ' '.join(tokens)