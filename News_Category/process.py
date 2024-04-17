import pandas as pd
from sklearn.model_selection import train_test_split
from process_text import process_text

# 可调整参数
input_path = "data/News_Category.json"
output_path = "data/processed_data.json"

# 读取数据集
data = pd.read_json(input_path, lines=True)

# 对文本进行处理
print("Start.")
data['processed_text'] = data['headline'] + ' ' + data['authors'] + ' ' + data['short_description'] + ' ' + data['link']
data['processed_text'] = data['processed_text'].apply(process_text)
data.to_json(output_path, orient='records', lines=True)
print("End.")