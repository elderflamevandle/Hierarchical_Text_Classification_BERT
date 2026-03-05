import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def preprocess_data(df, category_cols=['Cat1', 'Cat2', 'Cat3']):
    if 'Title' in df.columns and 'Text' in df.columns:
        df['text'] = df['Title'] + ' ' + df['Text']
        df['Text'] = df['Text'].apply(preprocess_text)
    elif 'text' not in df.columns:
        raise ValueError("DataFrame must contain either 'text' column or both 'Title' and 'Text' columns")

    if not all(col in df.columns for col in category_cols):
        raise ValueError(f"DataFrame must contain these category columns: {', '.join(category_cols)}")

    label_encoders = []
    encoded_cols = []

    for i, col in enumerate(category_cols):
        le = LabelEncoder()
        encoded_col_name = f'Cat{i+1}_encoded'
        df[encoded_col_name] = le.fit_transform(df[col])
        label_encoders.append(le)
        encoded_cols.append(encoded_col_name)

    df['text'] = df['text'].fillna('')
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

    df = df[['text'] + encoded_cols]

    return df, label_encoders

def preprocess_pipeline(csv_path, category_cols=['Cat1', 'Cat2', 'Cat3']):
    df = pd.read_csv(csv_path)
    df, label_encoders = preprocess_data(df, category_cols)
    
    for i, le in enumerate(label_encoders):
        joblib.dump(le, f'le{i+1}.joblib')
        
    return df, label_encoders