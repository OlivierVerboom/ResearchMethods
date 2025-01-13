import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download

download('punkt')
download('stopwords')
download('wordnet')

def clean_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]

    text = re.sub(r'<.*?>', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    sentences = sent_tokenize(text)

    return tokens, sentences

def process_files(input_dir, output_csv):
    results = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing: {file_path}")
            
            tokens, sentences = clean_and_preprocess(file_path)
            results.append({
                "file_name": file_name,
                "num_tokens": len(tokens),
                "num_sentences": len(sentences),
                "lexical_diversity": len(set(tokens)) / len(tokens) if tokens else 0,
                "avg_sentence_length": sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    input_dir = "data/post-21th century" # Name it as the directory which you need
    output_csv = "linguistic_features_post-21thcentury.csv"
    
    process_files(input_dir, output_csv)
