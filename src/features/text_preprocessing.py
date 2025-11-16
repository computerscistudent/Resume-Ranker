import pandas as pd
import re, string, spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import logger
from src.utils.exception import CustomException
import os
import joblib

class TextPreprocessor:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(max_features=3000)

    def clean_text(self, text):
        try:
            text = text.lower()
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
            text = re.sub(r"\d+", " ", text)
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and len(token)>2 ]
            return " ".join(tokens)
        except Exception as e:
            raise CustomException(e)

    def preprocess(self):
        try:
            input_path = os.path.join(self.processed_dir, "resume_texts.csv")
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} text samples for preprocessing")

            df["clean_text"] = df["text"].apply(self.clean_text)
            logger.info("Text cleaning complete")

            clean_path = os.path.join(self.processed_dir, "cleaned_resumes.csv")
            df.to_csv(clean_path)
            logger.info(f"Saved cleaned text to {clean_path}")

            X = self.vectorizer.fit_transform(df["clean_text"])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(X.toarray(), columns =feature_names)
            tfidf_df.insert(0, "filename", df["filename"])

            vec_path = os.path.join(self.processed_dir, "tfidf_features.csv")
            tfidf_df.to_csv(vec_path, index=False)
            logger.info(f"TF-IDF features saved to {vec_path}")
            joblib.dump(self.vectorizer, os.path.join(self.processed_dir, "tfidf_vectorizer.pkl"))
            logger.info("Saved TF-IDF vectorizer for future use")

            return df, tfidf_df
            

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    tp = TextPreprocessor()
    tp.preprocess()