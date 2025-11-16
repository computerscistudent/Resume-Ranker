import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.features.text_preprocessing import TextPreprocessor
from src.utils.logger import logger
from src.utils.exception import CustomException


class ResumeRanker :
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        self.vectorizer_path = os.path.join(self.processed_dir, "tfidf_vectorizer.pkl")
        self.resume_path = os.path.join(self.processed_dir, "cleaned_resumes.csv")

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.resumes = pd.read_csv(self.resume_path)
        logger.info("Loaded vectorizer and cleaned resume data")

    def rank_resume(self,job_text):
        try:
            preprocessor = TextPreprocessor()
            clean_job = preprocessor.clean_text(job_text)
            job_vec = self.vectorizer.transform([clean_job])
            resume_vec = self.vectorizer.transform(self.resumes["clean_text"])

            scores = cosine_similarity(job_vec, resume_vec)[0]
            self.resumes["similarity"] = scores

            ranked = self.resumes.sort_values(by="similarity", ascending=False)[["filename","similarity"]]
            ranked["similarity"] = (ranked["similarity"]*100).round(2)
            logger.info("Ranking completed successfully")

            output_path = os.path.join(self.processed_dir, "ranked_resumes.csv")
            ranked.to_csv(output_path, index=False)
            logger.info(f"Saved ranked resumes to {output_path}")

            return ranked.head(10)

        except Exception as e:
            raise CustomException(e)    
        
if __name__ == "__main__":
     job_description = """
    We are looking for a Data Scientist with experience in NLP, Python, and FastAPI.
    The candidate should be skilled in model building, deployment, and data visualization.
    """
     rr = ResumeRanker()
     top_resumes=rr.rank_resume(job_description)
     print("\n🏆 Top Matching Resumes:\n", top_resumes)        
