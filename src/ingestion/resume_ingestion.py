import os
import pdfplumber
import docx
import pandas as pd
from src.utils.logger import logger
from src.utils.exception import CustomException

class ResumeIngestion:
    def __init__(self,  raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def extract_text_from_pdf(self, filepath):
        try:
            with pdfplumber.open(filepath) as f:
                text = "\n".join([page.extract_text() or "" for page in f.pages])
            return text    
        except Exception as e:
            raise CustomException(e)

    def extract_text_from_docx(self, filepath):   
        try:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            raise CustomException(e) 

    def extract_text_from_txt(self, filepath):
        try:
            with open(filepath,"r",encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise CustomException(e)  
            
    def ingest_resumes(self):
        try:
                data=[]
                for file in os.listdir(self.raw_dir):
                    filepath = os.path.join(self.raw_dir, file)
                    if file.endswith(".pdf"):
                        text = self.extract_text_from_pdf(filepath)
                    elif file.endswith(".docx"):
                        text = self.extract_text_from_docx(filepath)
                    elif file.endswith(".txt"):
                        text = self.extract_text_from_txt(filepath)    
                    else:
                        logger.warning(f"Unsupported file format: {file}") 
                        continue
                    data.append({"filename":file,"text":text})
                    logger.info(f"Processed {file}")

                df = pd.DataFrame(data)
                output_path = os.path.join(self.processed_dir,"resume_texts.csv")
                df.to_csv(output_path,index=False)
                logger.info(f"Saved extracted text to {output_path}")       
        except Exception as e:
                raise CustomException(e)    
        
if __name__ == "__main__":
    ingestion = ResumeIngestion()
    ingestion.ingest_resumes()

