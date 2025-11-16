import sys
from src.utils.logger import logger

def error_message(err:Exception):
    _,_,exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(err)}]"
        return error_message
    else:
        return f"An error occurred: [{str(err)}]"
    
class CustomException(Exception):
    def __init__(self, err:Exception):
        formatted_message = error_message(err)
        super().__init__(formatted_message)
        self.error_message = formatted_message
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message   


