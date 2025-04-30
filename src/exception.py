import sys
from src.logger import  logging

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"[{filename}] line {line_number} - {error_message}"

    def __str__(self):
        return self.error_message
    

if __name__=='__main__':
    try:
        logging.info("enter the try block")
        a = 1/0
        print('this wont be rpinted')

    except Exception as e:
        raise NetworkSecurityException(e,sys)