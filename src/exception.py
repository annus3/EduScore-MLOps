import sys
import traceback
import logging

def error_message_detail(error) -> str:
    """
    Constructs a detailed error message with file name and line number.

    Parameters:
        error (Exception): The caught exception.

    Returns:
        str: Formatted error message with traceback info.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown file"
    line_number = exc_tb.tb_lineno if exc_tb else "Unknown line"
    return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"


class CustomException(Exception):
    """
    Custom exception class for capturing detailed tracebacks.
    """

    def __init__(self, error):
        super().__init__(error)
        self.error_message = error_message_detail(error)

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         # Simulate an error for demonstration
#         1 / 0
#     except Exception as e:
#         logging.error("An error occurred.")
#         raise CustomException(e)