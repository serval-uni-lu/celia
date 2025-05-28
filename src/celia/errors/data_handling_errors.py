

class UnSupportedDataTypeError(Exception):
    """Error to raise unsupported data types for the library.

    Args:
        Exception (_type_): _description_
    """
    
    def __init__(self, message:str = """Invalid data type, please assign on of the following:
    - Pandas DataFrame 
    - Dictionary
    - Numpy Array
    - Scipy Matrix (examples)"""):
        
        super().__init__(message)
        self.message = message 
    
    