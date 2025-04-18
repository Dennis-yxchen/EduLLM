import os
import json

class DatasetReader:
    def __init__(self, data_path, file_name='comp2501.json', preprocess_func=None):
        # Get the directory of the current script
        current_dir = os.path.dirname(__file__)
        # Construct the path to the data file
        self.data_path = os.path.join(data_path, file_name)
        # Preprocessing function
        self.preprocess_func = preprocess_func
    
    def read_data(self):
        # Read the JSON file
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def preprocess(self, data):
        # Use the provided preprocessing function or the default processing logic
        if self.preprocess_func:
            return self.preprocess_func(data)
        else:
            # Default processing logic, return the original data
            return data
    
    def get_data(self):
        # Read and preprocess data
        data = self.read_data()
        processed_data = self.preprocess(data)
        # print(f"processed_data: {processed_data}")
        return processed_data


if __name__ == '__main__':
    # Example of a custom preprocessing function
    def uppercase_values(data):
        result = []
        for i in data:
            tmp = str(i).upper()
            result.append(tmp)
        return result
    # Test case: Using a custom preprocessing function
    reader = DatasetReader(preprocess_func=uppercase_values)
    data = reader.get_data()
    print(data)
    print(f"end of uppercase_values test")
    # Test case: Not using a custom preprocessing function
    reader = DatasetReader()
    data = reader.get_data()
    print(data)