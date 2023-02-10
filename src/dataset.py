import pandas as pd

class Dataset():
    def __init__(self, file_path) -> None:
        initial_dataset = pd.read_csv(file_path, parse_dates=['Date', 'Open'])
        
        # !!! draw is not yet encoded
        self.target = initial_dataset["H"]
        # drop targets
        self.data = initial_dataset.drop(columns=["H", "A"])

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


    def append_new_match(self):
        pass