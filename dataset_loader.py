import pandas as pd
import json
import numpy as np


class DatasetLoader:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)

        self.data_frame = pd.DataFrame({
            "ActuatorPositions": data["ActuatorPositions"],
            "ActuatorDeviations": data["ActuatorDeviations"],
            "ActuatorActions": data["ActuatorActions"],
            "ThiknessProfiles": data["ThiknessProfiles"],
            "Averages": data["Averages"]
        })

        self.data_length = len(self.data_frame)
        print(f"DataFrame {self.data_length} satır ile başlatıldı.")

    def get_seq_data(self, seq_len, input_headers, output_headers):
        index = seq_len
        input_datas = []
        output_datas = []

        for row in range(self.data_length-seq_len):

            input_seq_data = []
            output_seq_data = []

            for seq in range(seq_len):

                input_row = []
                output_row = []

                for input_header in input_headers:
                    input_row = np.concatenate(
                        [input_row, self.data_frame.loc[row+index-seq, input_header]])

                for outout_header in output_headers:
                    output_row = np.concatenate(
                        [output_row, self.data_frame.loc[row+index-seq, outout_header]])
                    output_row = self.one_hot_encode(output_row, 3)
                input_seq_data.append(input_row)
                output_seq_data.append(output_row)

            input_datas.append(input_seq_data)
            output_datas.append(output_seq_data)

        return input_datas, output_datas

    def one_hot_encode(self, actions: np.ndarray, num_classes: int = 3) -> np.ndarray:
        actions = actions.astype(int)
        one_hot = np.eye(num_classes)[actions]
        return one_hot.reshape([len(actions)*num_classes, 1])

    def positional_encoding(self, data):
        positional_encoding = []
        # ....
        return data+positional_encoding


loader = DatasetLoader("Dataset_1_230520251433.json")

input_headers = ["ActuatorPositions", "ActuatorDeviations", "ThiknessProfiles"]
output_headers = ["ActuatorActions"]

input_datas, output_datas = loader.get_seq_data(
    8,
    input_headers,
    output_headers)
