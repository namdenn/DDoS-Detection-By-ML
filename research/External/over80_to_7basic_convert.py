import os
import pandas as pd

input_file = [
    # Training Dataset
    os.getenv("TRAINING_INPUT_FILE"),
    # Testing Dataset
    os.getenv("TESTING_INPUT_FILE")
]
output_file = [
    # Training Dataset
    'train_converted_flow_features.csv',
    # Testing Dataset
    'test_converted_flow_features.csv'
]

for i in range(2):
    column_mapping = {
        " Source IP": "Source",
        " Destination IP": "Destination",
        " Protocol": "Protocol",
        " Timestamp": "Time",
        " Label": "label",

    }

    output_header = ["Time", "Source", "Destination", "Protocol", "Length","Info","label"]

    data = pd.read_csv(input_file[i])

    data['Length'] = data["Total Length of Fwd Packets"] + data[" Total Length of Bwd Packets"]
    data['Info'] = data[" Source Port"].astype(str) + " -> " + data[" Destination Port"].astype(str)

    data = data.rename(columns=column_mapping)
    filtered_data = data[output_header]

    filtered_data.to_csv(output_file[i], index=False)
    print(f"Filtered data saved to {output_file[i]}.")
