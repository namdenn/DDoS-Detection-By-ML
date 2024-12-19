import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Load dataset
data = pd.read_csv('/home/ubuntu/DDoS-Detection-By-ML/extracted_flow_features.csv')

# Drop flag columns 
flag_columns = [
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Src IP', 'Src Port', 'Dst IP',
    'Dst Port', 'Protocol'
]
data = data.drop(columns=flag_columns)

# Randomly assign 'Label' column for demonstration (replace with real labels if available)
label_mapping = {'BENIGN': 0, 'UDP': 1, 'MSSQL': 1, 'LDAP': 1, 'NetBIOS': 1}  # Adjust according to your dataset
data['label'] = data['label'].map(label_mapping)

# Separate features and target
X = data.drop(['label','Flow ID'], axis=1)
y = data['label']

# Calculate information gain
info_gain = mutual_info_classif(X, y, random_state=42)
feature_info = pd.DataFrame({'Feature': X.columns, 'Information Gain': info_gain})

# Sort features by descending information gain
sorted_features = feature_info.sort_values(by='Information Gain', ascending=False)
print(sorted_features)

# Get the top 10 features
top_features = sorted_features.head(10)['Feature'].tolist()

# Create a new DataFrame with the top 10 features and the label
top_features_data = data[top_features + ['label']]

# Save the new DataFrame to a CSV file
output_path = '/home/ubuntu/DDoS-Detection-By-ML/top_10_features.csv'
top_features_data.to_csv(output_path, index=False)

print(f"CSV with top 10 features saved to: {output_path}")
