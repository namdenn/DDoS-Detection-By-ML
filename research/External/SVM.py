import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from CustomEncoder import CustomLabelEncoder

df_train = pd.read_csv("train_extracted_flow_features.csv")
df_test = pd.read_csv("test_extracted_flow_features.csv")

# Encoding the object value
columnNeedEncode = ['Flow ID', 'Src IP', 'Dst IP', 'label']
encoded_map = {}

for col in columnNeedEncode:
    encoder = CustomLabelEncoder()
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

    # Save the encoding map
    encoded_map[f"{col}_map"] = encoder.show_mapping()

# Split data into X and y
X_train = df_train.drop(columns=['label'])  
y_train = df_train['label']
X_test = df_test.drop(columns=['label'])
y_test = df_test['label']

# Initialize the SVM model
svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)  # Customize hyperparameters if needed

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Extract valid classes (support > 0)
valid_classes = [cls for cls, metrics in report.items() if isinstance(metrics, dict) and metrics["support"] > 0]
classes_toRemove = ['5', 'micro avg', 'macro avg', 'weighted avg']
valid_classes = [int(cls) for cls in valid_classes if cls not in classes_toRemove]

# Filter valid samples from y_test and y_pred
filtered_y_test = [y for y in y_test if y in valid_classes]
filtered_y_pred = [y for i, y in enumerate(y_pred) if y_test[i] in valid_classes]

# Calculate accuracy for valid classes only
filtered_accuracy = accuracy_score(filtered_y_test, filtered_y_pred)
print(f"Filtered Accuracy: {filtered_accuracy:.2f}")

# Print Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)