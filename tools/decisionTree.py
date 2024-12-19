import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your dataset
data = pd.read_csv('top_10_features.csv')

# Define features (X) and target (y)
X = data.drop('label', axis=1)  # All columns except the label
y = data['label']              # Target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)  # You can also use 'entropy'
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

