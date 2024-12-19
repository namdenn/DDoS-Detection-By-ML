from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.mapping = {}
        self.next_label = None

    def fit(self, data):
        """Fit the encoder on training data."""
        self.encoder.fit(data)
        self.mapping = {value: label for label, value in enumerate(self.encoder.classes_)}
        self.next_label = len(self.mapping)  # Start assigning new labels from here
        return self

    def transform(self, data):
        """Transform data, assigning new labels to unseen values."""
        encoded = []
        for value in data:
            if value in self.mapping:
                encoded.append(self.mapping[value])
            else:
                # Assign a new label for unseen value
                self.mapping[value] = self.next_label
                encoded.append(self.next_label)
                self.next_label += 1
        return encoded

    def fit_transform(self, data):
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, labels):
        """Inverse transform to recover original values."""
        reverse_mapping = {label: value for value, label in self.mapping.items()}
        return [reverse_mapping[label] for label in labels]
    
    def show_mapping(self):
        encoded_map = {label: value for value, label in self.mapping.items()}
        return encoded_map