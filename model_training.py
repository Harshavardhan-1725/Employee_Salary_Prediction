import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import cloudpickle

# Load and clean data
data = pd.read_csv("adult.csv")
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data = data.drop(columns=['education'])

# Encode categorical features
encoder = LabelEncoder()
for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
    data[col] = encoder.fit_transform(data[col])

X = data.drop(columns=['income'])
y = data['income']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("best_model.pkl", "wb") as f:
    cloudpickle.dump(model, f)

print("✅ Model trained and saved as best_model.pkl")
