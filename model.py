# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/housing.csv")
df = df.dropna()

# Features and label
X = df[['location', 'total_sqft', 'bath']]
y = df['price']  # Already in lakhs

# Column transformer for 'location'
column_trans = ColumnTransformer([
    ('location_enc', OneHotEncoder(handle_unknown='ignore'), ['location'])
], remainder='passthrough')

# Create pipeline
pipeline = Pipeline([
    ('transformer', column_trans),
    ('regressor', LinearRegression())
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model.pkl')
print("✅ Model retrained and saved.")
