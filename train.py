import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data.csv')  # Replace with actual file path

data.columns = ["Category", "Message"]
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Drop any missing values
data.dropna(subset=['Category', 'Message'], inplace=True)

# Ensure category mapping is correct
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Check for NaN in target
if data['Category'].isnull().sum() > 0:
    print("Warning: NaN values found in Category column after mapping.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train Model
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))