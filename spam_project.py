import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['label', 'text']]

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Show accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("\nModel saved successfully!")

# User input testing
def predict_sms(message):
    vec = vectorizer.transform([message])
    p = model.predict(vec)[0]
    return "SPAM" if p == 1 else "HAM"

# Interactive test
message = input("Enter any SMS to check if it is spam or not: ")
print("Result:", predict_sms(message))
