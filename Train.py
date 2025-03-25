import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load datasets
phishing_df = pd.read_csv("phishing.csv").drop(columns=['Index'], errors='ignore')
legitimate_df = pd.read_csv("legitimateurls.csv", header=None, names=["url"])
phishurls_df = pd.read_csv("phishurls.csv", header=None, names=["url"])

# Assign labels
phishing_df['label'] = -1  # Phishing websites
legitimate_df['label'] = 1  # Legitimate websites
phishurls_df['label'] = -1  # Additional phishing URLs

# Combine URL datasets
url_data = pd.concat([legitimate_df, phishurls_df], ignore_index=True)

# Feature Extraction (Basic)
def extract_features(df):
    df['length'] = df['url'].apply(len)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith("https") else 0)
    df['num_special_chars'] = df['url'].apply(lambda x: len([c for c in x if c in "@_!#$%^&*()<>?/|}{~:"]))
    df['subdomain_count'] = df['url'].apply(lambda x: x.count('.') - 1)
    return df.drop(columns=['url'])

url_data = extract_features(url_data)

# Combine with phishing dataset
final_data = pd.concat([phishing_df, url_data], ignore_index=True)
X = final_data.drop(columns=['label'])
y = final_data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "phishing_model.pkl")
print("Model saved as phishing_model.pkl")
