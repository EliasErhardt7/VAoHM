import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your features
data = pd.read_csv('features.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save classifier
joblib.dump(clf, 'exercise_classifier.pkl')
print("Classifier trained and saved as 'exercise_classifier.pkl'")
