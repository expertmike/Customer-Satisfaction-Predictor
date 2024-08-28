import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data preparation (replace with actual data)
data = {
    'interaction_length': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'issue_resolved': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],  # 1 if resolved, 0 if not
    'number_of_interactions': [1, 3, 2, 1, 4, 5, 1, 3, 2, 1],
    'customer_satisfaction': [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # 1 if satisfied, 0 if not
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df[['interaction_length', 'issue_resolved', 'number_of_interactions']]
y = df['customer_satisfaction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Example of predicting customer satisfaction for new interaction data
new_data = pd.DataFrame({
    'interaction_length': [22],
    'issue_resolved': [1],
    'number_of_interactions': [2]
})

prediction = model.predict(new_data)
print(f'Predicted Customer Satisfaction: {prediction[0]}')  # 1: Satisfied, 0: Not Satisfied
