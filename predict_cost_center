import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# New path
new_path = 'C:\\Users\\simon\\OneDrive\\Desktop\\'

# Load data from transaktionen2021.txt and transaktionen2022.txt
train_data_2021 = pd.read_csv(new_path + '2021.txt', encoding='cp1252', delimiter=';')
train_data_2022 = pd.read_csv(new_path + '2022.txt', encoding='cp1252', delimiter=';')

# Combine datasets
train_data = pd.concat([train_data_2021, train_data_2022], ignore_index=True)

# Explore and prepare data
print(train_data.info())
train_data = train_data.dropna(subset=['Buchungstext', 'KOST1 - Kostenstelle', 'Konto', 'Gegenkonto'])

# Load test data
test_data = pd.read_csv(new_path + '2023.txt', encoding='cp1252', delimiter=';')
test_data = test_data.dropna(subset=['Buchungstext', 'Konto', 'Gegenkonto', 'KOST1 - Kostenstelle'])  # Remove rows with missing values

# Select desired columns
selected_columns = ['Konto', 'BU-Schlüssel', 'Gegenkonto', 'Buchungstext', 'Stapel-Nr.', 'BSNr',
                    'Herkunft-Kz', 'Belegfeld 2', 'Gesellschaftername', 'Beteiligtennummer', 'Identifikationsnummer',
                    'Zeichnernummer', 'Bezeichnung SoBil-Sachverhalt', 'Kennzeichen SoBil-Buchung',
                    'Generalumkehr (GU)', 'Abrechnungsreferenz', 'BVV-Position']

# Combine selected columns to create a new feature
train_data['Combined_Feature'] = train_data[selected_columns].astype(str).agg(' '.join, axis=1)
test_data['Combined_Feature'] = test_data[selected_columns].astype(str).agg(' '.join, axis=1)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_data['Combined_Feature'])
X_test = vectorizer.transform(test_data['Combined_Feature'])

# Train a KNN model
y_train = train_data['KOST1 - Kostenstelle']
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

# Make predictions on the test set with the KNN model
predictions_knn = model_knn.predict(X_test)

# Evaluate the KNN model
y_test = test_data['KOST1 - Kostenstelle']
accuracy_knn = accuracy_score(y_test, predictions_knn)
print(f'Accuracy (KNN): {accuracy_knn}')

# Show additional evaluation metrics for the KNN model
print(classification_report(y_test, predictions_knn))

# Percentage of correct predictions for the KNN model
correct_predictions_knn = sum(predictions_knn == y_test)
percentage_correct_knn = (correct_predictions_knn / len(y_test)) * 100
print(f'Percentage of Correct Predictions (KNN): {percentage_correct_knn:.2f}%')

# Inspect the results for the KNN model
results_knn = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions_knn,
    'Combined_Feature': test_data['Combined_Feature'],
    'Konto': test_data['Konto'],
    'Gegenkonto': test_data['Gegenkonto'],
    'Buchungstext': test_data['Buchungstext']
})
print(results_knn)
