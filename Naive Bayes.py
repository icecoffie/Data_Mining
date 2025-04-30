import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load data dari file CSV
file_path = r"D:\Data Mining - Malika\Kelulusan.csv"
df = pd.read_csv(file_path)

# Konversi kolom 'Lulus' menjadi numerik
df['Lulus'] = df['Lulus'].map({'Ya': 1, 'Tidak': 0})

# Pisahkan fitur dan label
X = df[['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS']]
y = df['Lulus']

# Split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluasi
print("=== Naive Bayes ===")
print("Akurasi:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
