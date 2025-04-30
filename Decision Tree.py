import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Path ke file CSV
file_path = r"D:\Data Mining - Malika\Kelulusan.csv"

# Baca data
try:
    df = pd.read_csv(file_path)
    print("✅ File CSV berhasil dibaca.")
except Exception as e:
    print("❌ Gagal membaca file CSV:", e)
    exit()

# Konversi kolom 'Lulus' ke numerik
df['Lulus'] = df['Lulus'].map({'Ya': 1, 'Tidak': 0})

# Fitur dan label
X = df[['Kehadiran', 'Nilai_Tugas', 'Nilai_UTS', 'Nilai_UAS']]
y = df['Lulus']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
pred = model.predict(X_test)
print("=== Decision Tree ===")
print("Akurasi:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Visualisasi pohon keputusan
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=X.columns, class_names=["Tidak", "Ya"], filled=True, rounded=True)
plt.title("Visualisasi Pohon Keputusan (Decision Tree)")
plt.show()
