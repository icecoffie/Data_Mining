import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
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

# ===============================
# Model Decision Tree
# ===============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_probs = dt_model.predict_proba(X_test)[:, 1]  # Probabilitas kelas positif

# ROC Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# ===============================
# Model Naive Bayes
# ===============================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_probs = nb_model.predict_proba(X_test)[:, 1]

# ROC Naive Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# ===============================
# Plot ROC Curve
# ===============================
plt.figure(figsize=(10, 7))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})', color='blue')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})', color='green')

# Baseline
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.title('Perbandingan Kurva ROC - Decision Tree vs Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
