"""
Platzhalter-Retraining für Tag 11.

Läuft komplett auf dem GitHub-Runner, ohne Abhängigkeit zu eurer lokalen
MLflow-/MinIO-Infrastruktur - deshalb sklearn.datasets.load_iris() direkt
statt der CSV-Datei aus Tag 9 oder MLflow-Tracking aus Tag 6/7.

In der Praxis würde ein echtes Retraining gegen eure eigene Infrastruktur
laufen - das braucht einen Self-hosted Runner (Tag 13).
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Platzhalter-Retraining abgeschlossen. Accuracy: {accuracy:.3f}")
