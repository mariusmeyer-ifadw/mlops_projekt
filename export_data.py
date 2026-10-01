"""
Einmaliger Export des Iris-Datensatzes als CSV-Datei.

Bisher kam der Datensatz direkt aus sklearn.datasets.load_iris() - für DVC
brauchen wir aber eine echte Datei zum Versionieren, wie es bei echten
Projekten mit echten Datensätzen der Fall wäre.

Führt dieses Skript einmal aus, bevor ihr mit `dvc add` startet.
"""
import pandas as pd
from sklearn.datasets import load_iris
import os

os.makedirs("data", exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [
    "sepal_length", "sepal_width", "petal_length", "petal_width", "target"
]

df.to_csv("data/iris.csv", index=False)
print(f"Datensatz exportiert: data/iris.csv ({len(df)} Zeilen)")
