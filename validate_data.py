"""
Musterlösung zu validate_data.py (Tag 9) – NUR für den Kursleiter.
"""
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

schema = pa.DataFrameSchema({
    "sepal_length": Column(float, Check.in_range(0, 10)),
    "sepal_width": Column(float, Check.in_range(0, 10)),
    "petal_length": Column(float, Check.in_range(0, 10)),
    "petal_width": Column(float, Check.in_range(0, 10)),
    "target": Column(int, Check.isin([0, 1, 2])),
})

df = pd.read_csv("data/iris.csv")

try:
    validated = schema.validate(df)
    print(f"Validierung erfolgreich: {len(validated)} Zeilen geprüft, keine Auffälligkeiten.")
except pa.errors.SchemaError as e:
    print("Validierung fehlgeschlagen:")
    print(e)
