"""
Dummy-Tests für Tag 10 - zeigen den CI-Mechanismus, ohne die echte
Anwendung anzufassen. main.py braucht eine laufende MLflow-Registry
(Tag 8), die es in der CI-Umgebung nicht gibt - deshalb bewusst
unabhängige, triviale Tests.

Echte Tests mit gemocktem Modell kommen bei Tag 15.
"""


def test_addition():
    assert 1 + 1 == 2


def test_string_upper():
    assert "mlops".upper() == "MLOPS"
