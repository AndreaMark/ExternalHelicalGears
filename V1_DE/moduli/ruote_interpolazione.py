def interp(x1, y1, x2, y2, x):
    """
    Effettua un'interpolazione lineare tra due punti dati.

    Args:
        x1 (float): Coordinata x del primo valore.
        y1 (float): Coordinata y associata al primo valore.
        x2 (float): Coordinata x del secondo valore.
        y2 (float): Coordinata y associata al secondo punto.
        x (float): Valore x per cui si desidera l'interpolazione.

    Returns:
        float: Il valore interpolato corrispondente a x.
    """
    if x1 == x2:
        return y1  # Evita la divisione per zero
    else:
        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1
        return m * x + q
    

# valore = interp(5, 52, 7, 51, 5.62)
# print(valore)