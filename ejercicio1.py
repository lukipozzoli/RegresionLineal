import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer y limpiar los datos
df = pd.read_csv("ejercicio_1.csv", header=None)
df.columns = ['x', 'y']
df = df.iloc[1:].astype(float)

# Extraer x y y, y modificar y
x = df['x'].values
y = df['y'].values
y_modificado = y + 12  # y_i + 12

# Matriz X para y = ax + b
X = np.column_stack((x, np.ones_like(x)))

# Calcular beta* con y modificada
XtX = X.T @ X
XtY = X.T @ y_modificado
beta_mod = np.linalg.inv(XtX) @ XtY

# Predicción de y con la recta ajustada
y_pred_mod = X @ beta_mod

# Graficar puntos modificados y recta
plt.figure(figsize=(8, 6))
plt.scatter(x, y_modificado, color='green', label='Datos (y + 12)')
plt.plot(x, y_pred_mod, color='orange', label='Recta ajustada')
plt.title('Regresión con puntos (x, y + 12)')
plt.xlabel('x')
plt.ylabel('y + 12')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar coeficientes
print(f"a (pendiente) = {beta_mod[0]:.4f}")
print(f"b (ordenada al origen) = {beta_mod[1]:.4f}")
