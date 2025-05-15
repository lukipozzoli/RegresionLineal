import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usa backend que no requiere interfaz gráfica
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("ejercicio_2.csv", header=None)
df.columns = ['x', 'y']
df = df.iloc[1:].astype(float)

# Extraer variables
x = df['x'].values
y = df['y'].values

# Crear matriz X con columna de unos
X = np.column_stack((x, np.ones_like(x)))

# Calcular beta* usando la fórmula de mínimos cuadrados
XtX = X.T @ X
XtY = X.T @ y
beta = np.linalg.inv(XtX) @ XtY  # beta[0] = a, beta[1] = b

# Predecir y dibujar la recta
y_pred = X @ beta

# Calcular ECM (Error Cuadrático Medio)
ecm = np.mean((y - y_pred)**2)

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x, y_pred, color='red', label='Recta ajustada')
plt.title('ejercicio_2.csv')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar gráfico en archivo
plt.savefig("salida.png")  # Guarda el gráfico como imagen

# Mostrar coeficientes y ECM
print(f"a (pendiente) = {beta[0]:.4f}")
print(f"b (ordenada al origen) = {beta[1]:.4f}")
print(f"ECM (Error Cuadrático Medio) = {ecm:.4f}")
