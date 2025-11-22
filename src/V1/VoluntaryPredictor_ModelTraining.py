import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import joblib
import os

# Cargar el Dataset simulado
df = pd.read_csv('voluntariado_dataset.csv')

# Definir X (Variables Independientes) y Y (Variable Dependiente)
X = df[['X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
        'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad']]
Y = df['Y_Horas_Totales_Finales']

# Separar el Test Set (20%) del resto (80%)
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Separar Validation Set (25% de X_train_val = 20% del total)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.25, random_state=42
)

print(f"Total de muestras (100%): {len(df)}")
print(f"Muestras Train (60%): {len(X_train)}")
print(f"Muestras Validation (20%): {len(X_val)}")
print(f"Muestras Test (20%): {len(X_test)}")

# Instanciamos el StandardScaler para estandarizar las características
scaler = StandardScaler()

# Entrenar el Scaler 
scaler.fit(X_train)

# 2. Aplicar la Transformación a TODOS los conjuntos
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Instanciar el SGDRegressor:
model = SGDRegressor(
    loss='squared_error', # Función de coste para Regresión Lineal (Error Cuadrático Medio)
    max_iter=1000, 
    tol=1e-3, 
    random_state=42
)

# Entrenamiento: Ajuste de los parámetros (pesos Theta) para encontrar el patrón.
model.fit(X_train_scaled, Y_train)

# Mostrar el patrón aprendido por el modelo (coeficientes)
print("\n--- Patrón Aprendido (Coeficientes Theta) ---")
feature_names = X.columns
for name, coef in zip(feature_names, model.coef_):
    print(f"Peso de {name:<25}: {coef:>6.2f}")

# Predicciones en los tres conjuntos
Y_train_pred = model.predict(X_train_scaled)
Y_val_pred = model.predict(X_val_scaled)
Y_test_pred = model.predict(X_test_scaled)

# Función para calcular el RMSE (Raíz del Error Cuadrático Medio)
def calculate_rmse(Y_true, Y_pred):
    return np.sqrt(mean_squared_error(Y_true, Y_pred))

# Cálculo del RMSE para diagnóstico
rmse_train = calculate_rmse(Y_train, Y_train_pred)
rmse_val = calculate_rmse(Y_val, Y_val_pred)

print("\n--- Diagnóstico de Generalización ---")
print(f"RMSE (Train Set): {rmse_train:.2f} Horas")
print(f"RMSE (Validation Set): {rmse_val:.2f} Horas")

# Análisis de diagnóstico: Buscando Overfitting (Alta Varianza)
if rmse_val > rmse_train * 1.1: # Si el error de validación es > 10% mayor que el de train
    print("⚠ ¡ATENCIÓN! Alta Varianza (Overfitting). El modelo predice mejor sus datos de entrenamiento.")
else:
    print("✅ Buena Generalización. Los errores son similares, indicando bajo Overfitting.")

# Evaluación Final (Generalización Real)
rmse_test = calculate_rmse(Y_test, Y_test_pred)
r2_test = r2_score(Y_test, Y_test_pred) # Cálculo del R^2

print(f"\n--- Evaluación Final (Generalización Real) ---")
print(f"Error promedio de predicción (RMSE): ±{rmse_test:.2f} horas.")
print(f"Coeficiente de Determinación (R^2): {r2_test:.4f}")

print(f"Interpretación: El {r2_test*100:.2f}% de la variación en las horas finales es explicada por las variables del modelo.")

# Crear la carpeta para guardar los activos
if not os.path.exists('model_assets'):
    os.makedirs('model_assets')

joblib.dump(model, 'model_assets/voluntariado_regresor.pkl')
joblib.dump(scaler, 'model_assets/voluntariado_scaler.pkl')

print("\n✅ Persistencia exitosa: Modelo y Escalador guardados en 'model_assets/'.")