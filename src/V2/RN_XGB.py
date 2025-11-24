import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# 1. CARGA DE DATOS
ARCHIVO_DATASET = '../data/V2/dataset_voluntariado_aumentado1000.csv'
if not os.path.exists(ARCHIVO_DATASET):
     raise FileNotFoundError(f" No encuentro el archivo {ARCHIVO_DATASET}. Asegúrate de que esté en esta carpeta.")

df = pd.read_csv(ARCHIVO_DATASET)

X = df[['X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
        'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad', 
        'X7_Tipo_Carrera', 'X8_Beca', 'X9_Carrera_Id']]
Y = df['Y_Horas_Totales_Finales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==========================================
# MODELO 1: XGBOOST 
# ==========================================
print("--- Entrenando XGBoost ---")
# Usamos los parámetros que ya sabemos que funcionan bien
xgb_model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=4, 
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, Y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(Y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(Y_test, y_pred_xgb))

# ==========================================
# MODELO 2: RED NEURONAL 
# ==========================================
print("--- Entrenando Red Neuronal ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=2000, 
    random_state=42,
    early_stopping=True # Para evitar overfitting
)

nn_model.fit(X_train_scaled, Y_train)
y_pred_nn = nn_model.predict(X_test_scaled)
r2_nn = r2_score(Y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(Y_test, y_pred_nn))

# ==========================================
# COMPARATIVA
# ==========================================
print("\n" + "="*40)
print("     RESULTADOS DE LA BATALLA     ")
print("="*40)
print(f"{'Métrica':<10} | {'XGBoost (Árboles)':<18} | {'Red Neuronal (MLP)':<18}")
print("-" * 52)
print(f"{'R2 Score':<10} | {r2_xgb:.4f}             | {r2_nn:.4f}")
print(f"{'RMSE':<10} | {rmse_xgb:.2f}              | {rmse_nn:.2f}")
print("-" * 52)

# Gráfico de Comparación
plt.figure(figsize=(10, 5))
plt.bar(['XGBoost', 'Red Neuronal'], [r2_xgb, r2_nn], color=['#4c72b0', '#dd8452'])
plt.title('Comparación de Precisión (R2 Score)')
plt.ylim(0, 1.0)
plt.ylabel('R2 Score (Más alto es mejor)')
for i, v in enumerate([r2_xgb, r2_nn]):
    plt.text(i, v + 0.02, str(round(v, 4)), ha='center', fontweight='bold')

if not os.path.exists('graficos_analisis'):
    os.makedirs('graficos_analisis')
plt.savefig('graficos_analisis/comparativa_modelos.png')
print("\n📊 Gráfico guardado: graficos_analisis/comparativa_modelos.png")

if r2_xgb > r2_nn:
    print("\n🏆 GANADOR: XGBoost. (Defensa: 'Los métodos de ensamble suelen superar a las redes neuronales en datos tabulares pequeños')")
else:
    print("\n🏆 GANADOR: Red Neuronal. (Defensa: 'La red logró capturar relaciones no lineales complejas mejor que los árboles')")