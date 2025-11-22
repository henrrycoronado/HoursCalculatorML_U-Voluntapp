import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. CARGA Y CONFIGURACIÓN
# ==========================================
ARCHIVO_DATASET = '../data/V2/dataset_voluntariado_aumentado.csv'

print(f"--- Cargando dataset: {ARCHIVO_DATASET} ---")
df = pd.read_csv(ARCHIVO_DATASET)

# Definimos las variables predictoras (X) y el objetivo (Y)
# NOTA: XGBoost maneja muy bien las variables sin necesidad de escalar (StandardScaler),
# lo que hace que los gráficos sean más fáciles de interpretar (Unidades reales).
X = df[[
    'X1_Horas_Actuales', 
    'X2_Frec_Semanal', 
    'X3_Horas_Fallidas', 
    'X4_Semanas_Restantes', 
    'X5_Disp_Neta_Restante', 
    'X6_Antiguedad', 
    'X7_Tipo_Carrera', 
    'X8_Beca',
    'X9_Carrera_Id'
]]
Y = df['Y_Horas_Totales_Finales']

# División: 80% Entrenamiento, 20% Prueba (Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Datos de Entrenamiento: {len(X_train)} muestras")
print(f"Datos de Prueba (Test): {len(X_test)} muestras")

# ==========================================
# 2. ENTRENAMIENTO DEL MODELO XGBOOST
# ==========================================
print("\n--- Entrenando Modelo XGBoost ---")

# Configuración del "Cerebro" del XGBoost
model = xgb.XGBRegressor(
    objective='reg:squarederror', # Tarea de regresión
    n_estimators=100,             # Número de árboles de decisión
    learning_rate=0.1,            # Velocidad de aprendizaje
    max_depth=3,                  # Profundidad de los árboles (evita overfitting)
    random_state=42,
    n_jobs=-1                     # Usar todos los procesadores
)

model.fit(X_train, Y_train)

# ==========================================
# 3. EVALUACIÓN Y MÉTRICAS
# ==========================================
Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print("\nRESULTADOS DEL MODELO:")
print(f"✅ RMSE (Error Promedio): ±{rmse:.2f} Horas")
print(f"✅ R2 (Precisión General): {r2:.4f} (El {r2*100:.1f}% de la varianza es explicada)")

# ==========================================
# 4. GENERACIÓN DE GRÁFICOS PARA DEFENSA
# ==========================================
# Crear carpeta para guardar evidencias
if not os.path.exists('graficos_defensa'):
    os.makedirs('graficos_defensa')

# GRAFICO A: Importancia de Variables
plt.figure(figsize=(10, 6))
importancias = model.feature_importances_
nombres = X.columns
# Ordenar para mejor visualización
indices = np.argsort(importancias)

plt.title('Factores que más influyen en el Cumplimiento (XGBoost)')
plt.barh(range(len(indices)), importancias[indices], color='#4c72b0', align='center')
plt.yticks(range(len(indices)), [nombres[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.savefig('graficos_defensa/1_importancia_variables.png')
print("\n📊 Gráfico guardado: graficos_defensa/1_importancia_variables.png")

# GRAFICO B: Predicción vs Realidad (Dispersión)
plt.figure(figsize=(8, 8))
plt.scatter(Y_test, Y_pred, alpha=0.6, color='green')
# Línea de perfección ideal
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Predicción Perfecta')
plt.xlabel('Horas Reales (Histórico)')
plt.ylabel('Horas Predichas por XGBoost')
plt.title('Precisión: Real vs Predicho')
plt.legend()
plt.tight_layout()
plt.savefig('graficos_defensa/2_prediccion_vs_real.png')
print("📊 Gráfico guardado: graficos_defensa/2_prediccion_vs_real.png")

# ==========================================
# 5. GUARDAR EL MODELO (PERSISTENCIA)
# ==========================================
if not os.path.exists('model_assets'):
    os.makedirs('model_assets')

# Guardamos en formato JSON (Nativo de XGBoost, más rápido y compatible)
model.save_model('model_assets/modelo_voluntariado_xgb.json')
# También guardamos con joblib por costumbre, pero el JSON es mejor hoy día
joblib.dump(model, 'model_assets/modelo_voluntariado_xgb.pkl')

print("\n💾 ¡Modelo guardado exitosamente en carpeta 'model_assets'!")