import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
ARCHIVO_DATASET = '../data/V2/dataset_voluntariado_aumentado.csv'

print(f"--- Cargando dataset para optimización: {ARCHIVO_DATASET} ---")
if not os.path.exists(ARCHIVO_DATASET):
    raise FileNotFoundError(f"❌ No encuentro el archivo {ARCHIVO_DATASET}. Asegúrate de que esté en esta carpeta.")

df = pd.read_csv(ARCHIVO_DATASET)

# Definimos variables (Igual que antes)
X = df[[
    'X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
    'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad', 
    'X7_Tipo_Carrera', 'X8_Beca', 'X9_Carrera_Id'
]]
Y = df['Y_Horas_Totales_Finales']

# Split (mantenemos la misma semilla 42 para comparar manzanas con manzanas)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==========================================
# 2. DEFINIR LA REJILLA DE BÚSQUEDA
# ==========================================
# El modelo probará TODAS estas combinaciones (4 x 4 x 4 x 2 x 2 = 256 modelos distintos)
param_grid = {
    'n_estimators': [100, 200, 300, 500],      # Número de árboles
    'max_depth': [3, 4, 5, 6],                 # Profundidad (Complejidad)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Velocidad de aprendizaje
    'subsample': [0.8, 1.0],                   # % de filas usadas por árbol
    'colsample_bytree': [0.8, 1.0]             # % de columnas usadas por árbol
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

print("\n--- 🐢 Iniciando Búsqueda Intensiva (Grid Search) ---")
print("Esto puede tardar unos minutos dependiendo de tu PC...")

# Configuración de la búsqueda
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='r2', 
    cv=3,           # Validación cruzada de 3 pliegues (triple verificación)
    verbose=1, 
    n_jobs=-1       # Usar todos los núcleos del CPU
)

grid_search.fit(X_train, Y_train)

# ==========================================
# 3. RESULTADOS Y GUARDADO
# ==========================================
print("\n=============================================")
print("🏆 ¡OPTIMIZACIÓN COMPLETADA!")
print("=============================================")
print(f"Mejor combinación encontrada: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)

# Métricas del modelo ganador
new_r2 = r2_score(Y_test, Y_pred)
new_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print(f"\n📊 RESULTADOS DEL MEJOR MODELO:")
print(f"Nuevo R2: {new_r2:.4f}")
print(f"Nuevo RMSE: ±{new_rmse:.2f} Horas")

# Comparativa rápida
print(f"\nComparativa vs Anterior (0.6345):")
if new_r2 > 0.6345:
    diff = (new_r2 - 0.6345) * 100
    print(f"✅ MEJORA DETECTADA: +{diff:.2f}% de precisión explicada.")
    
    # Guardar el modelo ganador
    if not os.path.exists('model_assets'):
        os.makedirs('model_assets')
    
    best_model.save_model('model_assets/modelo_voluntariado_xgb_optimizado.json')
    print("💾 Modelo optimizado guardado en 'model_assets/modelo_voluntariado_xgb_optimizado.json'")
else:
    print("⚠️ El modelo no mejoró significativamente. Los datos tienen mucho ruido natural.")