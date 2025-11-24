import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

ARCHIVO_DATASET = '../data/V2/dataset_voluntariado_aumentado1000.csv'

print(f"--- Cargando dataset para optimización: {ARCHIVO_DATASET} ---")
if not os.path.exists(ARCHIVO_DATASET):
    raise FileNotFoundError(f" No encuentro el archivo {ARCHIVO_DATASET}. Asegúrate de que esté en esta carpeta.")

df = pd.read_csv(ARCHIVO_DATASET)

X = df[[
    'X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
    'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad', 
    'X7_Tipo_Carrera', 'X8_Beca', 'X9_Carrera_Id'
]]
Y = df['Y_Horas_Totales_Finales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 500],      # Número de árboles
    'max_depth': [3, 4, 5, 6],                 # Profundidad (Complejidad)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Velocidad de aprendizaje
    'subsample': [0.8, 1.0],                   # % de filas usadas por árbol
    'colsample_bytree': [0.8, 1.0]             # % de columnas usadas por árbol
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

print("\n--- Iniciando Búsqueda Intensiva (Grid Search) ---")
print("Esto puede tardar unos minutos dependiendo de tu PC...")

grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='r2', 
    cv=3,           # Validación cruzada
    verbose=1, 
    n_jobs=-1      
)

grid_search.fit(X_train, Y_train)


print("\n=============================================")
print("Optimizacion completada")
print(f"Mejor combinacion: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)

new_r2 = r2_score(Y_test, Y_pred)
new_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print(f"\n RESULTADOS DEL MEJOR MODELO:")
print(f"Nuevo R2: {new_r2:.4f}")
print(f"Nuevo RMSE: ±{new_rmse:.2f} Horas")



# Ver grafico comparativo
results_df = pd.DataFrame(grid_search.cv_results_)

# Filtramos las columnas que nos interesan
cols_interes = ['param_n_estimators', 'param_max_depth', 'param_learning_rate', 'mean_test_score']
results_df = results_df[cols_interes]

# Crear carpeta para gráficos si no existe
if not os.path.exists('graficos_analisis'):
    os.makedirs('graficos_analisis')

# GRÁFICO 1: Heatmap (Profundidad vs Learning Rate)
# Promediamos los scores para cada combinación de depth y learning_rate (ignorando n_estimators)
pivot_table = results_df.pivot_table(values='mean_test_score', index='param_max_depth', columns='param_learning_rate')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".3f")
plt.title('Impacto de Profundidad vs Velocidad de Aprendizaje (R2 Score)')
plt.ylabel('Profundidad del Árbol (max_depth)')
plt.xlabel('Velocidad (learning_rate)')
plt.savefig('graficos_analisis/heatmap_depth_vs_lr.png')
print("-> Gráfico 1 guardado: graficos_analisis/heatmap_depth_vs_lr.png")

# GRÁFICO 2: Línea de Rendimiento por N_Estimators
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='param_n_estimators', y='mean_test_score', hue='param_learning_rate', palette='tab10', marker='o')
plt.title('Rendimiento según Cantidad de Árboles')
plt.ylabel('R2 Score Promedio')
plt.xlabel('Número de Árboles (n_estimators)')
plt.grid(True)
plt.savefig('graficos_analisis/lineplot_n_estimators.png')
print("-> Gráfico 2 guardado: graficos_analisis/lineplot_n_estimators.png")

# GRÁFICO 3: Top 10 Mejores Modelos
top_10 = results_df.sort_values(by='mean_test_score', ascending=False).head(10)
print("\n--- TOP 5 MEJORES COMBINACIONES ---")
print(top_10[['param_n_estimators', 'param_max_depth', 'param_learning_rate', 'mean_test_score']].head(5))

best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
new_r2 = r2_score(Y_test, Y_pred)

print(f"\nRESULTADO FINAL (Test Set): R2 = {new_r2:.4f}")