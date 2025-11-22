import joblib
import pandas as pd
import numpy as np

HORAS_META_N = 100 

# cargar modelo entrenado
loaded_model = joblib.load('model_assets/voluntariado_regresor.pkl')
loaded_scaler = joblib.load('model_assets/voluntariado_scaler.pkl')

# ejemplo
nuevos_datos_voluntario = pd.DataFrame([{
    'X1_Horas_Actuales': 30,
    'X2_Frec_Semanal': 1.0,
    'X3_Horas_Fallidas': 15,
    'X4_Semanas_Restantes': 8,
    'X5_Disp_Neta_Restante': 70, # Capacidad restante de actividades
    'X6_Antiguedad': 1
}])

# usar el scaler cargado para transformar los nuevos datos
X_nuevo_escalado = loaded_scaler.transform(nuevos_datos_voluntario)

# generar prediccion con los nuevos datos escalados
prediccion_horas_array = loaded_model.predict(X_nuevo_escalado)
y_prediccion = prediccion_horas_array[0]

print(f"Predicción Numérica del Modelo (y_prediccion): {y_prediccion:.1f} horas")

# definir umbrales para el diagnóstico
UMBRAL_RIESGO = HORAS_META_N * 0.85 # 85 horas
UMBRAL_ALERTA = HORAS_META_N * 1.00 # 100 horas

diagnostico = {
    'estado': 'Indefinido',
    'color': 'gris',
    'recomendacion_voluntario': '',
    'carga_semestral': 0 
}

# Lógica de Diagnóstico
if y_prediccion < UMBRAL_RIESGO:
    
    # condicion de riesgo (no cumple de ninguna manera)
    horas_faltantes = HORAS_META_N - y_prediccion
    
    diagnostico['estado'] = "RIESGO ESTRUCTURAL / ANOMALÍA"
    diagnostico['color'] = "ROJO"
    diagnostico['recomendacion_voluntario'] = (
        f"Alerta inmediata: Se proyecta una falta de {horas_faltantes:.1f} horas. "
        "Debe solicitar un plan de recuperación urgente y priorizar el cumplimiento de X5."
    )
    diagnostico['carga_semestre_2'] = horas_faltantes

elif y_prediccion < UMBRAL_ALERTA:

    # condicion de cuerda floja
    diagnostico['estado'] = "CUERDA FLOJA / ALERTA"
    diagnostico['color'] = "AMARILLO"
    diagnostico['recomendacion_voluntario'] = (
        "Atención: Se proyecta que quedará ligeramente por debajo de la meta. "
        "Debe asistir a TODAS las actividades remanentes posibles para cumplir. El margen de error es mínimo."
    )
    diagnostico['carga_semestre_2'] = HORAS_META_N - y_prediccion
    
else:
    # condicion normal
    horas_extra = y_prediccion - HORAS_META_N
    
    diagnostico['estado'] = "OPERACIÓN NORMAL / CUMPLIMIENTO"
    diagnostico['color'] = "VERDE"
    diagnostico['recomendacion_voluntario'] = (
        f"Felicidades: Se proyecta que superará la meta con {horas_extra:.1f} horas de colchón."
    )
    diagnostico['carga_semestre_2'] = 0

# mostrar diagnóstico
print("\n=============================================")
print(f"| Diagnóstico ML para el Voluntario: {diagnostico['estado']}")
print(f"| Estado de Riesgo: {diagnostico['color']}")
print("=============================================")
print(f"Recomendación para el Voluntario: {diagnostico['recomendacion_voluntario']}")
print(f"Proyección de Carga para Semestre 2: {diagnostico['carga_semestre_2']:.1f} horas a arrastrar.")