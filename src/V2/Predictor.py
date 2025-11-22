import xgboost as xgb
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Ruta del modelo (Asegúrate de que el archivo exista)
MODEL_PATH = 'model_assets/modelo_voluntariado_xgb.json'
META_HORAS = 100.0

def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {MODEL_PATH}")
    
    # Inicializar y cargar
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"✅ Modelo cargado exitosamente desde: {MODEL_PATH}")
    return model

def obtener_diagnostico(prediccion):
    """Define el color del semáforo según la predicción."""
    diagnostico = {
        'horas_predichas': float(prediccion),
        'estado': 'INDEFINIDO',
        'color': 'GRIS',
        'accion': ''
    }
    
    # Lógica de Negocio (Umbrales)
    # < 85 horas: Zona de Riesgo (Probablemente repruebe la beca)
    if prediccion < (META_HORAS * 0.85):
        diagnostico['estado'] = "RIESGO CRÍTICO"
        diagnostico['color'] = "ROJO"
        diagnostico['accion'] = f"ALERTA: Faltan {META_HORAS - prediccion:.1f} horas para la meta. Requiere plan de recuperación inmediato."
        
    # 85 - 99 horas: Zona de Alerta (Pasa raspando o falla por poco)
    elif prediccion < META_HORAS:
        diagnostico['estado'] = "ALERTA / CUERDA FLOJA"
        diagnostico['color'] = "AMARILLO"
        diagnostico['accion'] = "ATENCIÓN: Está al límite. No puede faltar a ninguna actividad futura."
        
    # >= 100 horas: Zona Segura
    else:
        diagnostico['estado'] = "OPERACIÓN NORMAL"
        diagnostico['color'] = "VERDE"
        diagnostico['accion'] = f"Felicidades. Se proyecta un colchón de +{prediccion - META_HORAS:.1f} horas."
        
    return diagnostico

def predecir_nuevo_estudiante(modelo, datos):
    # Convertir diccionario a DataFrame (XGBoost requiere nombres de columnas exactos)
    df_input = pd.DataFrame([datos])
    
    # Asegurar el orden de columnas con el que fue entrenado
    columnas_ordenadas = [
        'X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
        'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad', 
        'X7_Tipo_Carrera', 'X8_Beca', 'X9_Carrera_Id'
    ]
    
    # Reordenar (y filtrar si sobran datos)
    try:
        df_input = df_input[columnas_ordenadas]
    except KeyError as e:
        return f"❌ Error: Faltan columnas en los datos de entrada: {e}"

    # Predicción
    prediccion_array = modelo.predict(df_input)
    return prediccion_array[0]

# ==========================================
# EJEMPLO DE USO (SIMULACIÓN)
# ==========================================
if __name__ == "__main__":
    try:
        mi_modelo = cargar_modelo()
        
        # --- DATOS DEL ESTUDIANTE A EVALUAR ---
        # Aquí conectas tu interfaz de usuario o Excel en el futuro
        datos_alumno = {
            'X1_Horas_Actuales': 45.5,   # ¿Cuántas horas tiene hoy?
            'X2_Frec_Semanal': 1.2,      # Promedio de veces que va por semana
            'X3_Horas_Fallidas': 2.0,    # Semanas que faltó
            'X4_Semanas_Restantes': 8,   # Constante (tiempo restante)
            'X5_Disp_Neta_Restante': 50, # Constante
            'X6_Antiguedad': 2,          # Años en la U
            'X7_Tipo_Carrera': 0,        # 0: Semestral, 1: Anual
            'X8_Beca': 0.75,             # Beca del 75%
            'X9_Carrera_Id': 3           # ID de la carrera (según tu dataset)
        }
        
        print("\n--- Evaluando Estudiante ---")
        resultado_valor = predecir_nuevo_estudiante(mi_modelo, datos_alumno)
        
        diagnostico = obtener_diagnostico(resultado_valor)
        
        print(f"\n📊 PREDICCIÓN FINAL: {diagnostico['horas_predichas']:.2f} Horas")
        print(f"🚦 SEMÁFORO: {diagnostico['color']}")
        print(f"📝 ESTADO:   {diagnostico['estado']}")
        print(f"💡 ACCIÓN:   {diagnostico['accion']}")
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")