import streamlit as st
import xgboost as xgb
import pandas as pd
import os
import numpy as np

# ============================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="SPAT - Predicción Voluntariado",
    page_icon="🎓",
    layout="wide"
)

# Título y Descripción
st.title("🎓 SPAT: Sistema de Alerta Temprana")
st.markdown("""
Este sistema utiliza **Inteligencia Artificial (XGBoost)** para predecir si un estudiante 
logrará cumplir su meta de voluntariado basándose en su desempeño actual.
""")

# ============================================
# 2. CARGA DEL MODELO (CON CACHÉ)
# ============================================
MODEL_PATH = 'model_assets/modelo_voluntariado_xgb.json'
META_HORAS = 100.0

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        return None
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

model = cargar_modelo()

# ============================================
# 3. INTERFAZ DE ENTRADA (SIDEBAR)
# ============================================
st.sidebar.header("📝 Datos del Estudiante")
st.sidebar.info("Ingrese los datos actuales del corte:")

# Variables de Entrada
x1 = st.sidebar.number_input("X1: Horas Acumuladas Actuales", min_value=0.0, max_value=200.0, value=45.5)
x2 = st.sidebar.number_input("X2: Frecuencia Semanal (Días/Semana)", min_value=0.0, max_value=7.0, value=1.2)
x3 = st.sidebar.number_input("X3: Semanas Fallidas (Inasistencias)", min_value=0.0, max_value=20.0, value=2.0)
x6 = st.sidebar.slider("X6: Antigüedad (Años)", 0, 10, 2)

# Selectores amigables (que luego convertimos a números)
tipo_carrera_txt = st.sidebar.selectbox("X7: Tipo de Carrera", ["Semestral", "Anual"])
x7 = 0 if tipo_carrera_txt == "Semestral" else 1

beca_txt = st.sidebar.selectbox("X8: Porcentaje de Beca", ["50%","70%","90%","100%"])
beca_map = {"50%": 0.50, "70%": 0.70, "90%": 0.90, "100%": 1.0}
x8 = beca_map[beca_txt]

# IMPORTANTE: Aquí debes mapear tus carreras según como quedaron en el entrenamiento
# Si tienes el CSV, revisa qué número le tocó a cada carrera.
# Este es un ejemplo genérico, AJÚSTALO a tus IDs reales.
st.sidebar.markdown("---")
st.sidebar.markdown("**Configuración Académica**")
carrera_id = st.sidebar.number_input("X9: ID de Carrera (Numérico)", min_value=0, max_value=50, value=3, help="Ingrese el ID numérico asignado a la carrera en el dataset de entrenamiento.")

# Variables Constantes (Ocultas en un expander para no ensuciar)
with st.sidebar.expander("⚙️ Variables Constantes (Avanzado)"):
    x4 = st.number_input("X4: Semanas Restantes", value=8)
    x5 = st.number_input("X5: Disponibilidad Neta", value=50)

# ============================================
# 4. LÓGICA DE PREDICCIÓN
# ============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=150) # Icono decorativo
    boton_calcular = st.button("🔍 CALCULAR RIESGO", type="primary", use_container_width=True)

if boton_calcular and model:
    # Preparar DataFrame
    datos = pd.DataFrame([{
        'X1_Horas_Actuales': x1,
        'X2_Frec_Semanal': x2,
        'X3_Horas_Fallidas': x3,
        'X4_Semanas_Restantes': x4,
        'X5_Disp_Neta_Restante': x5,
        'X6_Antiguedad': x6,
        'X7_Tipo_Carrera': x7,
        'X8_Beca': x8,
        'X9_Carrera_Id': carrera_id
    }])
    
    # Predecir
    prediccion = model.predict(datos)[0]
    
    # Lógica del Semáforo (Tu lógica exacta)
    if prediccion < (META_HORAS * 0.85):
        estado = "RIESGO CRÍTICO"
        color = "red"
        mensaje = f"⚠️ ALERTA: Faltan {META_HORAS - prediccion:.1f} horas. Requiere plan urgente."
        icono = "🔴"
    elif prediccion < META_HORAS:
        estado = "ALERTA / CUERDA FLOJA"
        color = "orange" # Streamlit usa orange para amarillo/alerta
        mensaje = "⚠️ ATENCIÓN: Está al límite. No puede faltar."
        icono = "🟡"
    else:
        estado = "OPERACIÓN NORMAL"
        color = "green"
        mensaje = f"✅ Felicidades. Colchón de +{prediccion - META_HORAS:.1f} horas."
        icono = "🟢"

    # ============================================
    # 5. VISUALIZACIÓN DE RESULTADOS
    # ============================================
    with col2:
        st.subheader("Diagnóstico del Modelo")
        
        # Tarjetas Métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("Horas Proyectadas", f"{prediccion:.1f} h", delta=f"{prediccion-META_HORAS:.1f} vs Meta")
        m2.metric("Estado", estado)
        m3.metric("Probabilidad Cumplimiento", f"{(prediccion/META_HORAS)*100:.0f}%")
        
        st.divider()
        
        # Mensaje Grande (Semáforo)
        if color == "red":
            st.error(f"### {icono} {estado}\n{mensaje}")
        elif color == "orange":
            st.warning(f"### {icono} {estado}\n{mensaje}")
        else:
            st.success(f"### {icono} {estado}\n{mensaje}")
            
        # Barra de Progreso Visual
        progreso = min(prediccion / 120.0, 1.0) # Tope visual en 120 horas
        st.write("Proyección Visual:")
        st.progress(progreso)
        st.caption(f"La barra muestra la proyección respecto a una sobre-meta de 120 horas.")

else:
    with col2:
        st.info("👈 Configure los datos del estudiante en el menú lateral y presione 'CALCULAR RIESGO'.")