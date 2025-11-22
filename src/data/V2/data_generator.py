import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
CUTOFF_DATE = '2025-04-15' 
TOTAL_SAMPLES_DESEADOS = 350 
SEMILLA_RANDOM = 42 
np.random.seed(SEMILLA_RANDOM)

def procesar_excel_maestro():
    print("--- PASO 1: Buscando archivo Excel (.xlsx) ---")
    excel_files = glob.glob("*.xlsx")
    
    if not excel_files:
        raise FileNotFoundError("❌ No encontré ningún archivo .xlsx en esta carpeta.")
    
    target_file = excel_files[0]
    print(f"-> Procesando archivo: {target_file}")
    
    # Leer todas las hojas
    dict_sheets = pd.read_excel(target_file, sheet_name=None)
    
    # Identificar hoja REPORTE
    sheet_reporte = None
    for sheet_name in dict_sheets.keys():
        if "REPORTE" in sheet_name.upper():
            sheet_reporte = sheet_name
            break
            
    if not sheet_reporte:
        raise ValueError("❌ No encontré la hoja 'REPORTE'.")
    
    print(f"-> Hoja Maestra: '{sheet_reporte}'")
    
    # Procesar Reporte
    df_rep = dict_sheets[sheet_reporte]
    # Mapeo de columnas (Asegúrate que CARRERA exista en el Excel)
    df_rep.rename(columns={'%': 'Beca', 'TIPO CARRERA': 'Tipo', 
                           'ESTUDIANTE': 'Nombre', 'REG': 'Registro',
                           'CARRERA': 'Carrera_Nombre'}, inplace=True)
    
    df_rep['MatchName'] = df_rep['Nombre'].astype(str).str.upper().str.strip()
    
    real_data_rows = []
    
    # Iterar hojas de estudiantes
    for sheet_name, df_raw in dict_sheets.items():
        if sheet_name == sheet_reporte: continue
            
        try:
            # Buscar cabecera dinámicamente
            header_idx = -1
            for i in range(min(20, len(df_raw))):
                row_vals = df_raw.iloc[i].astype(str).values
                if any("FECHA" in val.upper() for val in row_vals):
                    header_idx = i
                    break
            
            if header_idx == -1: continue
                
            df_raw.columns = df_raw.iloc[header_idx]
            df_data = df_raw.iloc[header_idx+1:].copy()
            df_data.columns = [str(c).upper().strip() for c in df_data.columns]
            
            if 'FECHA' not in df_data.columns or 'HORAS' not in df_data.columns: continue

            df_data['FECHA'] = pd.to_datetime(df_data['FECHA'], errors='coerce')
            df_data = df_data.dropna(subset=['FECHA'])
            if df_data.empty: continue
            
            # Filtro de Corte (Pasado)
            df_past = df_data[df_data['FECHA'] <= pd.to_datetime(CUTOFF_DATE)]
            
            # Variables Calculadas
            x1_horas = pd.to_numeric(df_past['HORAS'], errors='coerce').sum()
            
            weeks_passed = 12
            x2_frec = df_past['FECHA'].nunique() / weeks_passed
            
            if not df_past.empty:
                active_weeks = df_past['FECHA'].dt.isocalendar().week.nunique()
                x3_fallas = max(0, weeks_passed - active_weeks)
            else:
                x3_fallas = weeks_passed
            
            y_final = pd.to_numeric(df_data['HORAS'], errors='coerce').sum()
            match_name = sheet_name.replace("BECA CEIL", "").replace(".xlsx", "").upper().strip()
            
            real_data_rows.append({
                'MatchName': match_name,
                'X1_Horas_Actuales': x1_horas,
                'X2_Frec_Semanal': x2_frec,
                'X3_Horas_Fallidas': x3_fallas,
                'Y_Horas_Totales_Finales': y_final
            })
            
        except Exception: continue

    # Unir con Metadata del Reporte
    df_features = pd.DataFrame(real_data_rows)
    df_merged = pd.merge(df_features, df_rep[['MatchName', 'Tipo', 'Beca', 'Registro', 'Carrera_Nombre']], 
                         on='MatchName', how='left')
    
    # Rellenar nulos por defecto
    df_merged['Tipo'] = df_merged['Tipo'].fillna('SEMESTRAL')
    df_merged['Beca'] = df_merged['Beca'].fillna(0.5)
    df_merged['Carrera_Nombre'] = df_merged['Carrera_Nombre'].fillna('DESCONOCIDA')
    
    # --- INGENIERÍA DE VARIABLES ---
    current_year = 2025
    
    # X6 Antiguedad
    df_merged['X6_Antiguedad'] = df_merged['Registro'].apply(
        lambda x: current_year - int(str(x)[:4]) if pd.notnull(x) and str(x)[:4].isdigit() else 0
    )
    # X7 Tipo Carrera (1=Anual, 0=Semestral)
    df_merged['X7_Tipo_Carrera'] = df_merged['Tipo'].apply(lambda x: 1 if 'ANUAL' in str(x).upper() else 0)
    # X8 Beca
    df_merged['X8_Beca'] = df_merged['Beca']
    # X9 Carrera ID (Convertir texto a numero para XGBoost)
    # pd.factorize devuelve una tupla (array_codigos, index_unicos). Tomamos [0]
    df_merged['X9_Carrera_Id'] = pd.factorize(df_merged['Carrera_Nombre'])[0]
    
    # Constantes
    df_merged['X4_Semanas_Restantes'] = 8 
    df_merged['X5_Disp_Neta_Restante'] = 50.0
    
    print(f"-> Registros Reales Procesados: {len(df_merged)}")
    return df_merged

def generar_sinteticos(df_real):
    print("\n--- PASO 2: Data Augmentation (Mixup) ---")
    n_reales = len(df_real)
    n_a_generar = TOTAL_SAMPLES_DESEADOS - n_reales
    
    if n_a_generar <= 0: return df_real
    
    synthetic_samples = []
    # Solo interpolamos variables continuas numéricas
    cols_continuas = ['X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 'Y_Horas_Totales_Finales']
    
    for _ in range(n_a_generar):
        idx1, idx2 = np.random.choice(df_real.index, 2, replace=False)
        p1, p2 = df_real.loc[idx1], df_real.loc[idx2]
        lam = np.random.uniform(0.2, 0.8)
        
        child = {}
        # Interpolación con ruido
        for col in cols_continuas:
            val = p1[col]*lam + p2[col]*(1-lam)
            noise = np.random.normal(0, 0.02*val) if val > 0 else 0
            child[col] = max(0, val + noise)
            
        # Herencia Categórica (Carrera, Beca, Antiguedad se heredan del padre dominante)
        dom = p1 if lam > 0.5 else p2
        
        child['X4_Semanas_Restantes'] = 8
        child['X5_Disp_Neta_Restante'] = 50.0
        child['X6_Antiguedad'] = dom['X6_Antiguedad']
        child['X7_Tipo_Carrera'] = dom['X7_Tipo_Carrera']
        child['X8_Beca'] = dom['X8_Beca']
        child['X9_Carrera_Id'] = dom['X9_Carrera_Id'] # Hereda la carrera
        
        synthetic_samples.append(child)
        
    return pd.concat([df_real, pd.DataFrame(synthetic_samples)], ignore_index=True)

if __name__ == "__main__":
    try:
        df_base = procesar_excel_maestro()
        
        # Definir columnas finales
        cols = ['X1_Horas_Actuales', 'X2_Frec_Semanal', 'X3_Horas_Fallidas', 
                'X4_Semanas_Restantes', 'X5_Disp_Neta_Restante', 'X6_Antiguedad', 
                'X7_Tipo_Carrera', 'X8_Beca', 'X9_Carrera_Id', 
                'Y_Horas_Totales_Finales']
        
        df_final = generar_sinteticos(df_base[cols])
        
        archivo_salida = 'dataset_voluntariado_aumentado.csv'
        df_final.to_csv(archivo_salida, index=False)
        
        print(f"\n✅ ¡ÉXITO! Dataset generado: {archivo_salida}")
        print(f"Total Muestras: {len(df_final)}")
        print(f"Variables incluidas: {cols}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")