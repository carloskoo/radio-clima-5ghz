import pandas as pd
import glob

# ==========================================
# 1. BUSCAR SOLO LOS ARCHIVOS VALIDOS
# ==========================================

archivos = glob.glob("rssi_snr_SM_*.csv")
archivos = [f for f in archivos if "_old" not in f]

print("Archivos usados:", len(archivos))

if len(archivos) == 0:
    raise ValueError("No se encontraron archivos válidos con patrón rssi_snr_SM_*.csv")

# ==========================================
# 2. LEER Y UNIR TODOS LOS CSV
# ==========================================

df_list = []

for archivo in archivos:
    try:
        temp = pd.read_csv(archivo, sep=';')
        df_list.append(temp)
    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")

if len(df_list) == 0:
    raise ValueError("No se pudo leer ningún archivo CSV")

df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 3. NORMALIZAR NOMBRES DE COLUMNAS
# ==========================================

df.columns = df.columns.str.strip().str.lower()
print("Columnas detectadas:", df.columns.tolist())

columnas_esperadas = ['timestamp', 'dl_rssi', 'dl_snr', 'ul_rssi', 'ul_snr']
faltantes = [c for c in columnas_esperadas if c not in df.columns]

if faltantes:
    raise ValueError(f"Faltan columnas esperadas: {faltantes}")

# ==========================================
# 4. CONVERTIR TIPOS
# ==========================================

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'] = df['timestamp'].dt.tz_localize(None)

for col in ['dl_rssi', 'dl_snr', 'ul_rssi', 'ul_snr']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ==========================================
# 5. ELIMINAR NULOS
# ==========================================

df = df.dropna(subset=['timestamp', 'dl_rssi', 'dl_snr', 'ul_rssi', 'ul_snr'])

# ==========================================
# 6. ELIMINAR PERIODO DE DESALINEACION
# ==========================================

inicio_desalineacion = pd.Timestamp("2025-10-07 00:00:00")
fin_desalineacion = pd.Timestamp("2025-11-15 23:59:59")

df = df[
    (df['timestamp'] < inicio_desalineacion) |
    (df['timestamp'] > fin_desalineacion)
]

# ==========================================
# 7. FILTROS DE CONSISTENCIA
# ==========================================

df = df[
    (df['dl_rssi'] != 0) &
    (df['dl_snr'] != 0) &
    (df['ul_rssi'] != 0) &
    (df['ul_snr'] != 0)
]

df = df[
    (df['dl_rssi'] > -100) & (df['dl_rssi'] < -50) &
    (df['ul_rssi'] > -100) & (df['ul_rssi'] < -50)
]

df = df[
    (df['dl_snr'] > 10) & (df['dl_snr'] < 40) &
    (df['ul_snr'] > 10) & (df['ul_snr'] < 40)
]

# ==========================================
# 8. ORDEN Y DUPLICADOS
# ==========================================

df = df.sort_values(by='timestamp')
df = df.drop_duplicates(subset=['timestamp'])

# ==========================================
# 9. VARIABLES DERIVADAS
# ==========================================

df['rssi_mean'] = (df['dl_rssi'] + df['ul_rssi']) / 2
df['snr_mean'] = (df['dl_snr'] + df['ul_snr']) / 2
df['rssi_diff'] = df['dl_rssi'] - df['ul_rssi']
df['snr_diff'] = df['dl_snr'] - df['ul_snr']

# ==========================================
# 10. CLASIFICACION TEMPORAL
# ==========================================

def clasificar_periodo(fecha):
    if fecha.month == 9:
        return "seco"
    elif fecha.month in [10, 11]:
        return "transicion"
    else:
        return "humedo"

df['periodo'] = df['timestamp'].apply(clasificar_periodo)

# ==========================================
# 11. EXPORTAR
# ==========================================

df.to_csv("dataset_limpio_radio.csv", index=False)

df_model = df[
    ['timestamp', 'periodo', 'dl_rssi', 'dl_snr', 'ul_rssi', 'ul_snr',
     'rssi_mean', 'snr_mean', 'rssi_diff', 'snr_diff']
]
df_model.to_csv("dataset_modelado_radio.csv", index=False)

print("\n✔ Dataset limpio generado correctamente")
print("Filas finales:", len(df))
print(df.head())
