import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Configuración inicial
# --------------------------------------------------

st.set_page_config(
    page_title="Predictor de Mora - ConecTel",
    page_icon="📊",
    layout="centered"
)

st.title("Predictor de Riesgo de Morosidad")
st.subheader("ConecTel S.A.")

st.write(
    """
    Esta aplicación estima la probabilidad de que un cliente caiga en mora severa 
    mayor a 90 días. El resultado debe utilizarse como apoyo para priorizar acciones 
    preventivas de cobranza y gestión comercial.
    """
)

# --------------------------------------------------
# Cargar modelo
# --------------------------------------------------

@st.cache_resource
def cargar_modelo():
    paquete = joblib.load("modelo_conectel.joblib")
    return paquete

paquete_modelo = cargar_modelo()

modelo = paquete_modelo["modelo"]
umbral_final = paquete_modelo["umbral_final"]
columnas_modelo = paquete_modelo["columnas_modelo"]

# --------------------------------------------------
# Formulario de entrada
# --------------------------------------------------

st.markdown("## Datos del cliente")

with st.form("formulario_cliente"):

    region = st.selectbox(
        "Región",
        [
            "Metropolitana",
            "Valparaíso",
            "Biobío",
            "Coquimbo",
            "Maule",
            "La Araucanía",
            "Los Lagos",
            "Antofagasta"
        ]
    )

    genero = st.selectbox(
        "Género",
        ["Femenino", "Masculino", "Otro"]
    )

    tipo_contrato = st.selectbox(
        "Tipo de contrato",
        ["Mensual", "Anual", "Bianual"]
    )
    

    plan = st.selectbox(
        "Plan contratado",
        ["Básico", "Estándar", "Premium"]
    )

    metodo_pago = st.selectbox(
        "Método de pago",
        [
            "Débito automático",
            "WebPay",
            "Transferencia",
            "Efectivo",
            "Cheque"
        ]
    )

    descuento_activo = st.selectbox(
        "¿Tiene descuento activo?",
        ["Sí", "No"]
    )

    edad = st.number_input(
        "Edad",
        min_value=18,
        max_value=100,
        value=35
    )

    antiguedad_meses = st.number_input(
        "Antigüedad en meses",
        min_value=0,
        max_value=300,
        value=24
    )

    tiene_internet = st.selectbox(
        "¿Tiene internet?",
        [1, 0],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    velocidad_mbps = st.number_input(
        "Velocidad de internet Mbps",
        min_value=0,
        max_value=1000,
        value=300
    )

    tiene_tv = st.selectbox(
        "¿Tiene TV?",
        [1, 0],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    tiene_linea_movil = st.selectbox(
        "¿Tiene línea móvil?",
        [1, 0],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

    num_servicios = st.number_input(
        "Número de servicios contratados",
        min_value=1,
        max_value=3,
        value=2
    )

    factura_mensual_clp = st.number_input(
        "Factura mensual CLP",
        min_value=0,
        value=45000,
        step=1000
    )

    dias_mora_hist = st.number_input(
        "Días de mora histórica",
        min_value=0,
        max_value=365,
        value=0
    )

    reclamos_12m = st.number_input(
        "Reclamos en los últimos 12 meses",
        min_value=0,
        max_value=50,
        value=0
    )

    llamadas_soporte_6m = st.number_input(
        "Llamadas a soporte en los últimos 6 meses",
        min_value=0,
        max_value=100,
        value=1
    )

    nps = st.number_input(
        "NPS",
        min_value=1,
        max_value=10,
        value=7
    )

    meses_sin_reajuste = st.number_input(
        "Meses sin reajuste",
        min_value=0,
        max_value=120,
        value=12
    )

    ingreso_estimado_clp = st.number_input(
        "Ingreso estimado CLP",
        min_value=1,
        value=700000,
        step=10000
    )

    cambios_plan_12m = st.number_input(
        "Cambios de plan en los últimos 12 meses",
        min_value=0,
        max_value=20,
        value=0
    )

    boton_predecir = st.form_submit_button("Predecir riesgo de mora")

# --------------------------------------------------
# Predicción
# --------------------------------------------------

if boton_predecir:

    # Variables derivadas
    ratio_factura_ingreso = factura_mensual_clp / ingreso_estimado_clp
    indice_conflictividad = reclamos_12m + llamadas_soporte_6m
    mora_hist_binaria = 1 if dias_mora_hist > 0 else 0
    nps_bajo = 1 if nps <= 5 else 0

    if antiguedad_meses <= 12:
        antiguedad_categoria = "Nueva"
    elif antiguedad_meses <= 36:
        antiguedad_categoria = "Media"
    else:
        antiguedad_categoria = "Antigua"

    # Crear registro del cliente
    cliente = pd.DataFrame([{
        "region": region,
        "edad": edad,
        "genero": genero,
        "tipo_contrato": tipo_contrato,
        "antiguedad_meses": antiguedad_meses,
        "plan": plan,
        "tiene_internet": tiene_internet,
        "velocidad_mbps": velocidad_mbps,
        "tiene_tv": tiene_tv,
        "tiene_linea_movil": tiene_linea_movil,
        "num_servicios": num_servicios,
        "factura_mensual_clp": factura_mensual_clp,
        "metodo_pago": metodo_pago,
        "dias_mora_hist": dias_mora_hist,
        "reclamos_12m": reclamos_12m,
        "llamadas_soporte_6m": llamadas_soporte_6m,
        "nps": nps,
        "descuento_activo": descuento_activo,
        "meses_sin_reajuste": meses_sin_reajuste,
        "ingreso_estimado_clp": ingreso_estimado_clp,
        "cambios_plan_12m": cambios_plan_12m,
        "ratio_factura_ingreso": ratio_factura_ingreso,
        "indice_conflictividad": indice_conflictividad,
        "mora_hist_binaria": mora_hist_binaria,
        "nps_bajo": nps_bajo,
        "antiguedad_categoria": antiguedad_categoria
    }])

    # Ordenar columnas según entrenamiento
    cliente = cliente[columnas_modelo]

    # Predicción
    probabilidad_mora = modelo.predict_proba(cliente)[:, 1][0]
    prediccion = int(probabilidad_mora >= umbral_final)

    # Categoría de riesgo
    if probabilidad_mora < 0.30:
        categoria = "Bajo"
    elif probabilidad_mora < 0.60:
        categoria = "Medio"
    else:
        categoria = "Alto"

    st.markdown("## Resultado de la predicción")

    st.metric(
        label="Probabilidad estimada de mora",
        value=f"{probabilidad_mora * 100:.2f}%"
    )

    st.write(f"**Categoría de riesgo:** {categoria}")

    if prediccion == 1:
        st.error("Resultado: Cliente clasificado con riesgo de mora severa.")
    else:
        st.success("Resultado: Cliente clasificado sin riesgo severo según el umbral definido.")

    st.caption(
        f"Umbral utilizado para clasificar mora: {umbral_final:.2f}. "
        "La categoría bajo/medio/alto se calcula sobre la probabilidad estimada."
    )
# ---------------------------------
 # Local URL: http://localhost:8501
 # Network URL: http://172.20.10.2:8501