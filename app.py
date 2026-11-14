from pathlib import Path
import joblib
import sklearn
import streamlit as st
from inferencia import construir_entrada, predecir

st.set_page_config(page_title="Predicción de riesgo académico", page_icon="🎓", layout="centered")
st.title("🎓 Riesgo académico")
st.write("Ingresa los datos del estudiante para obtener la predicción del Random Forest seleccionado.")

@st.cache_resource
def cargar_modelo():
    return joblib.load(Path(__file__).with_name("modelo_rf.pkl"))

try:
    paquete = cargar_modelo()
except FileNotFoundError:
    st.info("Falta modelo_rf.pkl. Ejecuta exportar_modelo.py como celda al final de tu notebook y coloca el archivo generado junto a app.py.")
    st.stop()
except Exception as error:
    st.error(f"No se pudo cargar el archivo del modelo: {error}")
    st.stop()

requeridas = {"modelo", "scaler", "columnas", "campos", "generos", "educacion", "sklearn_version"}
if not isinstance(paquete, dict) or not requeridas.issubset(paquete):
    st.error("El archivo contiene un modelo sin sus datos de preparación. Vuelve a exportarlo con la celda proporcionada.")
    st.stop()
if sklearn.__version__ != paquete["sklearn_version"]:
    st.error(f"Instala la versión del entrenamiento: pip install scikit-learn=={paquete['sklearn_version']}")
    st.stop()

st.caption("Modelo: Random Forest inicial · Entrenamiento con SMOTE")
with st.form("prediccion"):
    st.subheader("Datos del estudiante")
    izquierda, derecha = st.columns(2)
    genero = izquierda.selectbox("Género", paquete["generos"])
    traduccion = {"High School": "Secundaria", "College": "College", "University": "Universidad"}
    educacion = derecha.selectbox("Nivel educativo", list(paquete["educacion"]), format_func=lambda x: traduccion.get(x, x))
    valores = {}
    contenedores = st.columns(2)
    for i, (nombre, campo) in enumerate(paquete["campos"].items()):
        convertir = int if campo["entero"] else float
        minimo, maximo, defecto = [convertir(campo[k]) for k in ("min", "max", "default")]
        with contenedores[i % 2]:
            if minimo == maximo:
                st.caption(f"{nombre.replace('_', ' ')}: {minimo} (constante)")
                valores[nombre] = minimo
            else:
                valores[nombre] = st.number_input(nombre.replace("_", " "), min_value=minimo, max_value=maximo,
                    value=defecto, step=1 if campo["entero"] else 0.1,
                    help=f"Rango observado en el dataset: {minimo} a {maximo}")
    enviar = st.form_submit_button("Predecir riesgo", type="primary", use_container_width=True)

if enviar:
    try:
        entrada = construir_entrada(paquete, valores, genero, educacion)
        clase, probabilidad = predecir(paquete, entrada)
        st.subheader("Resultado de la predicción")
        if clase == 1:
            st.warning("El modelo predice riesgo de fracaso académico.")
        else:
            st.success("El modelo predice ausencia de riesgo de fracaso académico.")
        st.metric("Estimación del modelo para la clase de riesgo", f"{probabilidad:.1%}")
        st.caption("Este valor no es una probabilidad calibrada ni una certeza sobre el estudiante. Es una estimación del clasificador entrenado con SMOTE.")
        with st.expander("Ver datos ingresados"):
            st.dataframe(entrada, hide_index=True)
    except Exception as error:
        st.error(f"No fue posible realizar la predicción: {error}")
st.caption("Proyecto académico. El resultado puede contener errores y debe interpretarse junto con las métricas de evaluación.")
