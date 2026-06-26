import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de página con título e icono
st.set_page_config(
    page_title="Predicción Accidente Ofídico - UNAB",
    page_icon="🐍",
    layout="wide"
)

# Estilos CSS personalizados para mejorar la interfaz
st.markdown("""
<style>
    .main-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #4B5563;
        font-size: 1.15rem;
        margin-bottom: 25px;
    }
    .author-footer {
        text-align: center;
        padding: 20px;
        color: #6B7280;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 40px;
    }
    .hospital-red {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 15px;
        border-radius: 4px;
        color: #991B1B;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 15px;
    }
    .home-green {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 4px;
        color: #065F46;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado Principal con Dos Columnas (Logo y Título)
col_header_1, col_header_2 = st.columns([1, 4])

with col_header_1:
    if os.path.exists("LogoUNAB.png"):
        st.image("LogoUNAB.png", width=150)
    else:
        st.info("Logo UNAB")

with col_header_2:
    st.markdown("<h1 class='main-title'>Aplicación de Calidad en el Ciclo de Vida de Aprendizaje Automático</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='subtitle'>Propuesta predictiva aplicada al accidente ofídico en Colombia</h4>", unsafe_allow_html=True)

# Imagen ilustrativa relacionada con el tema
st.image("https://asosec.co/wp-content/uploads/2019/05/mordedura-serpiente.jpg", 
         caption="Monitoreo y vigilancia del accidente ofídico - Modelo de Soporte a la Decisión Sanitaria", 
         use_container_width=True)

st.markdown("---")

# Carga de modelos y artefactos de forma segura empleando un bloque try-except
@st.cache_resource
def load_model_and_artifacts():
    try:
        model = joblib.load('random_forest_optimized_model.joblib')
        artifacts = joblib.load('transformation_pipeline_artifacts.joblib')
        return model, artifacts
    except Exception as e:
        st.error(f"Error al cargar los archivos del modelo: {e}")
        st.info("Por favor, asegúrese de que 'random_forest_optimized_model.joblib' y 'transformation_pipeline_artifacts.joblib' estén en la misma carpeta que este script.")
        return None, None

loaded_model, loaded_artifacts = load_model_and_artifacts()

if loaded_model is not None and loaded_artifacts is not None:
    
    st.markdown("### 📋 Formulario de Evaluación Epidemiológica del Paciente")
    st.write("Ingrese los datos correspondientes al caso notificado para estimar el requerimiento de hospitalización.")
    
    # Organizar el formulario en columnas para mejorar la estética visual
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad del Paciente:", min_value=0, max_value=120, value=30, step=1)
        sexo = st.selectbox("Sexo:", ['F', 'M'])
        etnia = st.selectbox("Etnia:", ['Otros', 'Indigena', 'Afro'])
        regimen_salud = st.selectbox("Régimen de Salud:", ['Contributivo', 'Subsidiado', 'No Asegurado'])
        departamento = st.selectbox("Departamento del Evento:", ['Antioquia', 'Amazonas', 'Chocó', 'Meta', 'Casanare', 'Otros'])

    with col2:
        area_ocurrencia = st.selectbox("Área de Ocurrencia:", ['Cabecera Municipal', 'Rural Disperso', 'Centro Poblado'])
        actividad = st.selectbox("Actividad en el Momento del Accidente:", ['Doméstico', 'Agricultura', 'Tránsito por trocha', 'Recreación'])
        genero_serpiente = st.selectbox("Género de la Serpiente:", ['Bothrops', 'Lachesis', 'Crotalus', 'Micrurus', 'Desconocido'])
        localizacion_mordedura = st.selectbox("Localización Anatómica de la Mordedura:", ['Miembros Inferiores', 'Miembros Superiores', 'Tronco/Cabeza'])

    with col3:
        tiempo_atencion_h = st.number_input("Tiempo de atención transcurrido (Horas):", min_value=0.0, max_value=168.0, value=2.5, step=0.5)
        medicina_tradicional = st.checkbox("¿Se empleó medicina tradicional previa?")
        edema = st.checkbox("¿Presenta Edema?")
        sangrado = st.checkbox("¿Presenta Sangrado?")
        flictenas = st.checkbox("¿Presenta Flictenas?")
        suero_previo = st.checkbox("¿Recibió suero previo a la atención?")

    # Botón de Procesamiento de la Predicción
    if st.button("📊 Estimar Probabilidad de Hospitalización", type="primary"):
        
        # 1. Estructurar los datos en un DataFrame igual al entrenamiento
        input_data = pd.DataFrame({
            'edad': [edad],
            'sexo': [sexo],
            'etnia': [etnia],
            'regimen_salud': [regimen_salud],
            'departamento': [departamento],
            'area_ocurrencia': [area_ocurrencia],
            'actividad': [actividad],
            'genero_serpiente': [genero_serpiente],
            'localizacion_mordedura': [localizacion_mordedura],
            'tiempo_atencion_h': [tiempo_atencion_h],
            'medicina_tradicional': [1 if medicina_tradicional else 0],
            'edema': [1 if edema else 0],
            'sangrado': [1 if sangrado else 0],
            'flictenas': [1 if flictenas else 0],
            'suero_previo': [1 if suero_previo else 0]
        })
        
        # 2. Ingeniería de Características basada en los artefactos cargados
        input_data['edad_categoria'] = pd.cut(
            input_data['edad'], 
            bins=loaded_artifacts['edad_bins'], 
            labels=loaded_artifacts['edad_labels'], 
            right=False
        )
        
        # Adaptar dinámicamente el valor máximo del tiempo de atención en caso de superar el original
        current_bins_tiempo = list(loaded_artifacts['tiempo_atencion_bins'])
        if tiempo_atencion_h >= current_bins_tiempo[-1]:
            current_bins_tiempo[-1] = tiempo_atencion_h + 1
            
        input_data['tiempo_atencion_categoria'] = pd.cut(
            input_data['tiempo_atencion_h'], 
            bins=current_bins_tiempo, 
            labels=loaded_artifacts['tiempo_atencion_labels'], 
            right=False
        )
        
        # 3. Aplicar One-Hot Encoding
        input_encoded = pd.get_dummies(input_data, columns=loaded_artifacts['ohe_categorical_cols'])
        
        # Reindexar con las columnas del X_train original
        input_processed = input_encoded.reindex(columns=loaded_artifacts['model_input_columns'], fill_value=0)
        
        # 4. Realizar Predicción
        prediccion = loaded_model.predict(input_processed)[0]
        probabilidad = loaded_model.predict_proba(input_processed)[0][1]
        
        # 5. Visualización Estilizada de Resultados
        st.markdown("### 📢 Resultado de la Evaluación Predictiva")
        st.write(f"**Probabilidad estimada de hospitalización:** `{probabilidad * 100:.2f}%`")
        
        if prediccion == 1:
            st.markdown(
                f"<div class='hospital-red'>⚠️ ALERTA: EL MODELO SUGIERE QUE EL PACIENTE REQUIERE HOSPITALIZACIÓN (Probabilidad alta: {probabilidad*100:.1f}%)</div>", 
                unsafe_allow_html=True
            )
            st.warning("**Acción Epidemiológica Inmediata:** Priorizar la canalización y el traslado seguro a una institución de salud que cuente con disponibilidad de suero antiofídico específico y capacidad de monitoreo de complicaciones sistémicas.")
        else:
            st.markdown(
                f"<div class='home-green'>✅ EL MODELO SUGIERE QUE EL PACIENTE NO REQUIERE HOSPITALIZACIÓN (Probabilidad baja: {probabilidad*100:.1f}%)</div>", 
                unsafe_allow_html=True
            )
            
            # Indicaciones de Manejo solicitadas
            st.markdown("""
            #### 🩺 Indicaciones de Manejo Clínico Preventivo y Ambulatorio:
            1. **Reposo Absoluto e Inmovilización:** Mantener la extremidad afectada inmóvil y a un nivel funcional (ligeramente elevado o neutro), evitando el movimiento para reducir la tasa de absorción del veneno residual.
            2. **Limpieza e Higiene Local:** Lavar la zona afectada minuciosamente con abundante agua y jabón neutro. No aplicar emplastos, ungüentos tradicionales, ni realizar incisiones o torniquetes.
            3. **Control y Monitoreo de Signos de Alarma:** Evaluar continuamente la evolución del edema local, aparición de equimosis, dolor intenso o sangrados espontáneos (gingivorragia, hematuria).
            4. **Hidratación y Soporte:** Mantener al paciente adecuadamente hidratado por vía oral si no presenta compromiso neurológico o emesis. En caso de dolor, administrar analgésicos formulados (evitar AINEs si hay riesgo hemorrágico latente).
            5. **Educación e Instrucción:** Informar rigurosamente al paciente y a sus familiares que ante el menor indicio de complicación o empeoramiento clínico, deben acudir de inmediato al centro asistencial de salud más cercano.
            """)

# Pie de página autoral con el formato solicitado
st.markdown(
    "<div class='author-footer'>Realizado por Alfredo Diaz, UNAB 2026<br>© Todos los derechos reservados</div>", 
    unsafe_allow_html=True
)
