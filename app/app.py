"""
🏦 Evaluación de Diabetes App - Demo de ML en Producción
================================================
Aplicación de Streamlit para demostrar cómo desplegar un modelo de ML.
Curso: Machine Learning Supervisado - PECD UNI

Autores: Sarai Cisneros y George Urbina
Fecha: Diciembre 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Evaluación de Diabetes - PECD UNI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# ESTILOS CSS PERSONALIZADOS
# ============================================
st.markdown(
    """
<style>
    /* Fondo del header */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1,.main-header h3, .main-header h4 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header p {
        color: #b8d4e8;
        text-align: center;
        margin: 5px 0 0 0;
    }
    
    /* Badges */
    .badge-approved {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    .badge-rejected {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    .badge-review {
        background-color: #ffc107;
        color: black;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 20px;
        font-size: 0.9em;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================
# Las variables seleccionadas para la demo
FEATURE_CONFIG = {
    "Pregnancies": {
        "label": "🤰🏻 Número de embarazos",
        "description": "Número de veces que quedó embarazada",
        "min": 0,
        "max": 20,
        "default": 0,
        "step": 1,
        "help": "",
    },
    "Glucose": {
        "label": "🩸 Glucosa",
        "description": "Concentración de glucosa plasmática a las 2 horas en una prueba de tolerancia a la glucosa oral",
        "min": 0,
        "max": 600,
        "default": 90,
        "step": 1,
        "help": "Porcentaje de 0 a 1. Valores cercanos a 1 indican uso excesivo del sobregiro.",
    },
    "BloodPressure": {
        "label": "❤️ Presión Arterial",
        "description": "Presión arterial diastólica (mm Hg)",
        "min": 0,
        "max": 250,
        "default": 80,
        "step": 1,
        "help": "",
    },
    "SkinThickness": {
        "label": "🩹 Grosor de Piel",
        "description": "Espesor del pliegue cutáneo del tríceps (mm)",
        "min": 0,
        "max": 100,
        "default": 20,
        "step": 1,
        "help": "",
    },
    "Insulin": {
        "label": "💉 Insulina",
        "description": "Insulina sérica de 2 horas (mu U/ml)",
        "min": 0,
        "max": 1000,
        "default": 25,
        "step": 1,
        "help": "",
    },
    "BMI": {
        "label": "✏️ Índice de Masa Corporal",
        "description": "ndice de masa corporal (peso en kg/(altura en m)^2)",
        "min": 0.0,
        "max": 90.0,
        "default": 18.5,
        "step": 0.1,
        "help": "",
    },
    "DiabetesPedigreeFunction": {
        "label": "👨🏻‍👩🏻‍👧🏻 Índice de Antecedentes Familiares de Diabetes",
        "description": "Función del pedigrí de la diabetes",
        "min": 0.00,
        "max": 200.00,
        "default": 00.40,
        "step": 0.01,
        "help": "",
    },
    "Age": {
        "label": "📆 Edad",
        "description": "Edad (años)",
        "min": 1,
        "max": 130,
        "default": 18,
        "step": 1,
        "help": "",
    },
}

# Umbrales de decisión
THRESHOLD_LOW = 0.3  # Por debajo: Riesgo Bajo
THRESHOLD_HIGH = 0.6  # Por encima: Riesgo Alto

# ============================================
# FUNCIONES DE CARGA
# ============================================


@st.cache_resource
def load_model():
    """Carga el modelo y sus metadatos."""
    model_path = Path(__file__).parent / "models" / "model_joblib.joblib"
    metadata_path = Path(__file__).parent / "models" / "model_metadata.json"

    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return model, feature_names, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontró el archivo del modelo. {e}")
        st.info("💡 Ejecuta primero el notebook `diabetes.ipynb`")
        return None, None, None


# ============================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================
def create_gauge_chart(probability: float) -> go.Figure:
    """Crea un gráfico de velocímetro para la probabilidad de disagnóstico de diabetes."""
    if probability < THRESHOLD_LOW:
        color = "#28a745"
    elif probability < THRESHOLD_HIGH:
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            delta={"reference": 50, "relative": False, "position": "bottom"},
            title={"text": "Probabilidad de Diabetes", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#1e3a5f"},
                "bar": {"color": color, "thickness": 0.75},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e3a5f"},
    )
    return fig


def create_feature_impact_chart(input_values: dict, feature_config: dict) -> go.Figure:
    """Crea un gráfico de barras mostrando el nivel de riesgo por variable."""
    risk_thresholds = {
        "Pregnancies": 6,
        "Glucose": 190,
        "BloodPressure": 180,
        "SkinThickness": 35,
        "Insulin": 60,
        "BMI": 40.0,
        "DiabetesPedigreeFunction": 1.0,
        "Age": 50,
    }

    features = []
    values_normalized = []
    colors = []

    for feat_name, config in feature_config.items():
        features.append(config["label"])
        val = input_values[feat_name]
        threshold = risk_thresholds[feat_name]
        # revisar
        if feat_name in ["", ""]:
            risk_level = max(0, (threshold - val) / threshold)
        else:
            risk_level = min(1, val / threshold)

        values_normalized.append(risk_level * 100)

        if risk_level < 0.5:
            colors.append("#28a745")
        elif risk_level < 0.8:
            colors.append("#ffc107")
        else:
            colors.append("#dc3545")

    fig = go.Figure(
        go.Bar(
            x=values_normalized,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in values_normalized],
            textposition="inside",
            textfont=dict(color="white", size=12),
        )
    )

    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="#6c757d",
        annotation_text="Umbral de Riesgo",
        annotation_position="top",
    )

    fig.update_layout(
        title="📊 Nivel de Riesgo por Variable",
        xaxis_title="Nivel de Riesgo (%)",
        yaxis_title="",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], gridcolor="#e9ecef"),
        yaxis=dict(gridcolor="#e9ecef"),
    )
    return fig


def get_decision_badge(probability: float) -> tuple:
    """Retorna el HTML del badge de decisión y la explicación."""
    if probability < THRESHOLD_LOW:
        badge = '<span class="badge-approved">✅ Riesgo Bajo</span>'
        explanation = """
        **Recomendación: Continuar con su estilo de vida saludable**

        Continuar con su estilo de vida saludable. Parámetros metabólicos normales y pocos factores hereditarios relevantes.      
        """
        color = "success"
    elif probability < THRESHOLD_HIGH:
        badge = '<span class="badge-review">⚠️ Riesgo Moderado</span>'
        explanation = """
        **Recomendación: Dieta Saludable y Ejercicios**

        Se le recomienda pasar consulta con el nutricionista y realizar ejercicios.
        """
        color = "warning"
    else:
        badge = '<span class="badge-rejected">❌ Riesgo Alto</span>'
        explanation = """
        **Recomendación: Tratamiento Urgente**

        Se le recomienda una dieta saludable, realizar ejercicios y chequeos constantes.
        """
        color = "error"

    return badge, explanation, color


# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🩺 Sistema de Evaluación de Diabetes</h1>
        <h4>Control 4: Modelo LightGBM</h4>
        <p>Machine Learning en Producción - PECD UNI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Cargar modelo
    model, feature_names, metadata = load_model()

    if model is None:
        st.stop()

    # Sidebar - Información del modelo
    with st.sidebar:
        st.markdown("### 📋 Información del Modelo")

        with st.expander("📊 Métricas", expanded=True):
            st.metric("AUC-ROC", f"{metadata['metrics']['auc_test']:.4f}")
            st.metric("Features Totales", metadata["n_features"])
            st.metric("Train Samples", f"{metadata['metrics']['n_train_samples']:,}")

        with st.expander("⚙️ Configuración"):
            st.write(f"**Modelo:** {metadata['model_type']}")
            st.write(f"**Versión:** v{metadata['training_info']['version']}")
            st.write(f"**Fecha:** {metadata['training_info']['date'][:10]}")

        st.markdown("---")
        st.markdown("### 🎚️ Umbrales de Decisión")
        st.write(f"🟢Riesgo Bajo: < {THRESHOLD_LOW * 100:.0f}%")
        st.write(
            f"🟡Riesgo Moderado: {THRESHOLD_LOW * 100:.0f}% - {THRESHOLD_HIGH * 100:.0f}%"
        )
        st.write(f"🔴Riesgo Alto: > {THRESHOLD_HIGH * 100:.0f}%")

        st.markdown("---")
        st.markdown("### 🎓 PECD - UNI")
        st.caption("Programa de Especialización en Ciencia de Datos")

    # Contenido principal - Dos columnas
    col_input, col_result = st.columns([1, 1.2])

    # Columna de inputs
    with col_input:
        st.markdown("### 📝 Datos del Paciente")
        st.markdown(
            "Complete los siguientes campos para evaluar el riesgo de diabetes:"
        )

        input_values = {}

        for feat_name, config in FEATURE_CONFIG.items():
            st.markdown(f"**{config['label']}**")
            st.caption(config["description"])

            if isinstance(config["default"], float):
                input_values[feat_name] = st.number_input(
                    label=feat_name,
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(config["default"]),
                    step=float(config["step"]),
                    help=config["help"],
                    label_visibility="collapsed",
                )
            else:
                input_values[feat_name] = st.number_input(
                    label=feat_name,
                    min_value=int(config["min"]),
                    max_value=int(config["max"]),
                    value=int(config["default"]),
                    step=int(config["step"]),
                    help=config["help"],
                    label_visibility="collapsed",
                )
            st.markdown("")

        predict_button = st.button(
            "🔮 **EVALUAR**", use_container_width=True, type="primary"
        )

    # Columna de resultados
    with col_result:
        if predict_button:
            # Preparar datos - rellenar las features no usadas con 0
            input_df = pd.DataFrame({feat: [0.0] for feat in feature_names})

            for feat, val in input_values.items():
                if feat in input_df.columns:
                    input_df[feat] = val

            with st.spinner("Analizando perfil..."):
                probability = model.predict_proba(input_df)[0, 1]

            st.markdown("### 📊 Resultado del Análisis")

            # Gauge
            gauge_fig = create_gauge_chart(probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Decisión
            badge_html, explanation, color = get_decision_badge(probability)

            st.markdown(
                f"""
            <div style="text-align: center; margin: 20px 0;">
                {badge_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

            if color == "success":
                st.success(explanation)
            elif color == "warning":
                st.warning(explanation)
            else:
                st.error(explanation)

            st.markdown("---")
            impact_fig = create_feature_impact_chart(input_values, FEATURE_CONFIG)
            st.plotly_chart(impact_fig, use_container_width=True)

            with st.expander("🔍 Ver datos técnicos"):
                st.json(
                    {
                        "probabilidad_diabetes": round(probability, 4),
                        "decision": "Riesgo Bajo"
                        if probability < THRESHOLD_LOW
                        else (
                            "Riesgo Moderado" if probability < THRESHOLD_HIGH else "Riesgo Alto"
                        ),
                        "inputs": input_values,
                        "threshold_low": THRESHOLD_LOW,
                        "threshold_high": THRESHOLD_HIGH,
                    }
                )
        else:
            st.markdown("### 📊 Resultado del Análisis")
            st.info(
                "👈 Complete los datos del paciente y presione **EVALUAR** para obtener el análisis."
            )

            st.markdown(
                """
            <div style="text-align: center; padding: 50px; color: #6c757d;">
                <h1>🩺</h1>
                <p>Esperando datos del paciente...</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="footer">
        <p>🎓 <strong>Machine Learning Supervisado</strong> - Programa de Especialización en Ciencia de Datos</p>
        <p>Universidad Nacional de Ingeniería (UNI) - 2025</p>
        <p><em>Este es un modelo de demostración con fines educativos.</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
