# Control 04 - Sarai Cisneros Lovaton y George Urbina Castro

## Descripción
El objetivo del proyecto es desarrollar y evaluar un modelo de machine learning capaz de predecir el riesgo de diabetes a partir de variables clínicas, comparando distintos algoritmos de clasificación. Asimismo, se busca garantizar la interpretabilidad del modelo mediante técnicas de Explainable AI (SHAP), permitiendo comprender la influencia de cada variable en la predicción y facilitando su uso como herramienta de apoyo a la toma de decisiones clínicas.

## Requisitos
```bash
pip install -r requirements.txt
```
## Pasos
- Ingresar a la carpeta notebook
- Ubicar el archivo Control_4.ipynb y ejecutar todo
- Se generán los archivos model_joblib.joblib y mode_metadata.json dentro de la captera app/models
- Para realizar el deply:


## Para realizar e Deploy
```bash
conda activate ml_pro
cd app
streamlit run app.py
