#  Proyecto Final: "Análisis Exploratorio de Datos EDA" — Insurance Company

## Descripción del Proyecto

Aplicación interactiva construida con **Streamlit** para el Análisis Exploratorio de Datos (EDA) del dataset `InsuranceCompany.csv`, desarrollada como Caso de Estudio N°3 de la Especialización Python for Analytics.

El objetivo es explorar los factores asociados a la **renovación de pólizas de seguro** mediante visualizaciones, estadísticas descriptivas y análisis bivariado.

---

## Estructura de la Aplicación

| Módulo | Descripción |
|--------|-------------|
| 🏠 Home | Presentación del proyecto y dataset |
| 📁 Carga del Dataset | st.file_uploader + vista previa |
| 🔬 Análisis EDA | 10 ítems de análisis completos |
| 📋 Conclusiones | 5 conclusiones basadas en el EDA |

## Ítems de Análisis EDA

1. Información general del dataset
2. Clasificación de variables (numéricas / categóricas)
3. Estadísticas descriptivas
4. Análisis de valores faltantes
5. Distribución de variables numéricas
6. Análisis de variables categóricas
7. Análisis bivariado (numérico vs categórico)
8. Análisis bivariado (categórico vs categórico)
9. Análisis dinámico por parámetros seleccionados
10. Hallazgos clave y mapa de correlación

---

## Instrucciones de Ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/insurance-eda.git
cd insurance-eda

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app.py
```

Luego sube el archivo `InsuranceCompany.csv` en el módulo **Carga del Dataset**.

---

## Dataset

- **Archivo:** `InsuranceCompany.csv`
- **Registros:** 79,852
- **Variables:** 13
- **Variable objetivo:** `renewal` (1 = renovó, 0 = no renovó)

## Tecnologías

`Python` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Autor

- **Jorge Augusto Zavaleta Quintana**

Curso: Especialización Python for Analytics 

2026
