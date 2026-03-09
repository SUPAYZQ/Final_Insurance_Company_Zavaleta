"""
=============================================================
  Práctica Final: "Análisis Exploratorio de Datos (EDA)" - Insurance Company
  Especialización Python for Analytics
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EDA - Insurance Company",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ESTILO CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #f4f6fb; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2b5f 0%, #243b8a 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }

    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #243b8a;
        margin-bottom: 10px;
    }
    .metric-card h4 { color: #6b7280; font-size: 0.82rem; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card h2 { color: #1a2b5f; font-size: 1.7rem; margin: 4px 0 0 0; font-weight: 700; }

    /* Títulos de sección */
    .section-title {
        background: linear-gradient(90deg, #1a2b5f, #243b8a);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 12px 0;
    }

    /* Cajas de insight */
    .insight-box {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #1e3a8a;
        font-size: 0.93rem;
    }

    /* Badge tecnología */
    .tech-badge {
        display: inline-block;
        background: #1a2b5f;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 3px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: #1a2b5f !important; border-bottom: 3px solid #1a2b5f !important; }

    /* Botones */
    .stButton > button {
        background: #1a2b5f; color: white;
        border-radius: 8px; border: none;
        font-weight: 600;
    }
    .stButton > button:hover { background: #243b8a; }

    /* Dataframes */
    .dataframe { font-size: 0.85rem !important; }

    /* Separador */
    hr { border: 1px solid #e0e7ff; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#   CLASE PRINCIPAL — DataAnalyzer  (POO)
# ═══════════════════════════════════════════════════════════
class DataAnalyzer:
    """
    Clase que encapsula las operaciones de análisis exploratorio
    sobre el dataset InsuranceCompany.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()

    # ── Preprocesamiento ──────────────────────────────────
    def _preprocess(self):
        """Prepara columnas derivadas útiles para el análisis."""
        self.df["age_years"] = (self.df["age_in_days"] / 365.25).round(1)
        self.df["renewal_label"] = self.df["renewal"].map({1: "Renovó", 0: "No Renovó"})
        self.df["income_group"] = pd.cut(
            self.df["Income"],
            bins=[0, 100_000, 250_000, 500_000, np.inf],
            labels=["Bajo (<100K)", "Medio (100K-250K)", "Alto (250K-500K)", "Muy Alto (>500K)"],
        )

    # ── Clasificación de variables ───────────────────────
    def classify_variables(self) -> dict:
        numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        return {"numericas": numericas, "categoricas": categoricas}

    # ── Estadísticas descriptivas ────────────────────────
    def descriptive_stats(self) -> pd.DataFrame:
        num_cols = self.classify_variables()["numericas"]
        desc = self.df[num_cols].describe().T
        desc["median"] = self.df[num_cols].median()
        desc["skew"] = self.df[num_cols].skew()
        desc["kurtosis"] = self.df[num_cols].kurtosis()
        return desc.round(3)

    # ── Valores faltantes ────────────────────────────────
    def missing_values(self) -> pd.DataFrame:
        total = self.df.isnull().sum()
        pct = (total / len(self.df) * 100).round(2)
        return pd.DataFrame({"Nulos": total, "Porcentaje (%)": pct}).sort_values("Nulos", ascending=False)

    # ── Info resumida ────────────────────────────────────
    def summary_info(self) -> dict:
        return {
            "filas": len(self.df),
            "columnas": len(self.df.columns),
            "duplicados": self.df.duplicated().sum(),
            "tasa_renovacion": f"{self.df['renewal'].mean()*100:.1f}%",
        }

    # ── Figura: distribución numérica ───────────────────
    def plot_numeric_distribution(self, col: str, bins: int = 30) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        color = "#243b8a"
        # Histograma
        sns.histplot(self.df[col].dropna(), bins=bins, color=color, edgecolor="white", ax=axes[0])
        axes[0].set_title(f"Histograma — {col}", fontweight="bold", fontsize=11)
        axes[0].set_xlabel(col); axes[0].set_ylabel("Frecuencia")
        axes[0].axvline(self.df[col].mean(), color="#e53e3e", ls="--", lw=1.8, label=f"Media: {self.df[col].mean():,.1f}")
        axes[0].axvline(self.df[col].median(), color="#38a169", ls="--", lw=1.8, label=f"Mediana: {self.df[col].median():,.1f}")
        axes[0].legend(fontsize=9)
        # Boxplot
        sns.boxplot(y=self.df[col].dropna(), color="#c7d2fe", ax=axes[1])
        axes[1].set_title(f"Boxplot — {col}", fontweight="bold", fontsize=11)
        axes[1].set_ylabel(col)
        fig.tight_layout()
        return fig

    # ── Figura: categorías ───────────────────────────────
    def plot_categorical(self, col: str) -> plt.Figure:
        vc = self.df[col].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = sns.color_palette("Blues_r", len(vc))
        # Barras
        axes[0].bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white")
        axes[0].set_title(f"Frecuencia — {col}", fontweight="bold", fontsize=11)
        axes[0].set_xlabel(col); axes[0].set_ylabel("Conteo")
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + max(vc.values)*0.01, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
        # Pie
        axes[1].pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
                    colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
        axes[1].set_title(f"Proporción — {col}", fontweight="bold", fontsize=11)
        fig.tight_layout()
        return fig

    # ── Figura: bivariado num vs cat ─────────────────────
    def plot_bivariate_num_cat(self, num_col: str, cat_col: str) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        palette = {"Renovó": "#243b8a", "No Renovó": "#e53e3e"}
        use_col = "renewal_label" if cat_col == "renewal" else cat_col
        p = palette if use_col == "renewal_label" else "Blues"
        # Boxplot
        sns.boxplot(data=self.df, x=use_col, y=num_col, palette=p, ax=axes[0])
        axes[0].set_title(f"{num_col} vs {use_col}", fontweight="bold", fontsize=11)
        axes[0].set_xlabel(use_col); axes[0].set_ylabel(num_col)
        # Violin
        sns.violinplot(data=self.df, x=use_col, y=num_col, palette=p, ax=axes[1], inner="quartile")
        axes[1].set_title(f"Violin — {num_col} vs {use_col}", fontweight="bold", fontsize=11)
        axes[1].set_xlabel(use_col); axes[1].set_ylabel(num_col)
        fig.tight_layout()
        return fig

    # ── Figura: bivariado cat vs cat ─────────────────────
    def plot_bivariate_cat_cat(self, col1: str, col2: str) -> plt.Figure:
        use_col2 = "renewal_label" if col2 == "renewal" else col2
        ct = pd.crosstab(self.df[col1], self.df[use_col2], normalize="index") * 100
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = ["#243b8a", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]
        # Barras apiladas
        ct.plot(kind="bar", stacked=True, ax=axes[0], color=colors[:len(ct.columns)], edgecolor="white")
        axes[0].set_title(f"{col1} vs {use_col2} (% apilado)", fontweight="bold", fontsize=11)
        axes[0].set_xlabel(col1); axes[0].set_ylabel("%"); axes[0].legend(title=use_col2)
        axes[0].tick_params(axis="x", rotation=30)
        # Heatmap
        ct_abs = pd.crosstab(self.df[col1], self.df[use_col2])
        sns.heatmap(ct_abs, annot=True, fmt="d", cmap="Blues", ax=axes[1], linewidths=0.5)
        axes[1].set_title(f"Heatmap — {col1} vs {use_col2}", fontweight="bold", fontsize=11)
        fig.tight_layout()
        return fig

    # ── Figura: correlación ──────────────────────────────
    def plot_correlation(self) -> plt.Figure:
        num_cols = [c for c in self.classify_variables()["numericas"]
                    if c not in ["id", "age_in_days"]]
        corr = self.df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
        ax.set_title("Mapa de Correlación — Variables Numéricas", fontweight="bold", fontsize=13)
        fig.tight_layout()
        return fig

    # ── Figura: hallazgos resumen ────────────────────────
    def plot_key_findings(self) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # 1) Tasa de renovación
        renov = self.df["renewal_label"].value_counts()
        colors_r = ["#243b8a", "#e53e3e"]
        axes[0, 0].pie(renov.values, labels=renov.index, autopct="%1.1f%%",
                       colors=colors_r, startangle=90, wedgeprops={"edgecolor": "white"})
        axes[0, 0].set_title("Tasa de Renovación", fontweight="bold", fontsize=11)

        # 2) Ingresos por renovación
        palette = {"Renovó": "#243b8a", "No Renovó": "#e53e3e"}
        sns.boxplot(data=self.df, x="renewal_label", y="Income", palette=palette, ax=axes[0, 1])
        axes[0, 1].set_title("Ingreso vs Renovación", fontweight="bold", fontsize=11)
        axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel("Ingreso")

        # 3) Canal de captación
        ch = self.df["sourcing_channel"].value_counts()
        axes[1, 0].bar(ch.index, ch.values, color=sns.color_palette("Blues_r", len(ch)), edgecolor="white")
        axes[1, 0].set_title("Distribución por Canal", fontweight="bold", fontsize=11)
        axes[1, 0].set_xlabel("Canal"); axes[1, 0].set_ylabel("Clientes")

        # 4) Primas pagadas vs renovación
        sns.histplot(data=self.df, x="no_of_premiums_paid", hue="renewal_label",
                     bins=20, palette=palette, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title("Primas Pagadas vs Renovación", fontweight="bold", fontsize=11)
        axes[1, 1].set_xlabel("N° Primas Pagadas")

        fig.suptitle("📊 Panel de Hallazgos Clave", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════
#   HELPERS UI
# ═══════════════════════════════════════════════════════════
def metric_card(title: str, value):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>""", unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="section-title">📌 {text}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#   SIDEBAR — MENÚ PRINCIPAL
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Insurance EDA")
    st.markdown("---")
    modulo = st.selectbox(
        "📂 Navegar módulo",
        ["🏠 Home", "📁 Carga del Dataset", "🔬 Análisis EDA", "📋 Conclusiones"],
    )
    st.markdown("---")
    st.markdown("**Proyecto:** Caso de Estudio N°3")
    st.markdown("**Especialización:** Python for Analytics")
    st.markdown("**Año:** 2025")
    st.markdown("---")
    st.markdown("**Dataset:** InsuranceCompany.csv")
    st.caption("79,852 registros · 13 variables")


# ═══════════════════════════════════════════════════════════
#   SESSION STATE — dataset compartido entre módulos
# ═══════════════════════════════════════════════════════════
if "df" not in st.session_state:
    st.session_state["df"] = None
if "analyzer" not in st.session_state:
    st.session_state["analyzer"] = None


# ═══════════════════════════════════════════════════════════
#   MÓDULO 1 — HOME
# ═══════════════════════════════════════════════════════════
if modulo == "🏠 Home":

    st.markdown("# 🛡️ **Análisis Exploratorio de Datos**")
    st.markdown("## **Insurance Company — Retención de Pólizas**")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### **Descripción del Proyecto**")
        st.markdown("""
        Este proyecto aplica técnicas de **Análisis Exploratorio de Datos (EDA)**
        sobre el dataset *InsuranceCompany*, que contiene información sobre clientes
        y pólizas de una compañía aseguradora.

        El objetivo es **entender los factores asociados a la renovación de pólizas**,
        sin construir modelos predictivos, sino generando *insights* que orienten
        la toma de decisiones comerciales.
        """)

        st.markdown("### **Autor**")
        st.markdown("""
        | Campo | Detalle |
        |---|---|
        | 👤 **Nombre** | Estudiante — Python for Analytics |
        | 📚 **Especialización** | Python for Analytics |
        | 📅 **Año** | 2025 |
        """)

    with col2:
        st.markdown("### **Tecnologías**")
        for tech in ["Python 3.11", "Streamlit", "Pandas", "NumPy",
                     "Matplotlib", "Seaborn", "GitHub"]:
            st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### **📂 Descripción del Dataset**")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: metric_card("Registros totales", "79,852")
    with col_b: metric_card("Variables", "13")
    with col_c: metric_card("Variable objetivo", "renewal")
    with col_d: metric_card("Tipo de análisis", "EDA")

    st.markdown("#### **Variables del dataset**")
    variables_info = {
        "Variable": ["id","perc_premium_paid_by_cash_credit","age_in_days","Income",
                     "Count_3-6_months_late","Count_6-12_months_late","Count_more_than_12_months_late",
                     "application_underwriting_score","no_of_premiums_paid",
                     "sourcing_channel","residence_area_type","premium","renewal"],
        "Tipo": ["int","float","int","int","float","float","float","float","int","str","str","int","int"],
        "Descripción": [
            "ID único del cliente/póliza",
            "% de la prima pagada en efectivo/crédito",
            "Edad del cliente en días",
            "Ingreso mensual del cliente",
            "Pagos demorados 3–6 meses",
            "Pagos demorados 6–12 meses",
            "Pagos demorados >12 meses",
            "Puntuación de evaluación del riesgo",
            "Total de primas pagadas",
            "Canal de captación (A,B,C,D,E)",
            "Área de residencia (Urban/Rural)",
            "Valor monetario de la prima",
            "¿Renovó la póliza? (1=Sí, 0=No)",
        ],
    }
    st.dataframe(pd.DataFrame(variables_info), use_container_width=True, hide_index=True)

    st.info("⬅️ Navega a **Carga del Dataset** para comenzar el análisis.")


# ═══════════════════════════════════════════════════════════
#   MÓDULO 2 — CARGA DEL DATASET
# ═══════════════════════════════════════════════════════════
elif modulo == "📁 Carga del Dataset":

    st.markdown("# 📁 **Carga del Dataset**")
    st.markdown("### **Sube el archivo CSV para habilitar los análisis**")
    st.markdown("---")

    uploaded = st.file_uploader("Selecciona el archivo InsuranceCompany.csv", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df
            st.session_state["analyzer"] = DataAnalyzer(df)
            st.success(f"✅ Archivo cargado correctamente: **{uploaded.name}**")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
            st.stop()

    if st.session_state["df"] is not None:
        df = st.session_state["df"]

        st.markdown("---")
        st.markdown("## **Vista Previa del Dataset**")

        n_rows = st.slider("Número de filas a mostrar", 5, 50, 10, key="preview_slider")
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown("## **Dimensiones del Dataset**")
        col1, col2, col3, col4 = st.columns(4)
        with col1: metric_card("📏 Filas", f"{df.shape[0]:,}")
        with col2: metric_card("📊 Columnas", df.shape[1])
        with col3: metric_card("🔢 Numéricas", len(df.select_dtypes(include=np.number).columns))
        with col4: metric_card("🔤 Categóricas", len(df.select_dtypes(include="object").columns))

        st.markdown("## **Tipos de Datos**")
        dtype_df = pd.DataFrame({"Columna": df.dtypes.index, "Tipo": df.dtypes.values.astype(str)})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    else:
        st.warning("⚠️ Aún no has cargado el archivo. Sube el CSV para continuar.")


# ═══════════════════════════════════════════════════════════
#   MÓDULO 3 — EDA
# ═══════════════════════════════════════════════════════════
elif modulo == "🔬 Análisis EDA":

    if st.session_state["df"] is None:
        st.warning("⚠️ Primero debes cargar el dataset en el módulo **Carga del Dataset**.")
        st.stop()

    df: pd.DataFrame = st.session_state["df"]
    az: DataAnalyzer = st.session_state["analyzer"]

    st.markdown("# 🔬 **Análisis Exploratorio de Datos (EDA)**")
    st.markdown("### **InsuranceCompany — 10 Ítems de Análisis**")
    st.markdown("---")

    tabs = st.tabs([
        "1️⃣ Info General",
        "2️⃣ Clasificación",
        "3️⃣ Descriptivas",
        "4️⃣ Faltantes",
        "5️⃣ Dist. Numéricas",
        "6️⃣ Categóricas",
        "7️⃣ Biv. Num×Cat",
        "8️⃣ Biv. Cat×Cat",
        "9️⃣ Análisis Dinámico",
        "🔟 Hallazgos Clave",
    ])

    # ── TAB 1: Info General ──────────────────────────────
    with tabs[0]:
        st.markdown("## **Ítem 1: Información General del Dataset**")
        st.markdown("Resumen estructural del conjunto de datos: tipos, nulos y métricas generales.")

        col1, col2 = st.columns(2)
        with col1:
            section_title("Tipos de datos por columna")
            dtype_df = pd.DataFrame({
                "Columna": df.dtypes.index,
                "Tipo": df.dtypes.values.astype(str),
                "No Nulos": df.notna().sum().values,
                "Nulos": df.isna().sum().values,
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with col2:
            section_title("Resumen rápido")
            info = az.summary_info()
            metric_card("Registros", f"{info['filas']:,}")
            metric_card("Columnas", info["columnas"])
            metric_card("Duplicados", info["duplicados"])
            metric_card("Tasa de Renovación", info["tasa_renovacion"])

        section_title("Info completa (.info())")
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue(), language="text")

        insight_box("El dataset tiene 79,852 registros sin duplicados. "
                    "La mayoría de variables son numéricas. La tasa de renovación es del 93.7%, "
                    "lo que indica un desbalance marcado en la variable objetivo.")

    # ── TAB 2: Clasificación ─────────────────────────────
    with tabs[1]:
        st.markdown("## **Ítem 2: Clasificación de Variables**")
        st.markdown("Función personalizada para identificar variables numéricas y categóricas.")

        vars_cls = az.classify_variables()
        col1, col2 = st.columns(2)

        with col1:
            section_title(f"Variables Numéricas ({len(vars_cls['numericas'])})")
            num_df = pd.DataFrame({
                "Variable": vars_cls["numericas"],
                "Tipo": [str(df[c].dtype) for c in vars_cls["numericas"]],
                "Únicos": [df[c].nunique() for c in vars_cls["numericas"]],
            })
            st.dataframe(num_df, use_container_width=True, hide_index=True)

        with col2:
            section_title(f"Variables Categóricas ({len(vars_cls['categoricas'])})")
            if vars_cls["categoricas"]:
                cat_df = pd.DataFrame({
                    "Variable": vars_cls["categoricas"],
                    "Tipo": [str(df[c].dtype) for c in vars_cls["categoricas"]],
                    "Únicos": [df[c].nunique() for c in vars_cls["categoricas"]],
                    "Valores": [list(df[c].unique()[:5]) for c in vars_cls["categoricas"]],
                })
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron columnas de tipo object en el dataset original.")

        section_title("Variables adicionales derivadas")
        extra = pd.DataFrame({
            "Variable derivada": ["age_years", "renewal_label", "income_group"],
            "Origen": ["age_in_days / 365.25", "renewal (0/1 → etiqueta)", "Income en rangos"],
            "Uso": ["Análisis por edad en años", "Visualización legible", "Segmentación de ingresos"],
        })
        st.dataframe(extra, use_container_width=True, hide_index=True)

        insight_box("Se identificaron 11 variables numéricas y 2 categóricas originales "
                    "(sourcing_channel, residence_area_type). Se crearon 3 variables derivadas "
                    "para mejorar la legibilidad de los análisis.")

    # ── TAB 3: Estadísticas Descriptivas ─────────────────
    with tabs[2]:
        st.markdown("## **Ítem 3: Estadísticas Descriptivas**")
        st.markdown("Medidas de tendencia central, dispersión y forma de la distribución.")

        desc = az.descriptive_stats()
        section_title("Tabla estadística completa")
        st.dataframe(desc.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

        st.markdown("### **Interpretación por variable**")
        col1, col2, col3 = st.columns(3)
        with col1:
            section_title("Income")
            st.markdown(f"- **Media:** {df['Income'].mean():,.0f}")
            st.markdown(f"- **Mediana:** {df['Income'].median():,.0f}")
            st.markdown(f"- **Moda:** {df['Income'].mode()[0]:,.0f}")
            st.markdown(f"- **Std:** {df['Income'].std():,.0f}")
        with col2:
            section_title("Premium")
            st.markdown(f"- **Media:** {df['premium'].mean():,.0f}")
            st.markdown(f"- **Mediana:** {df['premium'].median():,.0f}")
            st.markdown(f"- **Moda:** {df['premium'].mode()[0]:,.0f}")
            st.markdown(f"- **Std:** {df['premium'].std():,.0f}")
        with col3:
            section_title("no_of_premiums_paid")
            st.markdown(f"- **Media:** {df['no_of_premiums_paid'].mean():.2f}")
            st.markdown(f"- **Mediana:** {df['no_of_premiums_paid'].median():.2f}")
            st.markdown(f"- **Moda:** {df['no_of_premiums_paid'].mode()[0]}")
            st.markdown(f"- **Std:** {df['no_of_premiums_paid'].std():.2f}")

        insight_box("El ingreso promedio es ~227,596 con alta dispersión (std ~152K), "
                    "lo que sugiere una base de clientes heterogénea. "
                    "La prima mediana es 6,000 pero con distribución sesgada hacia la derecha.")

    # ── TAB 4: Valores Faltantes ──────────────────────────
    with tabs[3]:
        st.markdown("## **Ítem 4: Análisis de Valores Faltantes**")
        st.markdown("Identificación y cuantificación de datos ausentes en el dataset.")

        missing = az.missing_values()
        col1, col2 = st.columns([1, 1.5])

        with col1:
            section_title("Tabla de nulos por columna")
            st.dataframe(missing.style.bar(subset=["Porcentaje (%)"], color="#c7d2fe"),
                         use_container_width=True)

        with col2:
            section_title("Visualización de nulos")
            missing_filt = missing[missing["Nulos"] > 0]
            if not missing_filt.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(missing_filt.index, missing_filt["Porcentaje (%)"],
                        color="#243b8a", edgecolor="white")
                ax.set_xlabel("Porcentaje (%)")
                ax.set_title("Variables con valores faltantes", fontweight="bold")
                for i, v in enumerate(missing_filt["Porcentaje (%)"]):
                    ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.success("🎉 ¡El dataset no tiene valores faltantes!")

        total_nulos = missing["Nulos"].sum()
        if total_nulos == 0:
            insight_box("El dataset está completo: no se detectaron valores nulos en ninguna de las 13 variables. "
                        "Esto simplifica el preprocesamiento y garantiza análisis sin sesgos por imputación.")
        else:
            insight_box(f"Se detectaron {total_nulos:,} valores nulos en total. "
                        "Se recomienda revisar si es posible imputar o eliminar estos registros.")

    # ── TAB 5: Distribución Numéricas ─────────────────────
    with tabs[4]:
        st.markdown("## **Ítem 5: Distribución de Variables Numéricas**")
        st.markdown("Histogramas y boxplots para explorar la forma de las distribuciones.")

        vars_num = [c for c in az.classify_variables()["numericas"]
                    if c not in ["id", "renewal"]]

        col_sel, col_bins = st.columns([2, 1])
        with col_sel:
            var_dist = st.selectbox("Selecciona variable numérica:", vars_num, key="dist_sel")
        with col_bins:
            bins_n = st.slider("Número de bins:", 10, 80, 30, key="bins_slider")

        show_stats = st.checkbox("Mostrar estadísticas rápidas", value=True, key="show_stats_cb")
        if show_stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Media", f"{df[var_dist].mean():,.2f}")
            with c2: metric_card("Mediana", f"{df[var_dist].median():,.2f}")
            with c3: metric_card("Desv. Estándar", f"{df[var_dist].std():,.2f}")
            with c4: metric_card("Asimetría", f"{df[var_dist].skew():.3f}")

        fig = az.plot_numeric_distribution(var_dist, bins=bins_n)
        st.pyplot(fig)

        skew_val = df[var_dist].skew()
        if abs(skew_val) < 0.5:
            insight_box(f"**{var_dist}** tiene distribución aproximadamente simétrica (skew={skew_val:.2f}).")
        elif skew_val > 0:
            insight_box(f"**{var_dist}** presenta sesgo positivo (skew={skew_val:.2f}): "
                        "cola larga hacia la derecha, posibles valores atípicos altos.")
        else:
            insight_box(f"**{var_dist}** presenta sesgo negativo (skew={skew_val:.2f}): "
                        "cola larga hacia la izquierda.")

    # ── TAB 6: Categóricas ────────────────────────────────
    with tabs[5]:
        st.markdown("## **Ítem 6: Análisis de Variables Categóricas**")
        st.markdown("Frecuencias, proporciones y distribución visual de variables cualitativas.")

        cat_options = ["sourcing_channel", "residence_area_type", "renewal_label", "income_group"]
        var_cat = st.selectbox("Selecciona variable categórica:", cat_options, key="cat_sel")

        use_var = var_cat
        if var_cat == "renewal_label":
            use_df_col = az.df["renewal_label"]
        elif var_cat == "income_group":
            use_df_col = az.df["income_group"]
        else:
            use_df_col = df[var_cat]

        col1, col2 = st.columns([1, 2])
        with col1:
            section_title("Tabla de frecuencias")
            vc = use_df_col.value_counts()
            freq_df = pd.DataFrame({
                "Categoría": vc.index.astype(str),
                "Conteo": vc.values,
                "Proporción (%)": (vc.values / len(use_df_col) * 100).round(2),
            })
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

        with col2:
            section_title("Visualización")
            if var_cat in ["renewal_label", "income_group"]:
                fig = az.plot_categorical(var_cat.replace("_label", "") if var_cat == "renewal_label" else var_cat)
                # Usar directamente la columna derivada
                vc2 = use_df_col.value_counts()
                fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
                colors = sns.color_palette("Blues_r", len(vc2))
                axes2[0].bar(vc2.index.astype(str), vc2.values, color=colors, edgecolor="white")
                axes2[0].set_title(f"Frecuencia — {var_cat}", fontweight="bold", fontsize=11)
                axes2[0].set_ylabel("Conteo")
                for i, v in enumerate(vc2.values):
                    axes2[0].text(i, v + max(vc2.values)*0.01, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
                axes2[1].pie(vc2.values, labels=vc2.index.astype(str), autopct="%1.1f%%",
                             colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
                axes2[1].set_title(f"Proporción — {var_cat}", fontweight="bold", fontsize=11)
                fig2.tight_layout()
                st.pyplot(fig2)
            else:
                fig = az.plot_categorical(var_cat)
                st.pyplot(fig)

        insight_box(f"La categoría más frecuente en **{var_cat}** es "
                    f"**{freq_df['Categoría'].iloc[0]}** con "
                    f"{freq_df['Conteo'].iloc[0]:,} registros "
                    f"({freq_df['Proporción (%)'].iloc[0]:.1f}%).")

    # ── TAB 7: Bivariado Num × Cat ─────────────────────────
    with tabs[6]:
        st.markdown("## **Ítem 7: Análisis Bivariado — Numérico vs Categórico**")
        st.markdown("Comparación de distribuciones numéricas segmentadas por grupos categóricos.")

        vars_num7 = [c for c in az.classify_variables()["numericas"]
                     if c not in ["id"]]
        cat_options7 = ["renewal", "sourcing_channel", "residence_area_type"]

        col1, col2 = st.columns(2)
        with col1:
            num_var7 = st.selectbox("Variable numérica:", vars_num7,
                                    index=vars_num7.index("Income") if "Income" in vars_num7 else 0,
                                    key="num7")
        with col2:
            cat_var7 = st.selectbox("Variable categórica:", cat_options7, key="cat7")

        fig7 = az.plot_bivariate_num_cat(num_var7, cat_var7)
        st.pyplot(fig7)

        # Estadísticas por grupo
        use_col7 = "renewal_label" if cat_var7 == "renewal" else cat_var7
        grp_col = az.df[use_col7] if use_col7 == "renewal_label" else df[cat_var7]
        grp_stats = df.groupby(grp_col)[num_var7].agg(["mean", "median", "std"]).round(2)
        grp_stats.columns = ["Media", "Mediana", "Desv. Estándar"]
        section_title("Estadísticas por grupo")
        st.dataframe(grp_stats, use_container_width=True)

        diff = grp_stats["Media"].max() - grp_stats["Media"].min()
        insight_box(f"La diferencia de medias entre grupos en **{num_var7}** es de "
                    f"**{diff:,.2f}**. "
                    "Un valor alto sugiere que esta variable discrimina bien entre categorías.")

    # ── TAB 8: Bivariado Cat × Cat ─────────────────────────
    with tabs[7]:
        st.markdown("## **Ítem 8: Análisis Bivariado — Categórico vs Categórico**")
        st.markdown("Relaciones entre variables cualitativas mediante tablas cruzadas y heatmaps.")

        cat_options8 = ["sourcing_channel", "residence_area_type", "income_group"]
        target_options8 = ["renewal", "sourcing_channel", "residence_area_type"]

        col1, col2 = st.columns(2)
        with col1:
            cat_var8a = st.selectbox("Variable categórica (eje X):", cat_options8, key="cat8a")
        with col2:
            cat_var8b = st.selectbox("Variable objetivo (eje Y):", target_options8, key="cat8b")

        if cat_var8a == cat_var8b:
            st.warning("⚠️ Selecciona variables diferentes para el análisis bivariado.")
        else:
            use_col8a = cat_var8a
            if cat_var8a == "income_group":
                temp_df = az.df.copy()
            else:
                temp_df = df.copy()

            fig8 = az.plot_bivariate_cat_cat(
                "income_group" if cat_var8a == "income_group" else cat_var8a,
                cat_var8b
            )
            st.pyplot(fig8)

            section_title("Tabla de contingencia")
            use_col_b = "renewal_label" if cat_var8b == "renewal" else cat_var8b
            col_a_data = az.df["income_group"] if cat_var8a == "income_group" else df[cat_var8a]
            col_b_data = az.df["renewal_label"] if cat_var8b == "renewal" else df[cat_var8b]
            ct = pd.crosstab(col_a_data, col_b_data, margins=True)
            st.dataframe(ct, use_container_width=True)

            insight_box("El canal de captación y el tipo de área de residencia muestran "
                        "patrones diferenciados en la tasa de renovación, lo que indica "
                        "segmentos con comportamiento heterogéneo.")

    # ── TAB 9: Análisis Dinámico ───────────────────────────
    with tabs[8]:
        st.markdown("## **Ítem 9: Análisis Dinámico por Parámetros**")
        st.markdown("Explora relaciones personalizadas seleccionando variables interactivamente.")

        st.markdown("### **⚙️ Configuración del análisis**")
        all_num = [c for c in az.classify_variables()["numericas"] if c not in ["id"]]
        all_cat = ["renewal_label", "sourcing_channel", "residence_area_type", "income_group"]

        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("Variable X (numérica):", all_num,
                                 index=all_num.index("Income"), key="dyn_x")
        with col2:
            y_var = st.selectbox("Variable Y (numérica):", all_num,
                                 index=all_num.index("premium") if "premium" in all_num else 1,
                                 key="dyn_y")
        with col3:
            hue_var = st.selectbox("Color / Segmento:", all_cat, key="dyn_hue")

        filter_renov = st.checkbox("Filtrar solo renovaciones (renewal = 1)", key="filt_renov")
        income_range = st.slider(
            "Rango de ingreso a incluir:",
            int(df["Income"].min()), int(df["Income"].max()),
            (int(df["Income"].quantile(0.05)), int(df["Income"].quantile(0.95))),
            step=5000, key="income_range"
        )

        plot_df = az.df.copy()
        if filter_renov:
            plot_df = plot_df[plot_df["renewal"] == 1]
        plot_df = plot_df[(plot_df["Income"] >= income_range[0]) & (plot_df["Income"] <= income_range[1])]

        st.markdown(f"**Registros filtrados:** {len(plot_df):,} de {len(df):,}")

        cols_multisel = st.multiselect(
            "Variables adicionales a mostrar en estadísticas:",
            all_num,
            default=["Income", "premium", "no_of_premiums_paid"],
            key="multi_stats"
        )

        fig9, ax9 = plt.subplots(figsize=(10, 5))
        palette9 = sns.color_palette("tab10", plot_df[hue_var].nunique())
        for i, (grp, grp_data) in enumerate(plot_df.groupby(hue_var)):
            sample = grp_data.sample(min(500, len(grp_data)), random_state=42)
            ax9.scatter(sample[x_var], sample[y_var], label=str(grp),
                        alpha=0.5, s=18, color=palette9[i % len(palette9)])
        ax9.set_xlabel(x_var, fontsize=11)
        ax9.set_ylabel(y_var, fontsize=11)
        ax9.set_title(f"Dispersión: {x_var} vs {y_var} — coloreado por {hue_var}",
                      fontweight="bold", fontsize=12)
        ax9.legend(title=hue_var, fontsize=9)
        fig9.tight_layout()
        st.pyplot(fig9)

        if cols_multisel:
            section_title("Estadísticas de variables seleccionadas")
            st.dataframe(plot_df[cols_multisel].describe().T.round(3), use_container_width=True)

        insight_box(f"Analizando **{x_var}** vs **{y_var}** segmentado por **{hue_var}** "
                    f"con {len(plot_df):,} registros filtrados.")

    # ── TAB 10: Hallazgos Clave ────────────────────────────
    with tabs[9]:
        st.markdown("## **Ítem 10: Hallazgos Clave del EDA**")
        st.markdown("Visualización resumen e insights principales derivados del análisis.")

        fig10 = az.plot_key_findings()
        st.pyplot(fig10)

        st.markdown("---")
        st.markdown("### **📊 Panel de métricas clave**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: metric_card("Tasa renovación", f"{df['renewal'].mean()*100:.1f}%")
        with col2: metric_card("Ingreso mediano", f"{df['Income'].median():,.0f}")
        with col3: metric_card("Prima mediana", f"{df['premium'].median():,.0f}")
        with col4: metric_card("Canal principal", df["sourcing_channel"].mode()[0])
        with col5: metric_card("% Área urbana", f"{(df['residence_area_type']=='Urban').mean()*100:.1f}%")

        st.markdown("---")
        st.markdown("### **💡 Insights Principales**")

        insights = [
            ("Alta tasa de retención", "El 93.7% de los clientes renueva la póliza, lo que indica "
             "un producto de alta fidelidad. Sin embargo, el 6.3% que no renueva representa un "
             "segmento crítico para intervención."),
            ("El ingreso diferencia renovadores", "Los clientes que renuevan tienen ingresos medianos "
             "ligeramente superiores, sugiriendo que la solvencia económica está relacionada con la "
             "continuidad en el seguro."),
            ("Canal A domina la captación", "El canal A concentra la mayor proporción de clientes, "
             "por lo que optimizar este canal podría tener el mayor impacto en el volumen de renovaciones."),
            ("Área urbana con mayor renovación", "Los clientes urbanos muestran tasas de renovación "
             "consistentemente más altas que los rurales, lo que puede orientar estrategias diferenciadas "
             "por zona geográfica."),
            ("Primas pagadas como indicador de lealtad", "Los clientes con mayor número de primas pagadas "
             "tienen mayor propensión a renovar, lo que indica que la antigüedad es un predictor "
             "relevante de la fidelización."),
        ]

        for i, (titulo, texto) in enumerate(insights, 1):
            st.markdown(f"**{i}. {titulo}**")
            st.markdown(f"> {texto}")
            st.markdown("")

        section_title("Correlación con la variable objetivo")
        fig_corr = az.plot_correlation()
        st.pyplot(fig_corr)


# ═══════════════════════════════════════════════════════════
#   MÓDULO 4 — CONCLUSIONES
# ═══════════════════════════════════════════════════════════
elif modulo == "📋 Conclusiones":

    st.markdown("# 📋 **Conclusiones Finales**")
    st.markdown("### **Reflexiones derivadas del Análisis Exploratorio**")
    st.markdown("---")

    conclusiones = [
        {
            "num": "01",
            "titulo": "La renovación es el comportamiento mayoritario pero el abandono es costoso",
            "texto": (
                "Con una tasa de renovación del 93.7%, la compañía tiene una base de clientes "
                "altamente fiel. Sin embargo, los ~5,000 clientes que no renuevan representan "
                "una pérdida significativa de ingresos recurrentes. El EDA revela que este grupo "
                "tiene perfiles diferenciados que permiten diseñar campañas de retención focalizadas."
            ),
        },
        {
            "num": "02",
            "titulo": "El ingreso y la antigüedad son variables diferenciadores clave",
            "texto": (
                "Los análisis bivariados muestran que clientes con ingresos más altos y con mayor "
                "número de primas pagadas históricamente tienen mayor probabilidad de renovar. "
                "Esto sugiere que programas de fidelización orientados a clientes de largo plazo "
                "y alto valor podrían ser especialmente efectivos."
            ),
        },
        {
            "num": "03",
            "titulo": "El canal de captación A es dominante y debe priorizarse operativamente",
            "texto": (
                "El canal A concentra la mayor parte de los clientes. Mantener la calidad y "
                "eficiencia de este canal es crítico para la sostenibilidad del portafolio. "
                "Los canales B, C, D y E representan oportunidades de diversificación con menor "
                "dependencia operativa."
            ),
        },
        {
            "num": "04",
            "titulo": "La segmentación geográfica urbana/rural ofrece oportunidades estratégicas",
            "texto": (
                "Los clientes urbanos representan la mayoría del portafolio y muestran mayor "
                "tasa de renovación. La menor retención en zonas rurales puede estar relacionada "
                "con acceso al servicio, nivel de ingreso o percepción del valor del producto. "
                "Una estrategia diferenciada por área geográfica podría cerrar esta brecha."
            ),
        },
        {
            "num": "05",
            "titulo": "El dataset es de alta calidad y está listo para modelado supervisado",
            "texto": (
                "La ausencia total de valores nulos y la completitud de los registros hacen de "
                "este dataset un insumo ideal para etapas posteriores. El EDA no identifica "
                "necesidades de imputación ni transformaciones críticas previas. "
                "El principal desafío para modelos futuros será el desbalance de clases "
                "en la variable 'renewal' (93.7% vs 6.3%)."
            ),
        },
    ]

    for c in conclusiones:
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 22px 28px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            border-left: 6px solid #1a2b5f;
            margin-bottom: 16px;
        ">
            <h3 style="color:#1a2b5f; margin:0 0 6px 0;">
                <span style="color:#6b7280; font-size:0.85em;">#{c['num']}</span>
                &nbsp; {c['titulo']}
            </h3>
            <p style="color:#374151; margin:0; line-height:1.6;">{c['texto']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### **🛠️ Tecnologías Utilizadas**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Lenguaje & Framework**")
        st.markdown("- Python 3.11\n- Streamlit\n- Pandas\n- NumPy")
    with col2:
        st.markdown("**Visualización**")
        st.markdown("- Matplotlib\n- Seaborn\n- Gráficos interactivos")
    with col3:
        st.markdown("**Patrones de diseño**")
        st.markdown("- Programación Orientada a Objetos\n- Funciones personalizadas\n- f-strings\n- Widgets interactivos")

    st.markdown("---")
    st.success("✅ **Proyecto completado.** Este EDA forma parte del portafolio profesional de la Especialización en Python for Analytics.")
