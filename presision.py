


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from fpdf import FPDF
import io
import base64

def df_to_html_table(df):
    """Convierte un DataFrame a tabla HTML para incrustar en PDF si quieres."""
    return df.to_html()

def create_pdf(stats_df, resultados_df, imagenes_dict):
    """
    Crea un PDF con estadística y resultados y gráficos.
    imagenes_dict = {'nombre_imagen': bytes_imagen_png, ...}
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Evaluación Psicológica Integral - Reporte", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Estadísticas descriptivas:", ln=True)
    pdf.set_font("Arial", '', 10)

    # Convertir stats_df a texto plano y agregar línea a línea
    stats_text = stats_df.round(3).to_string()
    for line in stats_text.split('\n'):
        pdf.cell(0, 6, line, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resultados:", ln=True)
    pdf.set_font("Arial", '', 8)

    # Por simplicidad vamos a poner sólo las primeras filas en texto (puedes mejorarlo con tabla con fpdf más compleja)
    resultados_preview = resultados_df.head(10).to_string()
    for line in resultados_preview.split('\n'):
        pdf.cell(0, 6, line, ln=True)
    pdf.ln(10)

    # Agregar imágenes (gráficas)
    for nombre, img_bytes in imagenes_dict.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, nombre, ln=True)
        pdf.image(io.BytesIO(img_bytes), w=180)
        pdf.ln(10)

    # Guardar a bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes


plt.style.use('ggplot')

subcategorias = {
    "Personalidad": {
        "Extraversión": [
            "Me resulta fácil iniciar conversaciones con desconocidos.",
            "Prefiero pasar tiempo solo o en pequeños grupos que en grandes reuniones sociales.",
            "Tomo decisiones basadas en lógica más que en emociones.",
            "Me gusta aprender sobre ideas nuevas, incluso si son complejas.",
            "Me pongo metas altas y no me conformo con lo mínimo.",
            "Me afecta emocionalmente si no logró destacar.",
            "Competir con los demás me hace mejorar mi rendimiento.",
            "Me resulta fácil iniciar conversaciones con desconocidos"
        ],
        "Amabilidad": [
            "Me gusta ayudar a los demás sin esperar algo a cambio.",
            "Me cuesta confiar en personas nuevas.",
            "Me cuesta confiar en personas nuevas",
            "Suelo priorizar cómo se sienten los demás al tomar decisiones.",
            "Prefiero tener todo planificado y organizado."
        ],
        "Responsabilidad": [
            "Suelo organizar mis tareas con anticipación.",
            "Me esfuerzo por cumplir mis compromisos aunque esté cansado/a.",
            "Suelo dejar las cosas para último momento.",
            "Prefiero seguir rutinas antes que probar cosas nuevas.",
        ]
    },
    "Carácter": {
        "Autocontrol y emociones": [
            "Suelo reaccionar impulsivamente cuando algo me molesta.",
            "Puedo mantener la calma incluso si estoy bajo presión.",
            "Cuando tengo problemas, busco apoyo en otras personas.",
            "Me gusta liderar en equipos de trabajo.",
            "Me gusta liderar en equipos de trabajo",
            "Prefiero recibir instrucciones claras en vez de tomar la iniciativa."
        ],
        "Reflexión y responsabilidad": [
            "Pienso antes de actuar, sobre todo si mis decisiones afectan a otros.",
            "Acepto la crítica como una oportunidad de mejora.",
            "Me motivo al ver que otros tienen éxito.",
            "Me gusta colaborar y compartir logros con otros."
        ],
        "Fortaleza y adaptabilidad": [
            "Me adapto fácilmente a nuevas dinámicas de grupo.",
            "Enfrento los conflictos directamente, sin evadirlos.",
            "Me esfuerzo constantemente por ser el mejor en lo que hago.",
            "Me siento motivado/a cuando compito con otros.",
            "Me siento más cómodo trabajando de forma individual.",
            "Prefiero trabajar solo/a para tener control total del resultado.",
        ]
    }
}
def figura_a_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def grafica_comparacion_clusters(X_scaled, y_true, y_pred, centroides=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Grupos reales
    for i in np.unique(y_true):
        ax[0].scatter(
            X_scaled[y_true == i, 0],
            X_scaled[y_true == i, 1],
            label=f"Grupo {i}",
            edgecolor='black'
        )
    ax[0].set_title("Grupos reales")
    ax[0].legend()

    # Clusters predichos
    for i in np.unique(y_pred):
        ax[1].scatter(
            X_scaled[y_pred == i, 0],
            X_scaled[y_pred == i, 1],
            label=f"Cluster {i}",
            edgecolor='black'
        )
    if centroides is not None:
        ax[1].scatter(
            centroides[:, 0], centroides[:, 1],
            c='black', s=200, marker='*', label='Centroides'
        )
    ax[1].set_title("Clusters predichos")
    ax[1].legend()

    fig.tight_layout()
    return figura_a_bytes(fig)

def grafica_confusion_matrix(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred, rownames=['Grupo real'], colnames=['Cluster predicho'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Matriz de confusión")
    return figura_a_bytes(fig)

def procesar(df, temas_elegidos, entrenar=True, modelo=None):
    df.columns = df.columns.str.strip()
    df_subs = pd.DataFrame()

    for tema in temas_elegidos:
        for sub, preguntas in subcategorias[tema].items():
            preguntas_existentes = [p for p in preguntas if p in df.columns]
            if preguntas_existentes:
                df_subs[sub] = df[preguntas_existentes].mean(axis=1)

    if df_subs.empty:
        st.error("No hay suficientes datos para calcular subcategorías.")
        return None

    X_scaled = scale(df_subs)

    if entrenar:
        k_fijo = len(df_subs.columns)
        modelo = KMeans(n_clusters=k_fijo, random_state=123, n_init=25)
        modelo.fit(X_scaled)

    y_pred = modelo.predict(X_scaled)
    df_resultados = df.copy()
    df_resultados = pd.concat([df_resultados, df_subs], axis=1)
    df_resultados["Cluster"] = y_pred

    centroides = modelo.cluster_centers_
    cluster_descripciones = {}
    asignadas = set()
    for i, centroide in enumerate(centroides):
        indices_ordenados = np.argsort(centroide)[::-1]
        nombre_sub = None
        for idx_sub in indices_ordenados:
            subcat = df_subs.columns[idx_sub]
            if subcat not in asignadas:
                nombre_sub = subcat
                asignadas.add(subcat)
                break
        if nombre_sub is None:
            nombre_sub = df_subs.columns[indices_ordenados[0]]
        cluster_descripciones[i] = f"{nombre_sub}"

    df_resultados["Descripción"] = df_resultados["Cluster"].map(cluster_descripciones)

    silhouette = silhouette_score(X_scaled, y_pred)
    davies = davies_bouldin_score(X_scaled, y_pred)

    return df_resultados, modelo, cluster_descripciones, X_scaled, silhouette, davies


def figura_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def grafica_clusters(X_scaled, y_pred, modelo, df_subs, cluster_descripciones, temas_elegidos):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in np.unique(y_pred):
        ax.scatter(
            x=X_scaled[y_pred == i, 0],
            y=X_scaled[y_pred == i, 1] if X_scaled.shape[1] > 1 else np.zeros_like(X_scaled[y_pred == i, 0]),
            c=plt.cm.tab10(i),
            marker='o',
            edgecolor='black',
            label=f"Cluster {i}: {cluster_descripciones[i]}"
        )

    ax.scatter(
        x=modelo.cluster_centers_[:, 0],
        y=modelo.cluster_centers_[:, 1] if X_scaled.shape[1] > 1 else np.zeros_like(modelo.cluster_centers_[:, 0]),
        c='black', s=200, marker='*', label='Centroides'
    )
    ax.set_xlabel(df_subs.columns[0])
    if X_scaled.shape[1] > 1:
        ax.set_ylabel(df_subs.columns[1])
    ax.set_title(f'Clusters KMeans - {" & ".join(temas_elegidos)}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    return figura_png(fig)

def grafica_barras(df_resultados):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Descripción', data=df_resultados, palette='Set2', ax=ax)
    ax.set_title("Cantidad de personas por categoría")
    plt.tight_layout()
    st.pyplot(fig)
    return figura_png(fig)

def grafica_metricas(silhouette, davies):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Silhouette", "Davies-Bouldin"], [silhouette, davies], color=['skyblue', 'salmon'])
    ax.set_ylabel("Valor")
    ax.set_title("Métricas del Modelo")
    plt.tight_layout()
    st.pyplot(fig)
    return figura_png(fig)

def grafica_comparacion(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pd.crosstab(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    ax.set_title("Comparación verdadero vs predicho")
    plt.tight_layout()
    st.pyplot(fig)
    return figura_png(fig)

def descargar_excel(df):
    output = io.BytesIO()
    df.to_excel(output, index=False)
    return output.getvalue()

def hacer_columnas_unicas(df):

    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{cols[idx]}.{i}"
    df.columns = cols
    return df
def main():
    st.title("Evaluación Psicológica Integral")
    menu = st.sidebar.radio("Menú", ["Entrenar modelo", "Usar modelo"])

    modelos_dir = "modelos"
    os.makedirs(modelos_dir, exist_ok=True)

    if menu == "Entrenar modelo":
        st.header("Entrenar modelo")
        archivo = st.file_uploader("Carga tu archivo CSV o Excel para entrenar", type=["csv", "xls", "xlsx"])
        if archivo:
            if archivo.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(archivo, encoding='utf-8')
                except UnicodeDecodeError:
                    archivo.seek(0)
                    df = pd.read_csv(archivo, encoding='latin1')
                except Exception as e:
                    st.error(f"Error al leer el CSV: {e}")
                    return
            elif archivo.name.lower().endswith((".xls", ".xlsx")):
                try:
                    df = pd.read_excel(archivo)
                except Exception as e:
                    st.error(f"Error al leer el Excel: {e}")
                    return
            else:
                st.error("Formato no soportado. Por favor sube un archivo .csv o .xlsx.")
                return

            st.subheader("Datos cargados")
            st.dataframe(df)

            # ===== Agregado: Estadísticas descriptivas =====
            st.subheader("Estadísticas descriptivas del dataset")
            stats_df = df.describe()
            st.dataframe(stats_df)

            temas = ["Personalidad", "Carácter", "Ambos"]
            seleccion = st.radio("Selecciona el tema para el entrenamiento:", temas)

            if seleccion == "Ambos":
                temas_elegidos = ["Personalidad", "Carácter"]
            else:
                temas_elegidos = [seleccion]

            subcats_posibles = []
            for tema in temas_elegidos:
                for sub in subcategorias[tema]:
                    subcats_posibles.append(f"{tema} > {sub}")

            subcats_elegidas = st.multiselect(
                "Opcional: elige las subcategorías específicas (si no eliges, se usarán todas del tema):",
                subcats_posibles,
                default=subcats_posibles
            )

            nombre_modelo = st.text_input(
                "Nombre del modelo",
                f"modelo_{'_'.join([t.lower() for t in temas_elegidos])}"
            )

            if st.button("Entrenar"):
                if subcats_elegidas and len(subcats_elegidas) < 2:
                    st.error("Por favor, selecciona al menos dos subcategorías o deja vacío para usar todas.")
                    return

                temas_elegidos_dict = {}
                if subcats_elegidas and len(subcats_elegidas) < len(subcats_posibles):
                    for item in subcats_elegidas:
                        tema, subcat = item.split(" > ")
                        if tema not in temas_elegidos_dict:
                            temas_elegidos_dict[tema] = []
                        temas_elegidos_dict[tema].append(subcat)
                else:
                    for tema in temas_elegidos:
                        temas_elegidos_dict[tema] = list(subcategorias[tema].keys())

                df.columns = df.columns.str.strip()
                df_subs = pd.DataFrame()

                for tema, subs in temas_elegidos_dict.items():
                    for sub in subs:
                        preguntas = subcategorias[tema][sub]
                        preguntas_existentes = [p for p in preguntas if p in df.columns]
                        if preguntas_existentes:
                            df_subs[sub] = df[preguntas_existentes].mean(axis=1)

                if df_subs.empty:
                    st.error("No hay suficientes datos para calcular subcategorías.")
                    return

                X_scaled = scale(df_subs)
                k_fijo = len(df_subs.columns)
                modelo = KMeans(n_clusters=k_fijo, random_state=123, n_init=25)
                modelo.fit(X_scaled)
                y_pred = modelo.predict(X_scaled)

                df_subs_renombrado = df_subs.rename(columns=lambda x: f"{x} (modelo)")

                df_resultados = pd.concat([df, df_subs_renombrado], axis=1)
                df_resultados["Cluster (modelo)"] = y_pred

                centroides = modelo.cluster_centers_
                cluster_descripciones = {}
                asignadas = set()
                for i, centroide in enumerate(centroides):
                    indices_ordenados = np.argsort(centroide)[::-1]
                    nombre_sub = None
                    for idx_sub in indices_ordenados:
                        subcat = df_subs.columns[idx_sub]
                        if subcat not in asignadas:
                            nombre_sub = subcat
                            asignadas.add(subcat)
                            break
                    if nombre_sub is None:
                        nombre_sub = df_subs.columns[indices_ordenados[0]]
                    cluster_descripciones[i] = f"{nombre_sub}"

                df_resultados["Descripción (modelo)"] = df_resultados["Cluster (modelo)"].map(cluster_descripciones)

                silhouette = silhouette_score(X_scaled, y_pred)
                davies = davies_bouldin_score(X_scaled, y_pred)

                modelo_path = os.path.join(modelos_dir, f"{nombre_modelo}.pkl")
                with open(modelo_path, "wb") as f:
                    pickle.dump((modelo, df_subs.columns.tolist()), f)
                st.success(f"Modelo guardado en: {modelo_path}")

                st.subheader("Resultados")
                st.dataframe(df_resultados)

                csv_data = df_resultados.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar resultados (.csv)",
                    data=csv_data,
                    file_name=f"{nombre_modelo}_resultados.csv",
                    mime="text/csv"
                )

                png_metrics = grafica_metricas(silhouette, davies)
                st.download_button("Descargar métricas del modelo", png_metrics, "metricas.png", "image/png")

                # ---- Nuevas gráficas ----
                if 'Cluster' in df.columns:
                    y_true = df['Cluster'].values
                    y_pred = df_resultados['Cluster'].values

                    st.subheader("Comparación visual entre clusters verdaderos y predichos")
                    png_comp = grafica_comparacion_clusters(X_scaled, y_true, y_pred, modelo.cluster_centers_)
                    st.image(png_comp)
                    st.download_button(
                        label="Descargar gráfica de comparación",
                        data=png_comp,
                        file_name="comparacion_clusters.png",
                        mime="image/png"
                    )

                    st.subheader("Matriz de confusión")
                    png_confusion = grafica_confusion_matrix(y_true, y_pred)
                    st.image(png_confusion)
                    st.download_button(
                        label="Descargar matriz de confusión",
                        data=png_confusion,
                        file_name="matriz_confusion.png",
                        mime="image/png"
                    )

                elif 'Descripción' in df.columns:
                    true_labels = pd.factorize(df['Descripción'])[0]
                    pred_labels = df_resultados['Cluster'].values
                    st.subheader("Comparación entre descripciones verdaderas y clusters predichos")
                    png_comp = grafica_comparacion(true_labels, pred_labels)
                    st.download_button(
                        label="Descargar gráfica de comparación",
                        data=png_comp,
                        file_name="comparacion_descripciones.png",
                        mime="image/png"
                    )

                    png_clusters = grafica_clusters(
                        X_scaled,
                        df_resultados["Cluster"].values,
                        modelo,
                        df_resultados[df_resultados.columns[-len(cluster_descripciones):]],
                        cluster_descripciones,
                        temas_elegidos
                    )
                    st.download_button("Descargar gráfica de clusters", png_clusters, "clusters.png", "image/png")

                    png_metrics = grafica_metricas(silhouette, davies)
                    st.download_button("Descargar métricas en gráfica", png_metrics, "metricas.png", "image/png")

                    png_barras = grafica_barras(df_resultados)
                    st.download_button("Descargar barras categorías", png_barras, "barras_categorias.png", "image/png")

                    csv_data = df_resultados.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar resultados (.csv)",
                        data=csv_data,
                        file_name=f"{nombre_modelo}_resultados.csv",
                        mime="text/csv"
                    )

    elif menu == "Usar modelo":
        st.header("Usar modelo entrenado")
        archivo = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xls", "xlsx"], key="usar_csv")

        modelos_disponibles = [f for f in os.listdir(modelos_dir) if f.endswith(".pkl")]
        if modelos_disponibles:
            modelo_seleccionado = st.selectbox("Elige un modelo entrenado", modelos_disponibles)
        else:
            st.warning("No cuentas con ningún modelo entrenado.")
            return

        if archivo and modelo_seleccionado:
            if archivo.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(archivo, encoding='utf-8')
                except UnicodeDecodeError:
                    archivo.seek(0)
                    df = pd.read_csv(archivo, encoding='latin1')
                except Exception as e:
                    st.error(f"Error al leer el CSV: {e}")
                    return
            elif archivo.name.lower().endswith((".xls", ".xlsx")):
                try:
                    df = pd.read_excel(archivo)
                except Exception as e:
                    st.error(f"Error al leer el Excel: {e}")
                    return
            else:
                st.error("Formato no soportado. Por favor sube un archivo .csv o .xlsx.")
                return

            st.subheader("Datos cargados")
            st.dataframe(df)

            with open(os.path.join(modelos_dir, modelo_seleccionado), "rb") as f:
                modelo, columnas_modelo = pickle.load(f)

            temas_elegidos = []
            for tema in subcategorias:
                for sub in subcategorias[tema]:
                    if sub in columnas_modelo:
                        temas_elegidos.append(tema)
                        break

            resultado = procesar(df, temas_elegidos, entrenar=False, modelo=modelo)
            if resultado:
                df_resultados, _, descripciones, X_scaled, silhouette, davies = resultado

                df_resultados = hacer_columnas_unicas(df_resultados)

                st.subheader("Resultados")
                df.columns = df.columns.str.strip()
                df_resultados.columns = df_resultados.columns.str.strip()

                preguntas_definidas = []
                for tema in subcategorias:
                    for subcat, preguntas in subcategorias[tema].items():
                        preguntas_definidas.extend(preguntas)

                cols_preguntas = [col for col in df.columns if col in preguntas_definidas]

                subcategorias_calculadas = list(subcategorias["Carácter"].keys()) + list(subcategorias["Personalidad"].keys())
                subcategorias_calculadas = [sub.strip() for sub in subcategorias_calculadas]
                cols_subcategorias = [col for col in df_resultados.columns if col in subcategorias_calculadas]

                cols_no_preguntas = [
                    col for col in df_resultados.columns
                    if col not in cols_preguntas and col not in cols_subcategorias and col not in ["Cluster", "Descripción Cluster"]
                ]

                cols_cluster = []
                if "Cluster" in df_resultados.columns:
                    cols_cluster.append("Cluster")
                if "Descripción Cluster" in df_resultados.columns:
                    cols_cluster.append("Descripción Cluster")

                opciones_filtro = {
                    "Preguntas individuales": cols_preguntas,
                    "Subcategorías calculadas": cols_subcategorias,
                    "Información básica (Nombre, ID, etc.)": cols_no_preguntas,
                    "Cluster": cols_cluster,
                }

                grupos_seleccionados = st.multiselect(
                    "Selecciona qué grupos de columnas quieres mantener:",
                    options=list(opciones_filtro.keys()),
                    default=list(opciones_filtro.keys())
                )

                columnas_finales = []
                for grupo in grupos_seleccionados:
                    columnas_finales.extend(opciones_filtro[grupo])

                st.dataframe(df_resultados[columnas_finales])

                csv_data = df_resultados[columnas_finales].to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Descargar resultados filtrados (.csv)",
                    data=csv_data,
                    file_name="resultados_filtrados.csv",
                    mime="text/csv"
                )

                if "Cluster" in df_resultados.columns:
                    y_pred = df_resultados["Cluster"].values

                    if "Cluster" in df.columns:
                        y_true = df["Cluster"].values

                        st.subheader("Comparación visual entre clusters verdaderos y predichos")
                        png_comp = grafica_comparacion_clusters(X_scaled, y_true, y_pred, modelo.cluster_centers_)
                        st.image(png_comp)
                        st.download_button(
                            label="Descargar gráfica de comparación",
                            data=png_comp,
                            file_name="comparacion_clusters.png",
                            mime="image/png"
                        )

                        st.subheader("Matriz de confusión")
                        png_confusion = grafica_confusion_matrix(y_true, y_pred)
                        st.image(png_confusion)
                        st.download_button(
                            label="Descargar matriz de confusión",
                            data=png_confusion,
                            file_name="matriz_confusion.png",
                            mime="image/png"
                        )

                png_clusters = grafica_clusters(
                    X_scaled,
                    df_resultados["Cluster"].values,
                    modelo,
                    df_resultados[df_resultados.columns[-len(descripciones):]],
                    descripciones,
                    temas_elegidos
                )
                st.download_button("Descargar gráfica de clusters", png_clusters, "clusters.png", "image/png")

                png_barras = grafica_barras(df_resultados)
                st.download_button("Descargar barras categorías", png_barras, "barras_categorias.png", "image/png")

                png_metrics = grafica_metricas(silhouette, davies)
                st.download_button("Descargar métricas del modelo", png_metrics, "metricas.png", "image/png")

if __name__ == "__main__":
    main()