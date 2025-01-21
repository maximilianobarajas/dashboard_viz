import os
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from PIL import Image
from generarinformacion import GenerarInformacion
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def list_files_in_directory(directory="src/data"):
    try:
        return [f for f in os.listdir(directory) if f.endswith(".jsonl")]
    except FileNotFoundError:
        return []

def list_images_in_folder(folder="Informacion"):
    try:
        image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
        return [f for f in os.listdir(folder) if any(f.endswith(ext) for ext in image_extensions)]
    except FileNotFoundError:
        return []

def estimate_distributions(df, keys, sampler, num_samples):
    results = {}

    for key in keys:
        with pm.Model() as model:
            if key == "delegacion":
                categories = df[key].unique()
                probabilities = pm.Dirichlet("probabilities", a=np.ones(len(categories)))
                likelihood = pm.Categorical("likelihood", p=probabilities, observed=df[key].astype("category").cat.codes)
            elif key in ["precio", "superficie_de_terreno", "superficie_construida"]:
                mu = pm.Normal("mu", mu=df[key].mean(), sigma=df[key].std())
                sigma = pm.HalfNormal("sigma", sigma=df[key].std())
                likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=df[key])
            elif key in ["recamaras", "baños"]:
                lambda_param = pm.Exponential("lambda", lam=1/df[key].mean())
                likelihood = pm.Poisson("likelihood", mu=lambda_param, observed=df[key])

            trace = pm.sample(num_samples, tune=1000, return_inferencedata=True, step=sampler, progressbar=True)
            results[key] = trace

            fig = az.plot_trace(trace, var_names=["mu", "sigma"], figsize=(12, 6), compact=True)
            plt.suptitle(f"Posterior Distributions of Parameters for {key} \n", fontsize=14, color="darkgrey")
            st.pyplot(fig)

            summary = az.summary(trace)
            st.subheader(f"Summary Table for {key}")
            st.dataframe(summary)

    return results

def perform_clustering(df, clustering_technique, reduction_technique, reduction_params):
    features = ['precio', 'superficie_de_terreno', 'superficie_construida', 'recamaras', 'baños', 'estacionamiento']
    X = df[features]
    X = MinMaxScaler().fit_transform(X)

    if clustering_technique == "DBSCAN":
        eps = reduction_params.get("eps", 0.5)
        min_samples = reduction_params.get("min_samples", 5)
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_technique == "KMeans":
        n_clusters = reduction_params.get("n_clusters", 5)
        clustering_model = KMeans(n_clusters=n_clusters)
    else:
        raise ValueError("Unsupported clustering technique")

    clusters = clustering_model.fit_predict(X)
    df['cluster'] = clusters

    if reduction_technique == "UMAP":
        n_components = reduction_params.get("n_components", 2)
        reducer = umap.UMAP(n_components=n_components)
        embedding = reducer.fit_transform(X)
    elif reduction_technique == "PCA":
        n_components = reduction_params.get("n_components", 2)
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(X)
    elif reduction_technique == "TSNE":
        n_components = reduction_params.get("n_components", 2)
        perplexity = reduction_params.get("perplexity", 30)
        reducer = TSNE(n_components=n_components, perplexity=perplexity)
        embedding = reducer.fit_transform(X)
    else:
        raise ValueError("Unsupported reduction technique")

    if embedding.shape[1] == 2:
        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1], color=df['cluster'].astype(str),
            labels={'color': 'Cluster'}, title=f"Clusters Visualized with {reduction_technique}",
            template="plotly_dark"
        )
        st.plotly_chart(fig)
    elif embedding.shape[1] == 3:
        fig = px.scatter_3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            color=df['cluster'].astype(str),
            labels={'color': 'Cluster'}, title=f"Clusters Visualized with {reduction_technique} (3D)",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

st.set_page_config(page_title="Data Plots, Clustering, and Bayesian Analysis", layout="wide")

page = st.sidebar.selectbox("Select Page", ["Clustering", "Plots", "Bayesian Distribution Estimation"])

if page == "Clustering":
    st.title("Clustering Analysis")
    with st.sidebar.form("clustering_form"):
        files = list_files_in_directory()
        selected_file = st.selectbox("Select JSONL File for Clustering", options=files)
        clustering_technique = st.selectbox("Select Clustering Technique", ["DBSCAN", "KMeans"])
        reduction_technique = st.selectbox("Select Reduction Technique", ["UMAP", "PCA", "TSNE"])

        if clustering_technique == "DBSCAN":
            eps = st.number_input("DBSCAN: eps", value=0.5, min_value=0.1, step=0.1)
            min_samples = st.number_input("DBSCAN: min_samples", value=5, min_value=1, step=1)
        elif clustering_technique == "KMeans":
            n_clusters = st.number_input("KMeans: n_clusters", value=5, min_value=2, step=1)

        if reduction_technique == "UMAP":
            n_components = st.number_input("UMAP: n_components", value=2, min_value=2, max_value=3, step=1)
        elif reduction_technique == "PCA":
            n_components = st.number_input("PCA: n_components", value=2, min_value=2, max_value=3, step=1)
        elif reduction_technique == "TSNE":
            n_components = st.number_input("TSNE: n_components", value=2, min_value=2, max_value=3, step=1)
            perplexity = st.number_input("TSNE: perplexity", value=30, min_value=5, step=5)

        submitted = st.form_submit_button("Submit")

    if submitted:
        if selected_file:
            full_path = os.path.join("src", "data", selected_file)
            df = pd.read_json(full_path, lines=True)

            st.write("Preview of Data:")
            st.dataframe(df.head())

            st.subheader("Clustering Results")
            reduction_params = {
                "n_components": n_components,
                "perplexity": perplexity if reduction_technique == "TSNE" else None,
                "eps": eps if clustering_technique == "DBSCAN" else None,
                "min_samples": min_samples if clustering_technique == "DBSCAN" else None,
                "n_clusters": n_clusters if clustering_technique == "KMeans" else None
            }
            perform_clustering(df, clustering_technique, reduction_technique, reduction_params)
        else:
            st.warning("Please select a file.")

elif page == "Plots":
    st.title("Plots of Generated Data")
    with st.sidebar.form("plots_form"):
        files = list_files_in_directory()
        selected_file = st.selectbox("Select JSONL File for Plots", options=files)
        folder_name = st.text_input("Folder Name (Optional)", value="Informacion")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if selected_file:
            full_path = os.path.join("src", "data", selected_file)
            GenerarInformacion(full_path, folder_name)

            image_files = list_images_in_folder(folder_name)

            if image_files:
                st.subheader("Generated Images")
                cols = st.columns(3)
                for idx, img in enumerate(image_files):
                    img_path = os.path.join(folder_name, img)
                    cols[idx % 3].image(img_path, caption=img, use_column_width=True)

            else:
                st.warning("No images found in the folder.")
        else:
            st.warning("Please select a file.")

elif page == "Bayesian Distribution Estimation":
    st.title("Bayesian Distribution Estimation")
    with st.sidebar.form("bayesian_form"):
        files = list_files_in_directory()
        selected_file = st.selectbox("Select JSONL File for Bayesian Analysis", options=files)
        
        # Change multiselect to selectbox for single key selection
        key_to_estimate = st.selectbox(
            "Select a Key to Estimate Distributions",
            options=["delegacion", "precio", "superficie_de_terreno", "superficie_construida", "recamaras", "baños"]
        )
        
        sampler_choice = st.selectbox("Select Sampler", ["Metropolis", "NUTS", "HamiltonianMC"])
        num_samples = st.number_input("Number of Samples", value=2000, min_value=500, step=500)
        submitted = st.form_submit_button("Submit")

    if submitted:
        if selected_file:
            full_path = os.path.join("src", "data", selected_file)
            df = pd.read_json(full_path, lines=True)

            st.write("Preview of Data:")
            st.dataframe(df.head())

            if key_to_estimate:
                st.subheader("Bayesian Estimation Results")

                # Define the model context inside `estimate_distributions`
                def estimate_distribution_with_context(df, key, sampler_choice, num_samples):
                    with pm.Model() as model:
                        if key == "delegacion":
                            categories = df[key].unique()
                            probabilities = pm.Dirichlet("probabilities", a=np.ones(len(categories)))
                            likelihood = pm.Categorical("likelihood", p=probabilities, observed=df[key].astype("category").cat.codes)
                        elif key in ["precio", "superficie_de_terreno", "superficie_construida"]:
                            mu = pm.Normal("mu", mu=df[key].mean(), sigma=df[key].std())
                            sigma = pm.HalfNormal("sigma", sigma=df[key].std())
                            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=df[key])
                        elif key in ["recamaras", "baños"]:
                            lambda_param = pm.Exponential("lambda", lam=1/df[key].mean())
                            likelihood = pm.Poisson("likelihood", mu=lambda_param, observed=df[key])

                        # Map sampler choice
                        if sampler_choice == "Metropolis":
                            sampler = pm.Metropolis()
                        elif sampler_choice == "NUTS":
                            sampler = pm.NUTS()
                        elif sampler_choice == "HamiltonianMC":
                            sampler = pm.HamiltonianMC()

                        trace = pm.sample(num_samples, tune=1000, return_inferencedata=True, step=sampler, progressbar=True)

                        # Plot using ArviZ and Streamlit
                        st.write(f"Posterior Distributions for {key}")
                        az.plot_trace(trace)
                        st.pyplot()

                        summary = az.summary(trace)
                        st.subheader(f"Summary Table for {key}")
                        st.dataframe(summary)

                estimate_distribution_with_context(df, key_to_estimate, sampler_choice, num_samples)
            else:
                st.warning("Please select a key to estimate distributions.")
        else:
            st.warning("Please select a file.")
