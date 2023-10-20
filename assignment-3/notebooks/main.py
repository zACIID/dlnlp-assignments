# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment 3 - Clustering
#
# Perform classification of the [MNIST database](http://yann.lecun.com/exdb/mnist/) (or a sufficiently small subset of it) using:
#
# - mixture of Gaussians with diagonal covariance (Gaussian Naive Bayes with latent class label);
# - mean shift;
# - normalized cut.  
#
# The unsupervised classification must be performed at varying levels of dimensionality reduction through PCA (say going from 2 to 200) In order to asses the effect of the dimensionality in accuracy and learning time.  
#
# Provide the code and the extracted clusters as the number of clusters k varies from 5 to 15, for the mixture of Gaussians and normalized-cut, while for mean shift vary the kernel width. For each value of _k_ (or kernel width) provide the value of the Rand index:  
# $$
# R=2(a+b)/(n(n-1))
# $$
# where
#
# - _n_ is the number of images in the dataset.
# - _a_ is the number of pairs of images that represent the same digit and that are clustered together.
# - _b_ is the number of pairs of images that represent different digits and that are placed in different clusters.
#
# Explain the differences between the three models.
#
# **Tip:** the means of the Gaussian models can be visualized as a greyscale images after PCA reconstruction to inspect the learned model
#

# %% [markdown]
# ## Setup

# %%
# # %load_ext autoreload
# # %autoreload 2

import os
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io

import visualization.exploration as vis_expl
import visualization.mnist as vis_mnist
from classifiers.gmm import GMM
from classifiers.mean_shift import MeanShift
from classifiers.ncut import NCut
from pipeline.analysis.aggregate.a3_aggregate_analyzer import A3AggregateAnalyzer
from pipeline.analysis.datasets.dataset_analyzer import AnalysisPlotTypes
from pipeline.analysis.datasets.mnist_digits_analyzer import MNISTDigitsAnalyzer
from pipeline.analysis.models.gmm_evaluator import GMMEvaluator, internal_gmm_representation
from pipeline.analysis.models.mean_shift_evaluator import MeanShiftEvaluator, internal_mean_shift_representation
from pipeline.analysis.models.ncut_evaluator import NCutEvaluator
from pipeline.core import Pipeline, PipelineResults
from pipeline.dataset_providers.mnist_digits_provider import MNISTDigitsProvider
from pipeline.feature_engineering.pca_engineering import PCAEngineering
from src.pipeline.model_builder import ModelBuilder
from utils import serialization

INLINE_PLOT_BACKEND = "TkAgg"
NO_AUTO_PLOT_BACKEND = "Agg"

DATA_FRACTION = 0.1

# %%
IMAGES_FOLDER = os.path.abspath("../images")
if not os.path.exists(IMAGES_FOLDER):
    os.mkdir(IMAGES_FOLDER)

DATA_FOLDER = os.path.abspath("../data")
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

SERIALIZATION_FOLDER = os.path.abspath("../serialization")
if not os.path.exists(SERIALIZATION_FOLDER):
    os.mkdir(SERIALIZATION_FOLDER)

# %% [markdown]
# ## MNIST Dataset Exploration

# %%
expl_dataset_provider = MNISTDigitsProvider(
    storage_path=DATA_FOLDER,
    data_fraction=DATA_FRACTION,
    test_size=0.01
)
expl_dataset_provider.fetch_datasets()
exploration_dataset = expl_dataset_provider.get_training_dataset()

EXPLORATION_IMG_FOLDER = os.path.join(IMAGES_FOLDER, "exploration")
if not os.path.exists(EXPLORATION_IMG_FOLDER):
    os.mkdir(EXPLORATION_IMG_FOLDER)

# %% [markdown]
# ### DataFrame Exploration

# %%
exploration_dataset_df = pd.DataFrame(
    data=exploration_dataset.X[:, :],
    index=range(0, exploration_dataset.X.shape[0]),
    columns=(f"f{i}" for i in range(0, exploration_dataset.X.shape[1]))
)

# %%
exploration_dataset_df.info(verbose=True, show_counts=True)
exploration_dataset_df.describe().T

# %%
expl_y_series = pd.Series(data=exploration_dataset.y)
expl_y_series.info(show_counts=True)
expl_y_series.describe().T

# %% [markdown]
# ### 2D and 3D Visualization

# %% [markdown]
# 2D and 3D dataset plots to understand how data is distributed and clustered, even if in a lower dimensional space

# %%
data_3d_plot: go.Figure = vis_mnist.plot_3d(data=exploration_dataset)
data_3d_plot.show()
plotly.io.write_image(fig=data_3d_plot, width=2560, height=1440, file=os.path.join(EXPLORATION_IMG_FOLDER, "dataset-3d-plot.svg"))

# %%
data_2d_plot: plt.Figure = vis_mnist.plot_2d(data=exploration_dataset, figsize=(12, 12))
data_2d_plot.show()
data_2d_plot.savefig(os.path.join(EXPLORATION_IMG_FOLDER, "dataset-2d-plot.svg"), dpi=250, format="svg")

# %% [markdown]
# ### Label Counts

# %%
label_counts_plot: plt.Figure = vis_expl.label_counts_histogram(labels=exploration_dataset.y)
label_counts_plot.show()
label_counts_plot.savefig(os.path.join(EXPLORATION_IMG_FOLDER, "label-counts-plot.svg"), dpi=250, format="svg")

# %% [markdown]
# ### Mean Label Images

# %%
sample_images_plot: go.Figure = vis_mnist.plot_mean_cluster_images(
    dataset=exploration_dataset,
    top_k=10,
    figsize_pixels=(1280, 720)
)
sample_images_plot.show()
plotly.io.write_image(fig=sample_images_plot, width=1280, height=720, file=os.path.join(EXPLORATION_IMG_FOLDER, "sample-mean-images.svg"))

# %% [markdown]
# ### Feature Distribution Samples
#
# Feature distribution of some sampled features

# %%
matplotlib.use(NO_AUTO_PLOT_BACKEND)

mnist_analyzer = MNISTDigitsAnalyzer(
    feature_subplot_size=(1.5, 1.5),
    features_idx_to_plot=list(range(0, exploration_dataset.X.shape[1], 20))
)
analysis = mnist_analyzer.analyze(dataset=exploration_dataset)

# %%
# %matplotlib notebook
feat_distr_plots = analysis.plots[AnalysisPlotTypes.FEATURE_DISTRIBUTIONS]
feat_distr_plots.savefig(os.path.join(EXPLORATION_IMG_FOLDER, "feature-distros-plot.svg"), dpi=250, format="svg")
feat_distr_plots

# %% [markdown]
# ## Model Explorations: GMM, MeanShift

# %% [markdown]
# ### Pipeline
#
# The idea in this pipeline is to use the full dataset (so that mean images can be derived) to take a look at the internals of GMM and MeanShift models

# %%
model_exp_provider = MNISTDigitsProvider(
    storage_path=DATA_FOLDER,
    data_fraction=DATA_FRACTION,
)
model_exp_provider.fetch_datasets()

model_exp_dataset = model_exp_provider.get_training_dataset()
n_labels_expl: int = len(np.unique(exploration_dataset.y))
model_exp_pipeline = Pipeline(
    dataset_provider=model_exp_provider,
    feature_engineering_pipeline=[
    ],
    dataset_analyzers=[],
    model_builder=ModelBuilder(
        untrained_models=[
            GMM(n_mixtures=n_labels_expl),
            MeanShift(bandwidth=1),
        ]
    ),
    model_analyzers=[
        GMMEvaluator(),
        MeanShiftEvaluator(),
        NCutEvaluator()
    ],
    aggregate_analyzer=A3AggregateAnalyzer()
)

# %%
# %%time
matplotlib.use(NO_AUTO_PLOT_BACKEND)

results_path = os.path.join(SERIALIZATION_FOLDER, "model-expl-pipeline-res")
pipeline_results = None
if os.path.exists(results_path):
    pipeline_results = serialization.deserialize(type_=PipelineResults, filepath=results_path)
else:
    pipeline_results = model_exp_pipeline.execute()
    serialization.serialize(obj=pipeline_results, filepath=results_path)

len(pipeline_results.model_evaluations)

# %% [raw]
# print(pipeline_results.aggregate_analysis.plots)

# %% [markdown]
# ### Best and Worst Models Internals

# %%
internal_gmm_plot_best = internal_gmm_representation(
    evaluation=pipeline_results.aggregate_analysis.best_models[GMM]
)
plotly.io.write_image(fig=internal_gmm_plot_best, width=1280, height=720, file=os.path.join(EXPLORATION_IMG_FOLDER, "internal-gmm-best-means.svg"))

# %%
internal_gmm_plot_worst = internal_gmm_representation(
    evaluation=pipeline_results.aggregate_analysis.worst_models[GMM]
)
plotly.io.write_image(fig=internal_gmm_plot_worst, width=1280, height=720, file=os.path.join(EXPLORATION_IMG_FOLDER, "internal-gmm-worst-means.svg"))

# %%
internal_ms_plot_best = internal_mean_shift_representation(
    evaluation=pipeline_results.aggregate_analysis.best_models[MeanShift]
)
plotly.io.write_image(fig=internal_ms_plot_best, width=1280, height=720, file=os.path.join(EXPLORATION_IMG_FOLDER, "internal-ms-best-means.svg"))

# %%
internal_ms_plot_worst = internal_mean_shift_representation(
    evaluation=pipeline_results.aggregate_analysis.worst_models[MeanShift]
)
plotly.io.write_image(fig=internal_ms_plot_worst, width=1280, height=720, file=os.path.join(EXPLORATION_IMG_FOLDER, "internal-ms-worst-means.svg"))

# %%

# %% [markdown]
# ## Full Pipeline Execution: Multiple PCA, Multiple Configs of GMM, MeanShift, NCut

# %%
# %%time
matplotlib.use(NO_AUTO_PLOT_BACKEND)

from utils import serialization
from classifiers.base import BaseClassifier
from typing import List

PCA_DIMS = [2, 10, 25, 50, 100, 150, 200]
N_COMPONENTS = [5, 7, 10, 13, 15]
BANDWIDTHS = [1, 5, 15, 30, 50]

full_pipeline_results_by_pca_dim: Dict[int, PipelineResults] = {}
pcas = [
    PCAEngineering(
        n_components=n
    )
    for n in PCA_DIMS
]

def get_pca_res_serdes_path(n_components: int) -> str:
    return os.path.join(SERIALIZATION_FOLDER, f"pipeline-res_pca-{n_components}")

full_exec_provider = MNISTDigitsProvider(
    storage_path=DATA_FOLDER,
    data_fraction=DATA_FRACTION,
)
full_exec_provider.fetch_datasets()

for pca in pcas:
    models: List[BaseClassifier] = [
                 GMM(n_mixtures=k)
                 for k in N_COMPONENTS
             ] + [
                 MeanShift(bandwidth=k)
                 for k in BANDWIDTHS
             ] + [
                 NCut(n_components=k)
                 for k in N_COMPONENTS
             ]
    pipeline = Pipeline(
        dataset_provider=model_exp_provider,
        feature_engineering_pipeline=[pca],
        dataset_analyzers=[],
        model_builder=ModelBuilder(
            untrained_models=models
        ),
        model_analyzers=[
            GMMEvaluator(),
            MeanShiftEvaluator(),
            NCutEvaluator()
        ],
        aggregate_analyzer=A3AggregateAnalyzer()
    )

    results_path = get_pca_res_serdes_path(pca.n_components)
    results = None
    if os.path.exists(results_path):
        results = serialization.deserialize(type_=PipelineResults, filepath=results_path)
    else:
        results = pipeline.execute()
        serialization.serialize(obj=results, filepath=results_path)


    full_pipeline_results_by_pca_dim[pca.n_components] = results


# %%
full_pipeline_results_by_pca_dim = (
    full_pipeline_results_by_pca_dim
    if len(full_pipeline_results_by_pca_dim) > 0
    else {}
)
if len(full_pipeline_results_by_pca_dim) == 0:
    for k in PCA_DIMS:
        full_pipeline_results_by_pca_dim[k] = serialization.deserialize(
            type_=PipelineResults,
            filepath=get_pca_res_serdes_path(k)
        )

# %% [markdown]
# ## Aggregate Plots

# %%
# %load_ext autoreload
# %autoreload

# %%
# # %matplotlib notebook

# Load it here so it is for sure the same class loaded by the Pipeline
# Autoreload kinda fucks up imports because you might end up with two versions of the same import,
#   causing the enum not to match
from pipeline.analysis.aggregate.a3_aggregate_analyzer import A3AggregatePlotTypes, A3AggregateAnalyzer

AGGR_ANALYSIS_FOLDER = os.path.join(IMAGES_FOLDER, "aggregate")
if not os.path.exists(AGGR_ANALYSIS_FOLDER):
    os.mkdir(AGGR_ANALYSIS_FOLDER)

full_results_analyzer = A3AggregateAnalyzer()
all_evaluations = [
    e
    for p in full_pipeline_results_by_pca_dim.values()
    for e in p.model_evaluations
]
full_aggr_analysis = full_results_analyzer.analyze(
    evaluations=all_evaluations
)

# %% [markdown]
# ### Training Times

# %%
all_training_times_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.ALL_MODEL_TRAINING_TIMES_BY_DIM]
all_training_times_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "all-tr-times.svg"), dpi=250, format="svg")

# %%
gmm_training_times_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.GMM_TRAINING_TIMES_BY_DIM]
gmm_training_times_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "gmm-tr-times.svg"), dpi=250, format="svg")

# %%
ms_training_times_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.MEAN_SHIFT_TRAINING_TIMES_BY_DIM]
ms_training_times_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ms-tr-times.svg"), dpi=250, format="svg")

# %%
ncut_training_times_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.NCUT_TRAINING_TIMES_BY_DIM]
ncut_training_times_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ncut-tr-times.svg"), dpi=250, format="svg")

# %% [markdown]
# ### Rand Index

# %%
all_rand_idx_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.ALL_MODEL_RAND_INDEX_BY_DIM]
all_rand_idx_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "all-rand-idx.svg"), dpi=250, format="svg")

# %%
gmm_rand_idx_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.GMM_RAND_INDEX_BY_DIM]
gmm_rand_idx_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "gmm-rand_idx.svg"), dpi=250, format="svg")

# %%
ms_rand_idx_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.MEAN_SHIFT_RAND_INDEX_BY_DIM]
ms_rand_idx_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ms-rand_idx.svg"), dpi=250, format="svg")

# %%
ncut_rand_idx_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.NCUT_RAND_INDEX_BY_DIM]
ncut_rand_idx_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ncut-rand_idx.svg"), dpi=250, format="svg")

# %% [markdown]
# ### Number of Clusters

# %%
all_clusters_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.ALL_N_CLUSTERS_BY_DIM]
all_clusters_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "all-n-clusters.svg"), dpi=250, format="svg")

# %%
gmm_n_clusters_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.GMM_N_CLUSTERS_BY_DIM]
gmm_n_clusters_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "gmm-n-clusters.svg"), dpi=250, format="svg")

# %%
ms_n_clusters_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.MEAN_SHIFT_N_CLUSTERS_BY_DIM]
ms_n_clusters_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ms-n-clusters.svg"), dpi=250, format="svg")

# %%
ncut_n_clusters_full: plt.Figure = full_aggr_analysis.plots[A3AggregatePlotTypes.NCUT_N_CLUSTERS_BY_DIM]
ncut_n_clusters_full.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "ncut-n-clusters.svg"), dpi=250, format="svg")

# %% [markdown]
# ### Mean Predictions by Model - Best and Worst

# %%
from pipeline.analysis.models.model_evaluator import PlotType

# %% [markdown]
# #### GMM

# %%
gmm_best_full: go.Figure = full_aggr_analysis.best_models[GMM].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=gmm_best_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "gmm-means-best.svg"))

# %%

# %%
gmm_worst_full: go.Figure = full_aggr_analysis.worst_models[GMM].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=gmm_worst_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "gmm-means-worst.svg"))

# %% [markdown]
# #### MeanShift

# %%
ms_best_full: go.Figure = full_aggr_analysis.best_models[MeanShift].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=ms_best_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "ms-means-best.svg"))

# %%
ms_worst_full: go.Figure = full_aggr_analysis.worst_models[MeanShift].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=ms_worst_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "ms-means-worst.svg"))

# %% [markdown]
# #### NCut

# %%
ncut_best_full: go.Figure = full_aggr_analysis.best_models[NCut].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=ncut_best_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "ncut-means-best.svg"))

# %%
ncut_worst_full: go.Figure = full_aggr_analysis.worst_models[NCut].plots[PlotType.MEAN_CLASSIFICATIONS_PLOT]
plotly.io.write_image(fig=ncut_worst_full, width=1280, height=720, file=os.path.join(AGGR_ANALYSIS_FOLDER, "ncut-means-worst.svg"))

# %% [markdown]
# ### Other

# %%
label_counts_plot: plt.Figure = vis_expl.label_counts_histogram(labels=full_exec_provider.get_training_dataset().y)
label_counts_plot.show()
label_counts_plot.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "training-label-counts-full.svg"), dpi=250, format="svg")

# %%
label_counts_plot: plt.Figure = vis_expl.label_counts_histogram(labels=full_exec_provider.get_testing_dataset().y)
label_counts_plot.show()
label_counts_plot.savefig(os.path.join(AGGR_ANALYSIS_FOLDER, "testing-label-counts-full.svg"), dpi=250, format="svg")

# %%
