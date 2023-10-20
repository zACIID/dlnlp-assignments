# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Discriminative and Generative Classifiers
#
# Write a handwritten digit classifier for the MNIST database. These are composed of 70000 28x28 pixel gray-scale images of handwritten digits divided into 60000 training set and 10000 test set.
# In Python you can automatically fetch the dataset from the net and load it using the following code:
#
# ```Python
# from sklearn.datasets import fetch_openml
# X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
# y = y.astype(int)
# X = X/255.
# ```
#
# This will result in 784-dimensional feature vectors (28*28) of values between 0 (white) and 1 (black).
#
# Train the following classifiers on the dataset:
#
# 1. SVM  using linear, polynomial of degree 2, and RBF kernels;
# 2. Random forests
# 3. Naive Bayes classifier where each pixel is distributed according to a Beta distribution of parameters $α, β$:
# 4. k-NN
#
# You can use _scikit-learn_ or any other library for SVM and random forests, but you must implement the Naive Bayes and k-NN classifiers yourself.
#
# Use 10 way cross validation to optimize the parameters for each classifier.
#
# Provide the code, the models on the training set, and the respective performances in testing and in 10 way cross validation.
#
# Explain the differences between the models, both in terms of classification performance, and in terms of computational requirements (timings) in training and in prediction.

# %% [markdown]
# ## Hint
#
# P.S. For a discussion about maximum likelihood for the parameters of a beta distribution you can look here. However, for this assignment the estimators obtained with he moments approach will be fine:
#
# $$
# \begin{aligned}
#     & \alpha = KE[X] \\
#     & \beta = K(1 - E[X]) \\
#     & K = \frac{
#         E[X](1 - E[X])
#     }{
#         Var[X]
#     } - 1
# \end{aligned}
# $$
#
# Note: $\alpha/(\alpha + \beta)$ is the mean of the beta distribution. if you compute the mean for each of the 784 models and reshape them into 28x28 images you can have a visual indication of what the model is learning.

# %% [markdown]
# ## Setup

# %%
import datetime
import sys
import os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../../"))  # Add project root to path, else src imports won't work

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as mtr
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from loguru import logger

import src.utils.training as tr
import src.utils.serialization as ser
import src.utils.time_utils as tut
import src.classifiers as cls

RND_SEED = 23

# %%
images_folder = os.path.abspath("../../../images")
if not os.path.exists(images_folder):
    os.mkdir(images_folder)

# %% [markdown]
# ## MNIST Data

# %% [markdown]
# ### Fetching

# %%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X: pd.DataFrame = X / 255.0
y: pd.Series = y.astype(int)

# %% [markdown]
# Important to note that there are no null values, so there is not much cleaning needed here.
# Additionally, features are in the 0 to 1 range, and it can be seen that some of them consists of only 0s, possibly representing some padding pixels that are always empty.

# %% [markdown]
# ### Missing Values

# %%
X.info(verbose=True, null_counts=True)
X.describe().T

# %%
X

# %% [markdown]
# Even here it can be seen that there are no null values, and that labels are numbers from 0 to 9, each representing the corresponding digit.

# %%
y.info(verbose=True)
y.describe().T

# %%
y

# %% [markdown]
# ### Digit Plots

# %%
# Plot the first k samples of each digit
k = 3
X_plot = [
    # Reshape into 28x28 image matrices
    X[y.values == digit].to_numpy()[:k, :].reshape(k, 28, 28)
    for digit in range(0, 9+1)
]

# All digit arrays of length 10 are flattened into the same level
flatten_one_level = []
for digit_samples in X_plot:
    flatten_one_level += digit_samples.tolist()
X_plot = np.array(flatten_one_level)
print(X_plot.shape)

fig = px.imshow(
    img=X_plot,
    color_continuous_scale="gray",
    binary_string=True,
    width=800,
    height=600,
    facet_col=0,
    facet_col_wrap=6,
)
fig.for_each_annotation(lambda a: a.update(text=""))
fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showticklabels=False)
fig.show()
fig.write_image(os.path.join(images_folder, "digit-samples.png"), format="png", width=1920, height=1080)

# %% [markdown]
# ### Feature Engineering

# %% [markdown]
# #### Number of samples per digit
#
# The thing to note here is that the dataset is fairly balanced: every class is pretty much equal in proportion to the others.

# %%
unique_labels = np.unique(y)

samples_per_digit = {
    "digit": unique_labels,
    "tot_samples": [
        len(X[y == digit])
        for digit in unique_labels
    ]
}
samples_per_digit_df = pd.DataFrame(data=samples_per_digit)
samples_per_digit_df

# %% [markdown]
# #### Removing All-zero features

# %%
n_cols_before = len(X.columns)

all_cols = list(X.columns)
for col in all_cols:
    if (X[col] == 0).all():
        X = X.drop(col, axis="columns")

n_cols_after = len(X.columns)
print(f"Removed {n_cols_before - n_cols_after} columns containing only 0")

# %%
X.describe().T.sort_values(by=["std", "mean", "max"])

# %% [markdown]
# ### Train and Test Split
#
# The dataset is already divided (pre-ordered) in 60k training data points and 10k testing data points. Each point consists of a 784-dimensional vector, which represents a 28x28 digit image.

# %%
X_train = X[:60000].to_numpy()
y_train = y[:60000].to_numpy()
X_test =  X[60000:].to_numpy()
y_test =  y[60000:].to_numpy()

# %% [markdown]
# #### Downsampling

# %% [markdown]
# I am downsampling the dataset to 10% to reduce training time for classifiers like SVC, whose training complexity is quadratic to cubic, with respect to the number of samples and feature dimension, as well as requiring a lot of space in memory because of the need to store some kind of matrix, usually a square matrix with *n_samples* rows and columns. Downsampled training sets, in particular, are used in the hyperparameters tuning phase, because it involves the fitting of a lot of classifiers.

# %%
X_train_small, y_train_small = resample(
    X_train, y_train,
    n_samples=6000, stratify=y_train,  # Stratify with respect to the class labels, y
    random_state=RND_SEED
)

# %% [markdown]
# ## Tuning Discriminative Classifiers: SVM, Random Forest, KNN

# %%
N_JOBS = 7
VERBOSITY = 5

# %%
serialization_folder = os.path.abspath("../../../serialization")

if not os.path.exists(serialization_folder):
    os.mkdir(serialization_folder)


# %% [markdown]
# ### Random Forest

# %%
@tut.print_time_perf
def tune_rf() -> Tuple[float, RandomForestClassifier]:
    rf_search_params = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [100, 150, 200],
        "min_samples_leaf": [50, 100, 200],
        "max_leaf_nodes": [2, 10, 25, 50, 100],
    }

    logger.info("Tuning RF Classifier...")
    rf = RandomForestClassifier(n_jobs=N_JOBS, random_state=RND_SEED)
    search_result = tr.grid_search_cv_tuning(
        model=rf,
        train_data=X_train,
        train_target=y_train,
        hyper_params=rf_search_params,
        scoring="accuracy",  # F-Measure, harmonic mean of Precision and Recall
        k_folds=10,
        n_jobs=N_JOBS,
        verbosity=VERBOSITY,
        refit=False  # Manual refit below
    )
    logger.info("Tuning results:")
    logger.info(f"Best params: {search_result.best_params_}")

    logger.info("Fitting with best params and full training set...")
    @tut.get_execution_time  # Return execution time as well
    def fit_best_rf() -> Tuple[float, RandomForestClassifier]:
        rf_best = RandomForestClassifier(
            n_jobs=N_JOBS,
            random_state=RND_SEED,
            **search_result.best_params_
        )
        rf_best.fit(X=X_train, y=y_train)

        return rf_best

    return fit_best_rf()


# %%
tuned_rf_filepath = os.path.join(serialization_folder, "tuned_rf.ser")
rf_time, tuned_rf = ser.deserialize_or_create_object(
    type_=Tuple[float, RandomForestClassifier],
    filepath=tuned_rf_filepath,
    builder=tune_rf
)


# %% [markdown]
# ### SVC

# %%
# "linear", "poly", "rbf"
@tut.print_time_perf
def tune_svc(kernel: str) -> Tuple[float, SVC]:
    svc_search_params = {
        "C": [0.05, 0.2, 0.5, 1, 2.5],
        "kernel": [kernel],
        "degree": [2],
    }

    logger.info("Tuning SVC...")
    svc = SVC(random_state=RND_SEED)
    search_result = tr.grid_search_cv_tuning(
        model=svc,
        train_data=X_train_small,
        train_target=y_train_small,
        hyper_params=svc_search_params,
        scoring="accuracy",  # F-Measure, harmonic mean of Precision and Recall
        k_folds=10,
        n_jobs=N_JOBS,
        verbosity=VERBOSITY,
        refit=False  # Manual refit below
    )
    logger.info("Tuning results:")
    logger.info(f"Best params: {search_result.best_params_}")

    logger.info("Fitting with best params and full training set...")

    @tut.get_execution_time  # Return execution time as well
    def fit_best_svc() -> SVC:
        svc_best = SVC(
            random_state=RND_SEED,
            **search_result.best_params_
        )
        svc_best.fit(X_train, y_train)

        return svc_best

    return fit_best_svc()


# %%
tuned_svc_linear_filepath = os.path.join(serialization_folder, "tuned_svc_linear.ser")
svc_linear_time, tuned_svc_linear = ser.deserialize_or_create_object(
    type_=Tuple[float, SVC],
    filepath=tuned_svc_linear_filepath,
    builder=lambda: tune_svc("linear")
)

# %%
tuned_svc_poly_filepath = os.path.join(serialization_folder, "tuned_svc_poly.ser")
svc_poly_time, tuned_svc_poly = ser.deserialize_or_create_object(
    type_=Tuple[float, SVC],
    filepath=tuned_svc_poly_filepath,
    builder=lambda: tune_svc("poly")
)

# %%
tuned_svc_rbf_filepath = os.path.join(serialization_folder, "tuned_svc_rbf.ser")
svc_rbf_time, tuned_svc_rbf = ser.deserialize_or_create_object(
    type_=Tuple[float, SVC],
    filepath=tuned_svc_rbf_filepath,
    builder=lambda: tune_svc("rbf")
)

# %% [markdown]
# ### KNN

# %%
from typing import Tuple

def tune_knn() -> Tuple[float, cls.KNN]:
    knn_search_params = {
        "k": [5, 10, 25, 50, 100, 200],
    }

    logger.info("Tuning KNN...")
    knn = cls.KNN()
    search_result = tr.grid_search_cv_tuning(
        model=knn,
        train_data=X_train_small,
        train_target=y_train_small,
        hyper_params=knn_search_params,
        scoring="accuracy",  # F-Measure, harmonic mean of Precision and Recall
        k_folds=10,
        n_jobs=N_JOBS,
        verbosity=VERBOSITY,
        refit=False,  # Manual refit below
    )
    logger.info("Tuning results:")
    logger.info(f"Best params: {search_result.best_params_}")

    logger.info("Fitting with best params and full training set...")
    @tut.get_execution_time  # Return execution time as well
    def fit_best_knn():
        knn_best = cls.KNN(
            **search_result.best_params_
        )
        knn_best.fit(X_train, y_train)

        return knn_best

    return fit_best_knn()


# %% [markdown]
# ## Tuning Generative Classifiers: Naive Bayes

# %%
tuned_knn_filepath = os.path.join(serialization_folder, "tuned_knn.ser")
knn_time, tuned_knn = ser.deserialize_or_create_object(
    type_=Tuple[float, cls.KNN],
    filepath=tuned_knn_filepath,
    builder=lambda: tune_knn()
)


# %% [markdown]
# ### Naive Bayes

# %%
@tut.get_execution_time
def tune_beta_naive_bayes() -> cls.BetaNaiveBayes:
    return cls.BetaNaiveBayes().fit(X_train, y_train)


# %%
naive_bayes_filepath = os.path.join(serialization_folder, "tuned_beta_naive_bayes.ser")
bayes_time, tuned_bayes = ser.deserialize_or_create_object(
    type_=Tuple[float, cls.BetaNaiveBayes],
    filepath=naive_bayes_filepath,
    builder=tune_beta_naive_bayes
)

# %% [markdown]
# ## Serializing Execution Times

# %%
ex_times_path = os.path.join(serialization_folder, "execution_times.csv")
if not os.path.exists(ex_times_path):
    times = {
        "model": ["rf", "svc_linear", "svc_poly", "svc_rbf", "knn", "beta_bayes"],
        "training_time": [rf_time, svc_linear_time, svc_poly_time, svc_rbf_time, knn_time, bayes_time],
    }
    times_df = pd.DataFrame(data=times)
    times_df.to_csv(path_or_buf=ex_times_path, index=False)
else:
    times_df = pd.read_csv(ex_times_path)


# %% [markdown]
# ## Comparisons and Evaluation
#
# Comparisons between models are mainly between performances on the test set, but training set might also be taken into account to check if there is any interesting or unusual behavior.

# %%
@tut.get_execution_time
def predict_rf(X_: np.ndarray | pd.DataFrame):
    return tuned_rf.predict(X_)

@tut.get_execution_time
def predict_svc_linear(X_: np.ndarray | pd.DataFrame):
    return tuned_svc_linear.predict(X_)

@tut.get_execution_time
def predict_svc_poly(X_: np.ndarray | pd.DataFrame):
    return tuned_svc_poly.predict(X_)

@tut.get_execution_time
def predict_svc_rbf(X_: np.ndarray | pd.DataFrame):
    return tuned_svc_rbf.predict(X_)

@tut.get_execution_time
def predict_knn(X_: np.ndarray):
    return tuned_knn.predict(X_)

@tut.get_execution_time
def predict_bayes(X_: np.ndarray):
    return tuned_bayes.predict(X_)


# %%
def get_predictions() -> Dict[str, Dict[str, np.ndarray]]:
    print("Training Set Predictions")
    print(datetime.datetime.now())
    rf_pred_train_time, rf_pred_train = predict_rf(X_train_small)
    svc_linear_pred_train_time, svc_linear_pred_train = predict_svc_linear(X_train_small)
    svc_poly_pred_train_time, svc_poly_pred_train = predict_svc_poly(X_train_small)
    svc_rbf_pred_train_time, svc_rbf_pred_train = predict_svc_rbf(X_train_small)
    knn_pred_train_time, knn_pred_train = predict_knn(X_train_small)
    bayes_pred_train_time, bayes_pred_train = predict_bayes(X_train_small)
    print(datetime.datetime.now())

    print("Testing Set Predictions")
    print(datetime.datetime.now())
    rf_pred_test_time, rf_pred_test = predict_rf(X_test)
    svc_linear_pred_test_time, svc_linear_pred_test = predict_svc_linear(X_test)
    svc_poly_pred_test_time, svc_poly_pred_test = predict_svc_poly(X_test)
    svc_rbf_pred_test_time, svc_rbf_pred_test = predict_svc_rbf(X_test)
    knn_pred_test_time, knn_pred_test = predict_knn(X_test)
    bayes_pred_test_time, bayes_pred_test = predict_bayes(X_test)
    print(datetime.datetime.now())

    return {
        "rf": {
            "y_train": y_train_small,
            "y_train_preds": rf_pred_train,
            "y_train_preds_time": rf_pred_train_time,
            "y_test": y_test,
            "y_test_preds": rf_pred_test,
            "y_test_preds_time": rf_pred_test_time,
        },
        "svc_linear": {
            "y_train": y_train_small,
            "y_train_preds": svc_linear_pred_train,
            "y_train_preds_time": svc_linear_pred_train_time,
            "y_test": y_test,
            "y_test_preds": svc_linear_pred_test,
            "y_test_preds_time": svc_linear_pred_test_time,
        },
        "svc_poly": {
            "y_train": y_train_small,
            "y_train_preds": svc_poly_pred_train,
            "y_train_preds_time": svc_poly_pred_train_time,
            "y_test": y_test,
            "y_test_preds": svc_poly_pred_test,
            "y_test_preds_time": svc_poly_pred_test_time,
        },
        "svc_rbf": {
            "y_train": y_train_small,
            "y_train_preds": svc_rbf_pred_train,
            "y_train_preds_time": svc_rbf_pred_train_time,
            "y_test": y_test,
            "y_test_preds": svc_rbf_pred_test,
            "y_test_preds_time": svc_rbf_pred_test_time,
        },
        "knn": {
            "y_train": y_train_small,
            "y_train_preds": knn_pred_train,
            "y_train_preds_time": knn_pred_train_time,
            "y_test": y_test,
            "y_test_preds": knn_pred_test,
            "y_test_preds_time": knn_pred_test_time,
        },
        "beta_bayes": {
            "y_train": y_train_small,
            "y_train_preds": bayes_pred_train,
            "y_train_preds_time": bayes_pred_train_time,
            "y_test": y_test,
            "y_test_preds": bayes_pred_test,
            "y_test_preds_time": bayes_pred_test_time,
        },
    }


# %%
serialized_predictions_path = os.path.join(serialization_folder, "predictions.ser")
predictions = ser.deserialize_or_create_object(
    type_=Dict[str, Dict[str, np.ndarray]],
    builder=get_predictions,
    filepath=serialized_predictions_path
)

# %%
models = ["rf", "svc_linear", "svc_poly", "svc_rbf", "knn", "beta_bayes"]
performance_df = {
    "model": models + models,
    "f_measure": [
        mtr.f1_score(predictions["rf"]["y_train_preds"],
                     predictions["rf"]["y_train"], average="weighted"),
        mtr.f1_score(predictions["svc_linear"]["y_train_preds"],
                     predictions["svc_linear"]["y_train"], average="weighted"),
        mtr.f1_score(predictions["svc_poly"]["y_train_preds"],
                     predictions["svc_poly"]["y_train"], average="weighted"),
        mtr.f1_score(predictions["svc_rbf"]["y_train_preds"],
                     predictions["svc_rbf"]["y_train"], average="weighted"),
        mtr.f1_score(predictions["knn"]["y_train_preds"],
                     predictions["knn"]["y_train"], average="weighted"),
        mtr.f1_score(predictions["beta_bayes"]["y_train_preds"],
                     predictions["beta_bayes"]["y_train"], average="weighted"),

        mtr.f1_score(predictions["rf"]["y_test_preds"],
                     predictions["rf"]["y_test"], average="weighted"),
        mtr.f1_score(predictions["svc_linear"]["y_test_preds"],
                     predictions["svc_linear"]["y_test"], average="weighted"),
        mtr.f1_score(predictions["svc_poly"]["y_test_preds"],
                     predictions["svc_poly"]["y_test"], average="weighted"),
        mtr.f1_score(predictions["svc_rbf"]["y_test_preds"],
                     predictions["svc_rbf"]["y_test"], average="weighted"),
        mtr.f1_score(predictions["knn"]["y_test_preds"],
                     predictions["knn"]["y_test"], average="weighted"),
        mtr.f1_score(predictions["beta_bayes"]["y_test_preds"],
                     predictions["beta_bayes"]["y_test"], average="weighted"),
    ],
    "accuracy": [
        mtr.accuracy_score(predictions["rf"]["y_train_preds"],
                           predictions["rf"]["y_train"]),
        mtr.accuracy_score(predictions["svc_linear"]["y_train_preds"],
                           predictions["svc_linear"]["y_train"]),
        mtr.accuracy_score(predictions["svc_poly"]["y_train_preds"],
                           predictions["svc_poly"]["y_train"]),
        mtr.accuracy_score(predictions["svc_rbf"]["y_train_preds"],
                           predictions["svc_rbf"]["y_train"]),
        mtr.accuracy_score(predictions["knn"]["y_train_preds"],
                           predictions["knn"]["y_train"]),
        mtr.accuracy_score(predictions["beta_bayes"]["y_train_preds"],
                           predictions["beta_bayes"]["y_train"]),

        mtr.accuracy_score(predictions["rf"]["y_test_preds"],
                           predictions["rf"]["y_test"]),
        mtr.accuracy_score(predictions["svc_linear"]["y_test_preds"],
                           predictions["svc_linear"]["y_test"]),
        mtr.accuracy_score(predictions["svc_poly"]["y_test_preds"],
                           predictions["svc_poly"]["y_test"]),
        mtr.accuracy_score(predictions["svc_rbf"]["y_test_preds"],
                           predictions["svc_rbf"]["y_test"]),
        mtr.accuracy_score(predictions["knn"]["y_test_preds"],
                           predictions["knn"]["y_test"]),
        mtr.accuracy_score(predictions["beta_bayes"]["y_test_preds"],
                           predictions["beta_bayes"]["y_test"]),
    ],
    "predictions_time": [
        predictions["rf"]["y_train_preds_time"],
        predictions["svc_linear"]["y_train_preds_time"],
        predictions["svc_poly"]["y_train_preds_time"],
        predictions["svc_rbf"]["y_train_preds_time"],
        predictions["knn"]["y_train_preds_time"],
        predictions["beta_bayes"]["y_train_preds_time"],

        predictions["rf"]["y_test_preds_time"],
        predictions["svc_linear"]["y_test_preds_time"],
        predictions["svc_poly"]["y_test_preds_time"],
        predictions["svc_rbf"]["y_test_preds_time"],
        predictions["knn"]["y_test_preds_time"],
        predictions["beta_bayes"]["y_test_preds_time"],
    ],
    "is_test": [
        False,
        False,
        False,
        False,
        False,
        False,

        True,
        True,
        True,
        True,
        True,
        True,
    ]
}
scores_df = pd.DataFrame(data=performance_df)
scores_df.to_csv(os.path.join(serialization_folder, "scores.csv"), index=False)
scores_df

# %% [markdown]
# ### Performance Lineplots
#
# Training scores are plotted as well to verify that there isn't any unusual behavior in the classifiers, i.e. overfitting, which can most commonly be noticed if training accuracy is much higher than test accuracy.

# %%
pretty_model_names = {
    "rf": "Random Forest",
    "svc_linear": "Linear Kernel SVC",
    "svc_poly": "Polynomial (degree=2) Kernel SVC",
    "svc_rbf": "RBF Kernel SVC",
    "knn": "K-Nearest Neighbors",
    "beta_bayes": "Beta Distribution Naive Bayes"
}


# %% [markdown]
# #### Accuracy

# %%
def performance_plot(perf_df: pd.DataFrame, y_col: str, test_only: bool = True) -> plt.Figure:
    perf_df = perf_df.copy()

    perf_df["model"] = perf_df["model"].apply(lambda m: pretty_model_names[m])
    perf_df = perf_df[perf_df["is_test"]] if test_only else perf_df
    perf_df = perf_df.sort_values(by=["accuracy"])

    fig, ax = plt.subplots(figsize=(8, 8))
    plot = sns.lineplot(
        ax=ax,
        data=perf_df, x="model", y=y_col,
        hue="is_test" if not test_only else None,
        palette="pastel" if not test_only else None,
        style="is_test" if not test_only else None,
        marker="o"
    )
    plot.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")

    return fig

def performance_plot_accuracy(perf_df: pd.DataFrame):
    return performance_plot(perf_df, y_col="accuracy")

def performance_plot_f_measure(perf_df: pd.DataFrame):
    return performance_plot(perf_df, y_col="f_measure")


# %%
accuracy_plot = performance_plot_accuracy(scores_df)
accuracy_plot.savefig(os.path.join(images_folder, "test_accuracy.png"), dpi=250, format="png")

# %% [markdown]
# #### F Measure

# %%
f_measure_plot = performance_plot_f_measure(scores_df)

# %% [markdown]
# #### Training Time

# %%
training_times_data = {
    "model": [
        pretty_model_names["rf"],
        pretty_model_names["svc_linear"],
        pretty_model_names["svc_poly"],
        pretty_model_names["svc_rbf"],
        pretty_model_names["knn"],
        pretty_model_names["beta_bayes"],
    ],
    "training_time": [
        rf_time,
        svc_linear_time,
        svc_poly_time,
        svc_rbf_time,
        knn_time,
        bayes_time
    ],
    "prediction_time": [
        predictions["rf"]["y_test_preds_time"],
        predictions["svc_linear"]["y_test_preds_time"],
        predictions["svc_poly"]["y_tesst_preds_time"],
        predictions["svc_rbf"]["y_test_preds_time"],
        predictions["knn"]["y_test_preds_time"],
        predictions["beta_bayes"]["y_test_preds_time"],
    ]
}
training_times_df = pd.DataFrame(data=training_times_data)

def performance_plot_training_times(training_times_df: pd.DataFrame,
                                    y_time_col: str,
                                    normalized: bool = True) -> plt.Figure:
    sorted_tr_times_df = training_times_df.sort_values(by=[y_time_col])

    if normalized:
        max_training_time = sorted_tr_times_df[y_time_col].max()
        sorted_tr_times_df[y_time_col] = sorted_tr_times_df[y_time_col] / max_training_time

    fig, ax = plt.subplots(figsize=(6, 6))
    plot = sns.barplot(
        ax=ax,
        data=sorted_tr_times_df, x="model", y=y_time_col,
        hue="model",
        palette="pastel",
    )
    ax.set_ylabel(f"Time {'' if normalized else '[s]'}")
    ax.set_xlabel("")
    plot.tick_params(axis="x", rotation=90)

    return fig


# %%
training_time_plot = performance_plot_training_times(training_times_df,
                                                     y_time_col="training_time",
                                                     normalized=True)
training_time_plot.suptitle("Training Times")
training_time_plot.tight_layout()
training_time_plot.savefig(os.path.join(images_folder, "training_times.png"), dpi=250, format="png")

# %%
temp = performance_plot_training_times(training_times_df,
                                y_time_col="training_time",
                                normalized=False)

# %% [markdown]
# #### Prediction Time

# %%
pred_time_plot = performance_plot_training_times(training_times_df,
                                                 y_time_col="prediction_time",
                                                 normalized=True)
pred_time_plot.suptitle("Prediction Times")
pred_time_plot.tight_layout()
pred_time_plot.savefig(os.path.join(images_folder, "prediction_times.png"), dpi=250, format="png")

# %% [markdown]
# ### Confusion Matrices

# %%
for i, model in enumerate(list(predictions.keys())):
    plot = mtr.ConfusionMatrixDisplay.from_predictions(
        y_true=predictions[model]["y_test"],
        y_pred=predictions[model]["y_test_preds"],
        labels=[lbl for lbl in range(0, 9+1)],
        normalize="true",
        cmap="PuRd",
        colorbar=False
    )

    fig: plt.Figure = plot.figure_
    ax: plt.Axes = plot.ax_

    ax.set_title(label=pretty_model_names[model])
    ax.grid(False)
    fig.set_size_inches(w=8, h=8)
    fig.savefig(os.path.join(images_folder, f"{model}_conf-matrix.png"), dpi=250, format="png")


# %% [markdown]
# ### Beta Distribution Averages

# %%
def get_mean_image_beta_bayes(beta_bayes: cls.BetaNaiveBayes, n_features: int, for_lbl: int) -> np.ndarray:
    mean_img = np.array([])
    for feat_idx in range(n_features):
        distrib = beta_bayes.distributions_[cls.ClassFeatureIdx(for_lbl, feat_idx)]
        mean_val = distrib.mean()
        mean_img = np.append(mean_img, mean_val if not np.isnan(mean_val) else 0)

    n = int(np.floor(np.sqrt(n_features)))
    reshaped_mean_img = mean_img.reshape(n, n)
    return reshaped_mean_img

# Fetch data again to have full 28x28 images
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X: pd.DataFrame = X / 255.0
y: pd.Series = y.astype(int)
bayes_for_plot = cls.BetaNaiveBayes().fit(X.to_numpy(), y.to_numpy())

images = np.array([get_mean_image_beta_bayes(bayes_for_plot, n_features=X.shape[1], for_lbl=lbl)
          for lbl in unique_labels])
plot = px.imshow(
    img=images,
    color_continuous_scale="gray",
    binary_string=True,
    width=400,
    height=400,
    facet_col=0,
    facet_col_wrap=5
)
plot.for_each_annotation(lambda a: a.update(text=""))
plot.update_yaxes(showticklabels=False)
plot.update_xaxes(showticklabels=False)
plot.write_image(os.path.join(images_folder, f"beta_means.png"), format="png", width=1200, height=1200)
plot.show()

# %%
