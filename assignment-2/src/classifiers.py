from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Protocol, Any, Mapping, List, NamedTuple

import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from loguru import logger


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> BaseClassifier:
        pass

    @abstractmethod
    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        pass


class KNN(BaseClassifier):
    X_: np.ndarray = None
    """
    Training samples
    """

    y_: np.ndarray = None
    """
    Training labels
    """

    labels_: np.ndarray = None
    """
    Unique labels
    """

    k: int
    """
    Number of nearest neighbors to consider
    """

    def __init__(self, k: int = 1):
        self.k: int = k

    def fit(self, X, y) -> KNN:
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        # Return the classifier
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        return (
            self.X_ is not None
            and self.y_ is not None
            and self.labels_ is not None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation: array should be 2D with shape (n_samples, n_features)
        check_array(X)

        # Order each neighbor
        distances_asc_idx = np.argsort(
            euclidean_distances(self.X_, X),  # Training set as rows, target set as columns
            axis=0  # This causes each column to be ordered from smallest to greatest
        )
        predictions = []
        for i in range(X.shape[0]):
            # Distance indexes of the i-th row, which represents the
            #   i-th entry of the set of samples to predict
            ith_distances = distances_asc_idx[:, i]
            closest_k_idx = ith_distances[:self.k]
            closest_k_labels = self.y_[closest_k_idx]

            # st.mode with the below parameters returns a named tuple with fields ("mode", "count"),
            #   each of which has a single value (because keepdims = False)
            prediction: Tuple[np.ndarray, np.ndarray] = st.mode(a=closest_k_labels, axis=None, keepdims=False)
            predictions.append(prediction.mode)

        return np.array(predictions)


class ProbabilityDistribution(Protocol):
    def mean(self) -> float:
        pass

    def var(self) -> float:
        pass

    def pdf(self, data: Any, *args, **kwargs) -> np.ndarray:
        pass

    def cdf(self, data: Any, *args, **kwargs) -> np.ndarray:
        pass


class SingleValueDistribution(ProbabilityDistribution):
    k: Any
    """
    The only value considered in the distribution, with probability 1
    """

    def __init__(self, k: Any):
        self.k = k

    def mean(self) -> float:
        return self.k

    def var(self) -> float:
        return 0

    def pdf(self, data: Any, *args, **kwargs) -> np.ndarray:  # TODO possibly add support for data arrays
        return np.array([1]) if data == self.k else np.array([0])

    def cdf(self, data: Any, *args, **kwargs) -> np.ndarray:  # TODO possibly add support for data arrays
        return np.array([1]) if data == self.k else np.array([0])


class ClassFeatureIdx(NamedTuple):
    label: int | float
    """
    Label associated to some class to predict
    """

    feature_idx: int
    """
    Index of the feature of the sample to classify
    """


class NaiveBayes(BaseClassifier, ABC):
    distributions_: Mapping[ClassFeatureIdx, ProbabilityDistribution | None] = None
    """
    Estimated distributions, identified by the tuple (label, feature_index)

    For example, to retrieve the estimated probability distribution
    of the 23rd feature with respect to class 2:
    
    ```
    naive_bayes_instance.distributions_[(2, 23)]
    ```
    
    Note that 23 is an index, while 2 is the actual value/label
    associated to the class.
    """

    labels_: np.ndarray = None
    """
    Array of labels (classes) seen during training.
    """

    labels_frequencies_: Mapping[int | float, float] = None
    """
    Mapping of labels (classes) with their respective frequencies, 
    as seen in the training process.
    """

    single_value_features_mask_: Mapping[int | float, np.ndarray[bool]] = None
    """
    Mapping that pairs each class with a mask that identifies which features, represented
    by their index in the dataset used to fit the model, consists of just a single value.
    It is important to identify such occurrences, because the probability distributions
    of such cases assigns probability 1 to some value K, and 0 to all the others, i.e.
    it can be seen as a discrete uniform distribution in the range [K, K]
    """

    @abstractmethod
    def fit(self, X, y) -> NaiveBayes:
        pass

    def __sklearn_is_fitted__(self) -> bool:
        # Method used by sklearn validation utilities
        return (
                self.distributions_ is not None
                and self.labels_ is not None
                and self.labels_frequencies_ is not None
                and self.single_value_features_mask_ is not None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if fit has been called
        check_is_fitted(self)

        ClassProbability = namedtuple("ClassProbability", ["label", "probability"])
        predictions: List[float] = []
        for sample_idx in range(X.shape[0]):
            logger.debug(f"Predicting sample #{sample_idx}")

            # Class associated to the highest conditional probability
            #   obtained for the current sample, i.e. p(x=sample | y=class)
            best_class_prob: ClassProbability = ClassProbability(-1, -1.0)
            for lbl in self.labels_:
                # Conditional probability of each feature given class `lbl`
                class_feature_cond_probabilities = []

                for feat_idx in range(X.shape[1]):
                    ith_feat = X[sample_idx, feat_idx]

                    # This is the conditional probability p(x_i=ith_feat | y=lbl)
                    # Actual probability of ith-feat is obtained by looking at the cdf,
                    #   because the distribution is continuous
                    ith_feat_distr = self.distributions_[ClassFeatureIdx(lbl, feat_idx)]

                    # Check that feature is not single-valued
                    if not self.single_value_features_mask_[lbl][feat_idx]:
                        epsilon = 0.05
                        ith_cond_prob = (
                                ith_feat_distr.cdf(ith_feat + epsilon) - ith_feat_distr.cdf(ith_feat - epsilon)
                        )
                    else:
                        # In case it is single valued, we artificially assign its cond. prob.
                        #   to 1, so that it becomes and irrelevant term in the product below
                        ith_cond_prob = 1

                    class_feature_cond_probabilities.append(ith_cond_prob)

                # This is for sure a float since the list is 1D
                full_sample_cond_probability: float = (
                        np.prod(class_feature_cond_probabilities) * self.labels_frequencies_[lbl]
                )

                # If the probability of the sample for the current class is the highest,
                #   then we have found a new best prediction
                if full_sample_cond_probability > best_class_prob.probability:
                    best_class_prob = ClassProbability(lbl, full_sample_cond_probability)

            predictions.append(best_class_prob.label)

        return np.array(predictions)


class BetaNaiveBayes(NaiveBayes):
    def __init__(self):
        super().__init__()

    def fit(self, X, y) -> BetaNaiveBayes:
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        # temp var because the field exposes only getters
        distributions = {}
        single_value_features_mask = {
            lbl: np.full(X.shape[1], False)  # Init mask as all false
            for lbl in self.labels_
        }
        for lbl in self.labels_:
            class_samples: np.ndarray = X[y == lbl]
            class_means = class_samples.mean(axis=0)
            class_variances = class_samples.var(axis=0)

            for feat_idx in range(X.shape[1]):
                # This is the parameter estimation using the moments approach
                ith_mean = class_means[feat_idx]  # E[X_i]
                ith_var = class_variances[feat_idx]  # Var[X_i]

                k = ((ith_mean * (1 - ith_mean)) / ith_var) - 1  # K_i = (E[X_i] * (1 - E[X_i]) / Var[X_i]) - 1
                alpha: float = k * ith_mean  # K_i * E[X_i]
                beta: float = k * (1 - ith_var)  # K_i * (1 - E[X_i])

                # Check that distribution involves more than one value
                if ith_var > 0:
                    distributions[ClassFeatureIdx(lbl, feat_idx)] = st.beta(alpha, beta)
                else:
                    distributions[ClassFeatureIdx(lbl, feat_idx)] = SingleValueDistribution(k=ith_mean)
                    single_value_features_mask[lbl][feat_idx] = True

        self.distributions_ = distributions
        self.single_value_features_mask_ = single_value_features_mask

        self.labels_frequencies_ = {
            lbl: (y[y == lbl].size / y.size)
            for lbl in self.labels_
        }

        # Return the classifier
        return self
