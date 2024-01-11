"""Ordinal classifier lightgbm implementation """
import numpy as np
from lightgbm import LGBMRegressor
from scipy.optimize import minimize

from ordinalgbt.loss import (
    alpha2theta,
    lgb_ordinal_loss,
    ordinal_logistic_nll,
    probas_from_y_pred,
    theta2alpha,
)


class LGBMOrdinal(LGBMRegressor):
    def __init__(
        self,
        #  threshold_interval: float=2,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective="immediate-thresholds",
        class_weight=None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state=None,
        n_jobs: int = -1,
        silent="warn",
        importance_type: str = "split",
        **kwargs,
    ):
        super().__init__(
            objective=None,
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs,
        )
        # self.threshold_interval = threshold_interval

    def _initialise_theta(self):
        return np.linspace(0, (self.n_classes - 2) * 1, self.n_classes - 1)

    def _lgb_loss_factory(self):
        self.theta = self._initialise_theta()

        def loss(y_test, y_pred):
            return lgb_ordinal_loss(y_test, y_pred, self.theta)

        return loss

    @staticmethod
    def _alpha_loss_factory(y_true, y_preds):
        """
        Creates loss parametrised by alpha
        """

        def loss(alpha):
            theta = alpha2theta(alpha)
            return ordinal_logistic_nll(y_true=y_true, y_preds=y_preds, theta=theta)

        return loss

    def _optimise_alpha(self, y_true, y_preds):
        """
        Takes loss parametrised by alpha and optimises it.
        Can optionally take in gradient.
        """
        loss = self._alpha_loss_factory(y_true, y_preds)
        alpha = theta2alpha(self.theta)
        bounds = [(None,3.58)]*len(alpha)
        self._alpha_optimisation_report = minimize(loss, alpha, bounds=bounds)
        alpha = self._alpha_optimisation_report.x
        self.theta = alpha2theta(alpha)

    def _initialise_objective(self, y):
        """
        initialises the objective by creating the loss and setting the class
        attributes
        """
        self.n_classes = len(np.unique(y))
        self.objective = self._lgb_loss_factory()
        self._objective = self.objective

    def _output_to_probability(self, output):
        return probas_from_y_pred(output, self.theta)

    def _hot_start(self, X, y, hot_start_iterations=5, **kwargs):
        """
        TODO
        """
        fit_n_estimators = self.n_estimators
        self.n_estimators = hot_start_iterations

        # Fits the model for the default initialisation of alphas
        self._fit(X, y, **kwargs)

        # Updates the alpha to those that minimise the loss
        self._optimise_alpha(y, self.predict_proba(X, raw_score=True))
        self._Booster = None
        self.n_estimators = fit_n_estimators

    def fit(
        self,
        X,
        y,
        hot_start_iterations=5,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose="warn",
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ) -> "LGBMOrdinal":
        """Docstring is inherited from the LGBMModel."""
        self._initialise_objective(y)
        self._hot_start(X, y, hot_start_iterations=hot_start_iterations)
        self._fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            verbose=verbose,
        )
        return self

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose="warn",
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ) -> "LGBMOrdinal":
        """Docstring is inherited from the LGBMModel."""
        self = super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            verbose=verbose,
        )
        return self

    def predict(
        self,
        X,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        preds = self.predict_proba(
            X,
            raw_score=False,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        return np.argmax(preds, axis=1)

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs,
    ):
        preds = super().predict(
            X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        if not raw_score:
            return self._output_to_probability(preds)
        return preds
