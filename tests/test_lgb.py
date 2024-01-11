from inspect import signature

import numpy as np

from ordinalgbt.lgb import LGBMOrdinal
from ordinalgbt.loss import (
    gradient_ordinal_logistic_nll,
    hessian_ordinal_logistic_nll,
    probas_from_y_pred,
    theta2alpha,
)


def test_initialise_theta():
    model = LGBMOrdinal()
    model.n_classes = 5
    expected_theta = np.array([0., 1., 2., 3.])
    assert np.array_equal(model._initialise_theta(), expected_theta)

def test_lgb_loss_factory():
    model = LGBMOrdinal()
    model.n_classes = 5
    model.theta = np.array([0., 2., 4., 6.])
    loss = model._lgb_loss_factory()
    y_test = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 3, 5, 7])
    expected_grad = gradient_ordinal_logistic_nll(y_test, y_pred, model.theta)
    expected_hess = hessian_ordinal_logistic_nll(y_test, y_pred, model.theta)

    grad, hess =  loss(y_test, y_pred)
    assert np.isclose(grad, expected_grad).all()
    assert np.isclose(hess, expected_hess).all()

def test_alpha_loss_factory():
    model = LGBMOrdinal()
    model.n_classes = 5
    theta_1, theta_2 = np.array([0., 2., 4., 6.]), np.array([0., 1., 2., 3.])
    alpha_1,alpha_2 = theta2alpha(theta_1), theta2alpha(theta_2)
    y_true = np.array([0, 1, 2, 3, 4])
    y_preds = np.array([0, 1, 2, 3, 4])
    loss = model._alpha_loss_factory(y_true, y_preds)

    assert loss(alpha_1) > loss(alpha_2)

def test_optimise_alpha():
    model = LGBMOrdinal()
    model.n_classes = 5
    model.theta = np.array([0., 2., 4., 6.])
    y_true = np.array([0, 1, 2, 3, 4])
    y_preds = np.array([0, 1, 2, 3, 4])
    model._optimise_alpha(y_true, y_preds)
    # assert that alpha_optimisation_report is not None and theta has changed
    assert model._alpha_optimisation_report is not None
    assert not np.array_equal(model.theta, np.array([0., 2., 4., 6.]))

def test_initialise_objective():
    model = LGBMOrdinal()
    y = np.array([0, 1, 2, 3, 4])
    model._initialise_objective(y)
    # assert that n_classes and objective are not None
    assert model.n_classes == 5
    assert len(signature(model.objective).parameters) == 2

def test_output_to_probability():
    model = LGBMOrdinal()
    model.n_classes = 5
    model.theta = np.array([0., 2., 4., 6.])
    output = np.array([0, 1, 2, 3, 4])
    probas = model._output_to_probability(output)
    # assert that probas is not None and has the correct shape
    assert probas is not None
    assert probas.shape == (5,5)
    assert ( probas == probas_from_y_pred(output,model.theta) ).all()
