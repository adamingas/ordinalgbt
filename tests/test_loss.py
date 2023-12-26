import numpy as np
import pytest
from ordinal_gbt.loss import (
    alpha2theta,
    dec_clip_y_pred,
    grad_sigmoid,
    gradient_ordinal_logistic_nll,
    hess_sigmoid,
    hessian_ordinal_logistic_nll,
    lgb_ordinal_loss,
    ordinal_logistic_nll,
    probas_from_y_pred,
    sigmoid,
    stack_zeros_ones,
    theta2alpha,
)
from pandas import Series


def test_stack_zeros_ones():
    a = np.array([[1,2], [3,4], [5,6]])
    expected_result = np.array([[0,1,2,1], [0,3,4,1], [0,5,6,1]])
    assert np.array_equal(stack_zeros_ones(a), expected_result)

def test_sigmoid():
    z = np.array([0,1,2])
    expected_result = np.array([0.5, 0.73105858, 0.88079708])
    assert np.allclose(sigmoid(z), expected_result, rtol=1e-05)

def test_grad_sigmoid():
    z = np.array([0,1,2])
    expected_result = np.array([0.25, 0.19661193, 0.10499359])
    assert np.allclose(grad_sigmoid(z), expected_result, rtol=1e-05)

def test_hess_sigmoid():
    z = np.array([0,1,2])
    expected_result = np.array([0, -0.09085775, -0.0799625])
    assert np.allclose(hess_sigmoid(z), expected_result, rtol=1e-05)

def test_alpha2theta():
    alpha = np.array([0,0,0])
    expected_result = np.array([0, 1, 2])
    assert np.allclose(alpha2theta(alpha), expected_result, rtol=1e-05)

def test_theta2alpha():
    theta = np.array([0,1,2])
    expected_result = np.array([0, 0, 0])
    assert np.allclose(theta2alpha(theta), expected_result, rtol=1e-05)

def test_probas_from_y_pred():
    y_preds = np.array([0,1,2])
    theta = np.array([1,2])
    expected_result = np.array([[0.73105858, 0.1497385 , 0.11920292],
                                [0.5       , 0.23105858, 0.26894142],
                                [0.26894142, 0.23105858, 0.5       ]])
    assert np.allclose(probas_from_y_pred(y_preds, theta), expected_result, rtol=1e-05)

def test_ordinal_logistic_nll():
    y_preds = np.array([1, 4, 3])
    y_true = np.array([1, 2, 0])
    theta = np.array([0, 2])
    expected_loss = -np.sum(np.log(
        sigmoid(np.array([1,500,-3])) - sigmoid(np.array([-1,-2,-500]))
        ))
    loss = ordinal_logistic_nll(y_true, y_preds, theta)
    assert isinstance(loss, float)
    assert loss == pytest.approx(expected_loss)

def test_gradient_ordinal_logistic_nll():
    y_preds = np.array([1.5, 15, 12])
    y_true = np.array([1, 2, 0])
    theta = np.array([1,2])

    np.testing.assert_almost_equal(
        gradient_ordinal_logistic_nll(y_true, y_preds, theta),
                                   np.array([0, 0, 1]),
                                   decimal=3)

def test_hessian_ordinal_logistic_nll():
    y_preds = np.array([1.5, 15, -38])
    y_true = np.array([1, 2, 0])
    theta = np.array([1,2])


    np.testing.assert_almost_equal(hessian_ordinal_logistic_nll(y_true, y_preds, theta),
                                   np.array([0.47, 0, 0]),
                                   decimal=5)

def test_lgb_ordinal_loss():
    y_preds = np.array([1.5, 15, -38])
    y_true = np.array([1, 2, 0])
    theta = np.array([1,2])
    assert lgb_ordinal_loss(y_true, y_preds, theta) is not None
    assert lgb_ordinal_loss(Series(y_true), y_preds, theta) is not None

def test_dec_clip_y_pred():
    def test(y_true,y_preds,theta):
        return y_preds
    fun = dec_clip_y_pred(test)
    y_preds = np.array([1.5, 710, -38])
    y_true = np.array([1, 710, 0])
    theta = np.array([1,2])
    assert float(fun(y_true = y_true,y_preds = y_preds, theta = theta).max()) ==\
        pytest.approx(700 + 1)
