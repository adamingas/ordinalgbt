from functools import wraps

import numpy as np


def dec_clip_y_pred(fun):
    @wraps(fun)
    def wrapped(*, y_true, y_preds, theta):
        y_preds = np.clip(y_preds, max(theta)-36, a_max=700 + min(theta))
        return fun(y_true=y_true, y_preds=y_preds, theta=theta)

    return wrapped


def stack_zeros_ones(a: np.ndarray, only_zeros=False) -> np.ndarray:
    """Stacks zeroes and ones on the left and rights part of the array
    Stacks horizontally zeros and ones on the left and right hand side of the array
    respectively. If only_zeros is true then it only stacks zeros. This is for the
    gradient which is zero at both ends of the sigmoid.
    e.g.::

        a = [[1,2],
            [3,4],
            [5,6]]
        returns
            [[0,1,2,1],
            [0,3,4,1],
            [0,5,6,1]]


    Parameters
    ----------
    a: np.ndarray
        A 2D array to pad with zeroes and ones
    only_zeros: bool, default False
        If true, then it pads with only zeroes

    Returns
    -------
    np.ndarray

    """
    if only_zeros:
        return np.hstack(
            (
                np.zeros(a.shape[0])[:, np.newaxis],
                a,
                np.zeros(a.shape[0])[:, np.newaxis],
            )
        )

    return np.hstack(
        (np.zeros(a.shape[0])[:, np.newaxis], a, np.ones(a.shape[0])[:, np.newaxis])
    )


def sigmoid(z)->np.ndarray:
    """Sigmoid
    Sigmoid implementation in numpy

    Parameters
    ----------
    z : np.ndarray

    Returns
    -------
    np.ndarray
        Sigmoid
    """
    return 1 / (1 + np.exp(-z))


def grad_sigmoid(z) -> np.ndarray:
    """Gradient
    Gradient of the sigmoid

    Parameters
    ----------
    z : np.ndarray

    Returns
    -------
    np.ndarray
        Gradient of Sigmoid

    """
    phat = sigmoid(z)
    return phat * (1 - phat)


def hess_sigmoid(z) -> np.ndarray:
    """Hessian
    Hessian of the sigmoid
    Sigmoid implementation in numpy

    Parameters
    ----------
    z : np.ndarray

    Returns
    -------
    np.ndarray
        Hessian of sigmoid

    """
    grad = grad_sigmoid(z)
    sig = sigmoid(z)
    return grad * (1 - sig) - sig * (grad)


# class y2ord():# Convert ordinal to 1, 2, ... K
#     def __init__(self):
#         self.di = {}
#     def fit(self,y):
#         self.uy = np.sort(np.unique(y))
#         self.di = dict(zip(self.uy, np.arange(len(self.uy))+1))
#     def transform(self,y):
#         return(np.array([self.di[z] for z in y]))


def alpha2theta(alpha):  # theta[t] = theta[t-1] + exp(alpha[t])
    return np.cumsum(np.append(alpha[0], np.exp(alpha[1:])))


def theta2alpha(theta):  # alpha[t] = log(theta[t] - theta[t-1])
    return np.append(theta[0], np.log(theta[1:] - theta[:-1]))


# def alpha_beta_wrapper(alpha_beta, X, lb=20, ub=20):
#     K = len(alpha_beta) + 1
#     if X is not None:
#         K -= X.shape[1]
#         beta = alpha_beta[K - 1:]
#     else:
#         beta = np.array([0])
#     alpha = alpha_beta[:K - 1]
#     theta = alpha2theta(alpha, K)
#     theta = np.append(np.append(theta[0] - lb, theta), theta[-1] + ub)
#     return(alpha, theta, beta, K)


def probas_from_y_pred(y_preds, theta):
    """
    convers y_preds to probabilities
    """
    s_array: np.ndarray = sigmoid(theta - y_preds[:, np.newaxis])
    # Adding boundary terms of 1 and 0 to make sure that we have probabilities for
    # all classes :TODO: Explain in detail
    # Cumulative probabilities, for column k and row i this matrix represents
    # P(y_i<=k)
    c_probas = stack_zeros_ones(s_array)

    probas = c_probas[:, 1 : len(theta) + 2] - c_probas[:, 0 : len(theta) + 1]
    # probas = np.clip(
    #     probas, a_min=np.finfo(float).eps, a_max=1 - len(theta) * np.finfo(float).eps
    # )
    return probas

@dec_clip_y_pred
def ordinal_logistic_nll(y_true: np.ndarray, y_preds: np.ndarray, theta: np.ndarray):
    """Ordinal Negative log lilelihood

    Parameters
    ----------
    y_true : np.ndarray
        1-D array with correct labels, starts from 0 and goes up to the number
        of unique classes minus one (so unique values are 0,1,2 when dealing
        with three classes)
    y_preds : np.ndarray
        1-D array with predictions in latent space
    theta : np.ndarray
        thresholds, 1-D array, size is the number of classes minus one.

    Returns
    -------
    np.ndarray
        logistic ordinal negative log likelihood

    """
    probas = probas_from_y_pred(y_preds, theta)
    # probabilities associated with the correct label
    label_probas = probas[np.arange(0, len(y_true)), y_true]
    label_probas = np.clip(
        label_probas,
        a_min=np.finfo(float).eps,
        a_max=1 - len(theta) * np.finfo(float).eps
    )
    # loss
    return -np.sum(np.log(label_probas))


# Gradient
def gradient_ordinal_logistic_nll(
    y_true: np.ndarray, y_preds: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Gradient of ordinal nll
    Gradient of the ordinal logistic regression with respect to the predictions

    Parameters
    ----------
    y_true : np.ndarray
        1-D array with correct labels, starts from 0 and goes up to the number
        of unique classes minus one (so unique values are 0,1,2 when dealing
        with three classes)
    y_preds : np.ndarray
        1-D array with predictions in latent space
    theta : np.ndarray
        thresholds, 1-D array, size is the number of classes minus one.

    Returns
    -------
    np.ndarray
        Gradient of logistic ordinal negative log likelihood

    """
    y_preds = np.clip(y_preds, -20, a_max=700 + min(theta))
    probas = probas_from_y_pred(y_preds, theta)
    y_true = y_true.astype(int)
    gs_array: np.ndarray = grad_sigmoid(theta - y_preds[:, np.newaxis])
    gc_probas = stack_zeros_ones(gs_array, only_zeros=True)
    g_probas = -(gc_probas[:, 1 : len(theta) + 2] - gc_probas[:, 0 : len(theta) + 1])
    gradient = -(g_probas / probas)[np.arange(0, len(y_true)), y_true]
    return gradient


def hessian_ordinal_logistic_nll(
    y_true: np.ndarray, y_preds: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Hessian of ordinal nll
    Hessian of the ordinal logistic regression with respect to the predictions

    Parameters
    ----------
    y_true : np.ndarray
        1-D array with correct labels, starts from 0 and goes up to the number
        of unique classes minus one (so unique values are 0,1,2 when dealing
        with three classes)
    y_preds : np.ndarray
        1-D array with predictions in latent space
    theta : np.ndarray
        thresholds, 1-D array, size is the number of classes minus one.

    Returns
    -------
    np.ndarray
        Hessian of logistic ordinal negative log likelihood

    """
    y_preds = np.clip(y_preds, -20, a_max=700 + min(theta))
    probas = probas_from_y_pred(y_preds, theta)

    y_true = y_true.astype(int)
    gs_array: np.ndarray = grad_sigmoid(theta - y_preds[:, np.newaxis])
    gc_probas = stack_zeros_ones(gs_array, only_zeros=True)
    hs_array: np.ndarray = hess_sigmoid(theta - y_preds[:, np.newaxis])
    hc_probas = stack_zeros_ones(hs_array, only_zeros=True)
    g_probas = -(gc_probas[:, 1 : len(theta) + 2] - gc_probas[:, 0 : len(theta) + 1])
    h_probas = hc_probas[:, 1 : len(theta) + 2] - hc_probas[:, 0 : len(theta) + 1]

    hessian = -(h_probas / probas - np.power(g_probas / probas, 2))[
        np.arange(0, len(y_true)), y_true
    ]
    # hessian[np.abs(hessian) <=np.finfo(float).eps] = -np.finfo(float).eps
    return hessian


def lgb_ordinal_loss(
    y_true: np.ndarray, y_pred: np.ndarray, theta: np.ndarray
):
    """Ordinal loss for lightgbm use
    The ordinal loss used in the lightgbm framework. Returns the
    gradient and hessian of the loss.

    Parameters
    ----------
    y_true : np.ndarray
        1-D array with correct labels, starts from 0 and goes up to the number
        of unique classes minus one (so unique values are 0,1,2 when dealing
        with three classes)
    y_preds : np.ndarray
        1-D array with predictions in latent space
    theta : np.ndarray
        thresholds, 1-D array, size is the number of classes minus one.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Gradient and Hessian of logistic ordinal negative log likelihood
    """
    grad = gradient_ordinal_logistic_nll(y_true, y_pred, theta)
    hess = hessian_ordinal_logistic_nll(y_true, y_pred, theta)
    return (grad, hess)


# def lgb_loss_factory():
#     theta = np.array([0,4])
#     alpha = theta2alpha(theta)
#     return partial(lgb_ordinal_loss, alpha = alpha)

# Likelihood function
# def nll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
#     alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
#     score = np.dot(X,beta)
#     ll = 0
#     for kk, idx in enumerate(idx_y):
#         ll += sum(np.log(sigmoid(theta[kk+1]-score[idx]
# )-sigmoid(theta[kk]-score[idx])))
#     nll = -1*(ll / X.shape[0])
#     return(nll)

# # Gradient wrapper
# def gll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
#     grad_alpha = gll_alpha(alpha_beta, X, idx_y)
#     grad_X = gll_beta(alpha_beta, X, idx_y)
#     return(np.append(grad_alpha,grad_X))

# # gradient function for beta
# def gll_beta(alpha_beta, X, idx_y, lb=20, ub=20):
#     alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
#     score = np.dot(X, beta)
#     grad_X = np.zeros(X.shape[1])
#     for kk, idx in enumerate(idx_y):  # kk = 0; idx=idx_y[kk]
#         den = sigmoid(theta[kk + 1] - score[idx]) - sigmoid(theta[kk] - score[idx])
#         num = -grad_sigmoid(theta[kk + 1] - score[idx]) \
#             + grad_sigmoid(theta[kk] - score[idx])
#         grad_X += np.dot(X[idx].T, num / den)
#     grad_X = -1 * grad_X / X.shape[0]  # negative average of gradient
#     return(grad_X)

# # gradient function for theta=exp(alpha)
# def gll_alpha(alpha_beta, X, idx_y, lb=20, ub=20):
#     alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
#     score = np.dot(X, beta)
#     grad_alpha = np.zeros(K - 1)
#     for kk in range(K-1):
#         idx_p, idx_n = idx_y[kk], idx_y[kk+1]
#         den_p = sigmoid(theta[kk + 1] - score[idx_p]) \
#             - sigmoid(theta[kk] - score[idx_p])
#         den_n = sigmoid(theta[kk + 2] - score[idx_n]) \
#             - sigmoid(theta[kk+1] - score[idx_n])
#         num_p = grad_sigmoid(theta[kk + 1] - score[idx_p])
#         num_n = grad_sigmoid(theta[kk + 1] - score[idx_n])
#         grad_alpha[kk] += sum(num_p/den_p) - sum(num_n/den_n)
#     grad_alpha = -1* grad_alpha / X.shape[0]  # negative average of gradient
#     grad_alpha *= np.append(1, np.exp(alpha[1:])) # apply chain rule
#     return(grad_alpha)

# # inference probabilities
# def prob_ordinal(alpha_beta, X, lb=20, ub=20):
#     alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
#     score = np.dot(X, beta)
#     phat = (np.atleast_2d(theta) - np.atleast_2d(score).T)
#     phat = sigmoid(phat[:, 1:]) - sigmoid(phat[:, :-1])
#     return(phat)

# Wrapper for training/prediction
# class ordinal_reg():
#     def __init__(self,standardize=True):
#         self.standardize = standardize
#     def fit(self,data,lbls):
#         self.p = data.shape[1]
#         self.Xenc = StandardScaler().fit(data)
#         self.yenc = y2ord()
#         self.yenc.fit(y=lbls)
#         ytil = self.yenc.transform(lbls)
#         idx_y = [np.where(ytil == yy)[0] for yy in list(self.yenc.di.values())]
#         self.K = len(idx_y)
#         theta_init = np.array([(z + 1) / self.K for z in range(self.K - 1)])
#         theta_init = np.log(theta_init / (1 - theta_init))
#         alpha_init = theta2alpha(theta_init, self.K)
#         param_init = np.append(alpha_init, np.repeat(0, self.p))
#         self.alpha_beta = minimize(
#    fun=nll_ordinal, x0=param_init, method='L-BFGS-B', jac=gll_ordinal,
#                                    args=(self.Xenc.transform(data), idx_y)).x
#     def predict(self,data):
#         phat = prob_ordinal(self.alpha_beta,self.Xenc.transform(data))
#         return(np.argmax(phat,axis=1)+1)
#     def predict_proba(self,data):
#         phat = prob_ordinal(self.alpha_beta,self.Xenc.transform(data))
#         return(phat)
