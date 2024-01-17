.. role:: raw-latex(raw)
   :format: latex
..

Loss
====

Maths
-----

Immediate Thresholds
~~~~~~~~~~~~~~~~~~~~

As is usual in ordinal regression model formulations, we build a
regressor that learns a latent variable :math:`y`, and then use a set of
thresholds :math:`\Theta` to produce the probability estimates for each
label. The set of thresholds is ordered, does not include infinities,
and has as many members as the numbers of labels minus one.

We want to come up with a a way to map the latent variable :math:`y` to
the probability space such that when :math:`y` is in
:math:`(\theta_{k-1},\theta_{k})` the probability of label :math:`k` is
maximised.

In a three ordered labeled problem, we only need two thresholds,
:math:`\theta_1` and :math:`\theta_2`, to define the three regions which
are associated to each label
:math:`(-\infty,\theta_1], (\theta_1, \theta_2], (\theta_2, \infty)`.

Deriving the probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

A property we want our mapping from latent variable to probability to
have is for the cummulative probability of label :math:`z` being at most
label :math:`k` to increase as the label increases. This means that
:math:`P(z\leq k;y,\Theta)` should increase as :math:`k` increases
(i.e. as we consider more labels).

Another property is that as the latent variable :math:`y` gets smaller,
the cummulative probability should also increase, and as it gets larger
it should decrease.

To satisfies this properties we use a function :math:`F` which grows as
its argument grows and shrinks as the arguments shrink. We can then
define the cumulative probability as:

.. math::


   P(z \leq k; y,\Theta  ) = F(\theta_k - y) ,

making sure that the range of :math:`F` is contrained to the
:math:`(0,1)` interval. This formulation satisfies all of our
properties. As we consider larger (higher in the order) labels
:math:`k`, the threshold :math:`\theta_k` grows and so does the
cumulative probability. As :math:`y` grows, the input to :math:`F`
shrinks and so does the cumulative probability.

Naturally, the probability of :math:`z` being any particular label is
then:

.. math::
   P(z = k; y,\Theta  ) &=P(z \leq k; y,\Theta) -P(z \leq k-1; y,\Theta  )  \hspace{2mm} \\
   &= F(\theta_k - y) - F(\theta_{k-1} - y)

A function that satisfies all these conditions is the sigmoid function,
hereafter denoted as :math:`\sigma`. ### Deriving the loss function

Given n samples, the likelihood of our set of predictions :math:`\bf y`
is:

.. math::


   L({\bf y} ;\Theta) = \prod_{i =0}^n I(z_i=k)P(z_i = k; y_i,\Theta)

As is usual in machine learning we use the negative log likelihhod as
our loss:

.. math::
   l({\bf y};\Theta) &= -\log L({\bf y},\theta)\\
   &= -\sum_{i=0}^n I(z_i=k)\log(P(z_i = k; y_i,\Theta)) \\
   &= -\sum_{i=0}^n I(z_i=k)\log \left(\sigma(\theta_k - y_i) - \sigma(\theta_{k-1} - y_i)\right)

Deriving the gradient and hessian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a custom loss function with gradient boosting tree frameworks
(i.e. lightgbm), we have to first derive the gradient and hessian of the
loss with respect to **the raw predictions**, in our case the latent
variable :math:`y_i`.

We denote the first and second order derivative of the sigmoid as
:math:`\sigma'` and :math:`\sigma''` respectively.

The gradient is denoted as:

.. math::
   \mathcal{G}&=\frac{\partial l({\bf y};\Theta)}{\partial {\bf y}} \\
   &= -\frac{\partial }{\partial {\bf y}} \sum_{i=0}^n I(z_i=k)\log \left(\sigma(\theta_k - y_i) - \sigma(\theta_{k-1} - y_i)\right)  \\
   &=
   \begin{pmatrix}
   -\frac{\partial }{\partial y_1} \sum_{i=0}^n I(z_i=k)\log \left(\sigma(\theta_k - y_i) - \sigma(\theta_{k-1} - y_i)\right)  \\
   ... \\
   -\frac{\partial }{\partial y_n} \sum_{i=0}^n I(z_i=k)\log \left(\sigma(\theta_k - y_i) - \sigma(\theta_{k-1} - y_i)\right)  \\
   \end{pmatrix} \\
   &=
   \begin{pmatrix}
   I(z_1 = k) \left( \frac{\sigma'(\theta_k-y_1) - \sigma'(\theta_{k-1}-y_1)}{\sigma(\theta_k-y_1) - \sigma(\theta_{k-1}-y_1)} \right)  \\ 
   ... \\
   I(z_n = k) \left( \frac{\sigma'(\theta_k-y_n) - \sigma'(\theta_{k-1}-y_n)}{\sigma(\theta_k-y_n) - \sigma(\theta_{k-1}-y_n)} \right)  \\ 
   \end{pmatrix}

The summmation is gone when calculating the derivative for variable
:math:`y_i` as every element of the summation depends only on one latent
variable: 

.. math::
   \frac{\partial f(y_1)+f(y_2)+f(y_3)}{\partial {\bf y}} &=
   \begin{pmatrix}
   \frac{\partial f(y_1)+f(y_2)+f(y_3)}{\partial y_1} \\
   \frac{\partial f(y_1)+f(y_2)+f(y_3)}{\partial y_2} \\
   \frac{\partial f(y_1)+f(y_2)+f(y_3)}{\partial y_3} \\
   \end{pmatrix} \\
   &=
   \begin{pmatrix}
   \frac{\partial f(y_1)}{\partial y_1} \\
   \frac{\partial f(y_2)}{\partial y_2} \\
   \frac{\partial f(y_3)}{\partial y_3} \\
   \end{pmatrix}

The hessian is the partial derivative of the gradient with respect to
the latent variable vector. This means that for each element of the
gradient vector we calculate the partial derivative w.r.t. the whole
latent variable vector. Thus the hessian is a matrix of partial
derivatives:

.. math::
   \begin{pmatrix}
   \frac{\partial}{\partial y_1 y_1} & ... &
   \frac{\partial}{\partial y_1 y_n} \\
   .&&.\\.&&.\\.&&.\\
   \frac{\partial}{\partial y_n y_1} & ... &
   \frac{\partial}{\partial y_n y_n}
   \end{pmatrix}l({\bf y};\Theta)

However, since we know that the partial derivative of the loss w.r.t.
the latent variable :math:`y_i` depends only on the :math:`i^{th}`
element of the :math:`y` vector, the off diagonal elements of the
hessian matrix are reduced to zero:

.. math::


   \frac{\partial}{\partial y_i y_j} l({\bf y};\Theta) = 0 \text{ if } i \neq j

The hessian is then reduced to a vetor:

.. math::


   \mathcal{H} &=  
       \begin{pmatrix}
           \frac{\partial}{\partial y_1 y_1}  \\
           ... \\
           \frac{\partial}{\partial y_n y_n}
       \end{pmatrix}l({\bf y};\Theta) \\
       &=
       \begin{pmatrix}
           \frac{\partial}{\partial y_1 }I(z_1 = k) \left( \frac{\sigma'(\theta_k-y_1) - \sigma'(\theta_{k-1}-y_1)}{\sigma(\theta_k-y_1) - \sigma(\theta_{k-1}-y_1)} \right)  \\ 
           ... \\
           \frac{\partial}{\partial y_n }
           I(z_n = k) \left( \frac{\sigma'(\theta_k-y_n) - \sigma'(\theta_{k-1}-y_n)}{\sigma(\theta_k-y_n) - \sigma(\theta_{k-1}-y_n)} \right)  
       \end{pmatrix}\\
       &=
       \begin{pmatrix}
           -I(z_i = k) \left( \frac{\sigma''(\theta_k-y_1) - \sigma''(\theta_{k-1}-y_1)}{\sigma(\theta_k-y_1) - \sigma(\theta_{k-1}-y_1)} \right)  +
             I(z_n = k)\left( \frac{\sigma'(\theta_k-y_1) - \sigma'(\theta_{k-1}-y_1)}{\sigma(\theta_k-y_1) - \sigma(\theta_{k-1}-y_1)} \right)^2 \\ 
           ... \\
           -I(z_n = k) \left( \frac{\sigma''(\theta_k-y_n) - \sigma''(\theta_{k-1}-y_n)}{\sigma(\theta_k-y_n) - \sigma(\theta_{k-1}-y_n)} \right)  +
             I(z_n = k)\left( \frac{\sigma'(\theta_k-y_n) - \sigma'(\theta_{k-1}-y_n)}{\sigma(\theta_k-y_n) - \sigma(\theta_{k-1}-y_n)} \right)^2 \\ 
       \end{pmatrix}

Miscellanious
~~~~~~~~~~~~~

The gradient of the sigmoid function is:

.. math::


   \sigma'(x) = \sigma(x)(1-\sigma(x))

and the hessian is:

.. math::


       \sigma''(x) &= \frac{d}{dx}\sigma(x)(1-\sigma(x)) \\
       &= \sigma'(x)(1-\sigma(x)) - \sigma'(x)\sigma(x)\\
       &= \sigma(x)(1-\sigma(x))(1-\sigma(x)) -\sigma(x)(1-\sigma(x))\sigma(x) \\ 
       &= (1-\sigma(x))\left(\sigma(x)-2\sigma(x)^2\right)

.. raw:: html

   <!-- 

   $$
   \begin{align*}
   \log L(\bbeta) &= l(\bbeta;\btheta) = \sum_{i=1}^n I(y_i=k) \log  \big[ \sigma(\theta_k - \eta_i) - \sigma(\theta_{k-1} - \eta_i) \big] \\
   \eta_i &= \bx_i^T \bbeta \\
   \frac{\partial l(\bbeta;\btheta)}{\partial \bbeta} &= \nabla_\bbeta = -\sum_{i=1}^n \bx_i I(y_i = k) \Bigg( \frac{\sigma'(\theta_k-\eta_i) + \sigma'(\theta_{k-1}-\eta_i)}{d_{ik}} \Bigg) \\
   d_{ik} &= \sigma(\theta_k-\eta_i) - \sigma(\theta_{k-1}-\eta_i) \\
   \frac{\partial l(\bbeta;\btheta)}{\partial \btheta} &= \nabla_\btheta = \sum_{i=1}^n \Bigg( I(y_i = k) \frac{\sigma'(\theta_k-\eta_i)}{d_{ik}} - I(y_i = k+1) \frac{\sigma'(\theta_k-\eta_i)}{d_{ik+1}} \Bigg)
   \end{align*}
   $$


   $$
   \begin{align*}
   \tilde y &= \arg\max_k [P(y=k|\bbeta;\btheta;\tilde\bx)] \\
   P(y=k|\bbeta;\btheta;\tilde\bx)  &= \begin{cases}
   1 - \sigma(\theta_{K-1}-\tilde\eta) & \text{ if } k=K \\
   \sigma(\theta_{K-1}-\tilde\eta) - \sigma(\theta_{K-2}-\tilde\eta) & \text{ if } k=K-1 \\
   \vdots & \vdots \\
   \sigma'(\theta_{1}-\tilde\eta) - 0 & \text{ if } k=1
   \end{cases}
   \end{align*}
   $$ -->

Code
----

   Coming soon
