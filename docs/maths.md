# Maths
## Immediate Thresholds

As is usual in ordinal regression model formulations, we build a regressor that learns a latent variable $y$, and then use a set of thresholds $\Theta$ to produce the probability estimates for each label. The set of thresholds is ordered, does not include infinities, and has as many members as the numbers of labels minus one.

We want to come up with a a way to map the latent variable $y$ to the probability space such that when $y$ is in $(\theta_{k-1},\theta_{k})$ the probability of label $k$ is maximised.

In a three ordered labeled problem, we only need two thresholds, $\theta_1$ and $\theta_2$, to define the three regions which are associated to each label $(-\infty,\theta_1], (\theta_1, \theta_2], (\theta_2, \infty)$.

## Deriving probabilities

A property we want our mapping from latent variable to probability to have is for the cummulative probability of label $z$ being at most label $k$ to increase as the label increases. This means that $P(z\leq k;y,\Theta)$ should increase as $k$ increases (i.e. as we consider more labels).

Another property is that as the latent variable $y$ gets smaller, the cummulative probability should also increase, and as it gets larger it should decrease.

To satisfies this properties we use a function $F$ which grows as its argument grows and shrinks as the arguments shrink. We can then define the cumulative probability as:
$$
P(z \leq k; y,\Theta  ) = F(\theta_k - y) ,
$$

making sure that the range of $F$ is contrained to the $(0,1)$ interval. This formulation satisfies all of our properties. As we consider larger (higher in the order) labels $k$, the threshold $\theta_k$ grows and so does the cumulative probability. As $y$ grows, the input to $F$ shrinks and so does the cumulative probability.

Naturally, the probability of $z$ being any particular label is then:
$$
\newcommand{\problessthank}{P(z \leq k; y,\Theta  )}
% \newcommand{\bbeta}{\mathbf{b}}
% \newcommand{\btheta}{\mathbf{\theta}}
\begin{align*}
P(z = k; y,\Theta  ) &=P(z \leq k; y,\Theta) -P(z \leq k-1; y,\Theta  )  \hspace{2mm} \\
&= F(\theta_k - y) - F(\theta_{k-1} - y)
\end{align*}
$$


A function that satisfies all these conditions is the sigmoid function, hereafter denoted as $\sigma$.
## Deriving the loss function

Given n samples, the likelihood of our set of predictions $y_i$ is:
$$
L(Y;\Theta) = \prod_{i =0}^n I(z_i=k)P(z_i = k; y_i,\Theta)
$$

As is usual in machine learning we use the negative log likelihhod as our loss:

$$
\begin{align}
l(Y;\Theta) &= -\log L(Y,\theta)\\
&= -\sum_{i=0}^n I(z_i=k)\log(P(z_i = k; y_i,\Theta)) \\
&= -\sum_{i=0}^n I(z_i=k)\log \left(\sigma(\theta_k - y) - \sigma(\theta_{k-1} - y)\right)
\end{align}
$$
## Deriving the gradient and hessian

To use a custom loss function with gradient boosting tree frameworks (i.e. lightgbm), we have to first derive the gradient and hessian of the loss with respect to **the raw predictions**, in our case the latent variable $y_i$.




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
$$