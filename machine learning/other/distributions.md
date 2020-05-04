# Distributions


## Weibull Distribution

> ref: https://www.weibull.com/hotwire/issue14/relbasics14.htm </br>
> ref: https://en.wikipedia.org/wiki/Weibull_distribution


The most general expression of the Weibull *pdf* is given by the three-parameter Weibull distribution expression, 

$$
f(x; k, \lambda, \theta) = 
\left\{\begin{array}{cc}
\dfrac{k}{\lambda}
\left( \dfrac{x - \theta}{\lambda} \right)^{k - 1}
\exp\left(- \left(\frac{x - \theta}{\lambda}\right)^{k}\right), &x \ge \theta\\
0, & x < \theta
\end{array}\right.
$$

where
- $k > 0$ is the shape parameter, also known as the Weibull slope
- $\lambda > 0$ is the scale parameter
- $\theta$ is the location parameter

Frequently, the location parameter $\theta$ is not used, and the value for this parameter can be set to zero, which reduces to the 2-parameter distribution:

$$
f(x; k, \lambda) = 
\left\{\begin{array}{cc}
\dfrac{k}{\lambda}
\left( \dfrac{x}{\lambda} \right)^{k - 1}
e^{- \left(x / \lambda \right)^{k}}, &x \ge 0\\
0, & x < 0
\end{array}\right.
$$

The *cdf* of 2-parameter distribution is 
$$
F(x; k, \lambda) = 
\left\{\begin{array}{cc}
    1 - e^{- (x / \lambda)^k}, & x \ge 0\\
    0, & x < 0
\end{array}\right.
$$

The quantile (inverse cumulative distribution) function is
$$
Q(p; k, \lambda) = \lambda ( - \ln(1-p))^{1/k}, ~~ p \in [0, 1)
$$

Notes: gamma function $\Gamma(n) = \int_0^{\infty} e^{-x} x^{n-1} dx$

<!-- The mean is:
$$
\begin{aligned}
    E(f(x; k, \lambda)) 
    &= \int_0^{\infty} x \dfrac{k}{\lambda}
    \left( \dfrac{x}{\lambda} \right)^{k - 1}
    e^{- \left(x / \lambda \right)^{k}} dx\\
    &= \int_0^{\infty}     
\end{aligned}
$$ -->

<!-- If $U \sim (0,1)$ is a uniformly distribution, then the random vaariable $W = \lambda(- \ln(U))^{1/k}$ is Weibull distribution with parameters $k$ and $\lambda$.  -->

<!-- The Maximum likelihood estimator for $\lambda$ parameter given $k$ is -->
