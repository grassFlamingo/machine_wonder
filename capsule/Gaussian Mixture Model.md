# Build and Learn the model - Gaussian mixture model (GMM)

Let $X = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_I\}$ as $I$ **independent** datapoints, and $C = \{\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_J\}$ as $J$ clusters with dimension $d$. The pobility of datapoint $\mathbf{x}_i$ match to cluster $c_j$ is $p(\mathbf{c}_j) = \phi_{j}$, $\sum_j \phi_{j} = 1$.

The aim is to estimate the unknown parameters representing the mixing value between the Gaussians and the means and covariances of each:

$$\theta = \left( \phi_{j}, \boldsymbol{\mu}_j, \Sigma_j \right)$$

where $\mu_j$ are custer centers and $\Sigma_j$ are covariance matrixes.

The likelihood $L(\theta; X,C) = p(X,C | \theta)$.

$$
\arg \max_{\theta} L(\theta; X,C) \Leftrightarrow \arg \max_{\theta} \log L(\theta; X, C)
$$

$$
\begin{aligned}
\log L(\theta; X,C) &= \log p(X,C | \theta)\\
&= \log \prod_{i,j}^{I,J} \phi_{j} \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \Sigma_j) \\
&= \log \prod_{i,j}^{I,J} \phi_{j} \dfrac{1}{\sqrt{(2 \pi)^d |\Sigma_j|}} \exp\left(
        - \frac{1}{2} (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
    \right)\\
&= \sum_{i,j}^{I,J}
    \log\left(
        \phi_{j} \dfrac{1}{\sqrt{(2 \pi)^d |\Sigma_j|}} \exp\left(
        - \frac{1}{2} (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
    \right)
\right)\\
&= \sum_{i,j}^{I,J} \left(
    \log\phi_{j} 
    - \frac{d}{2} \log 2 \pi
    - \frac{1}{2} \log |\Sigma_j|
    - \frac{1}{2}
         (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
    \right)\\
\end{aligned}
$$


## E-Step

Given our current estimate parameters $\theta^{(t)}$, the conditional distribution of the $c_j$ is determined by Bayes theorem:

$$
p(C| X, \theta^{(t)})
= \dfrac{
    p(\mathbf{x}_i, \theta^{(t)} | \mathbf{c}_j) p(\mathbf{c}_j)
}{
    \sum_k^J p(\mathbf{x}_i, \theta^{(t)} | \mathbf{c}_k) p(\mathbf{c}_k)
}
= \dfrac{
    \phi_{j} \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \Sigma_j)
}{
    \sum_k^{J} \phi_{k} \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \Sigma_k)
}
\equiv T_{ij}^{(t)}
$$

Obviously, $\sum_j T_{ij}^{(t)} = 1$, $\sum_{ij} T_{ij}^{(t)} = J$.

The expection is 

$$
\mathbb{E}_{C|X, \theta^{(t)}} [\log L(\theta; X, C)]
= \sum_{i,j}^{I,J} T_{ij}^{(t)} \left(
    \log\phi_{j}
    - \frac{d}{2} \log 2 \pi
    - \frac{1}{2} \log |\Sigma_j|
    - \frac{1}{2} (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
\right)
$$

## M-Step

### Estimate $\phi_{j}$

$$
\begin{aligned}
    \phi_{j}^{(t+1)} 
    &= \argmax\limits_{\phi_{j}} \mathbb{E}_{C|X, \theta^{(t)}} [\log L(\theta; X, C)]\\
    &\Rightarrow \argmin\limits_{\phi_{j}} \sum_{i,j}^{I,J} - T_{ij}^{(t)} \log \phi_{j}\\
    &\qquad \text{s.t.} \sum_j \phi_j = 1
\end{aligned}
$$

Lagrange method:

$$
\mathcal{L}(\phi_j, \lambda) = - T_{ij}^{(t)} \log \phi_j - \lambda (1 - \sum_j \phi_j)
$$

Please notice that $T_{ij}^{(t)}$ here is come from E-step, there it is **not** a function of $\phi_j$. We take the partial derivatives and set them to zero. 

$$
\begin{aligned}
    \dfrac{\partial \mathcal{L}}{\partial \phi_j} 
    &= - \dfrac{1}{\phi_j} \sum_{i} T_{ij}^{(t)} + \lambda \equiv 0\\
    \dfrac{\partial \mathcal{L}}{\partial \lambda} 
    &= 1 - \sum_j \phi_j \equiv 0
\end{aligned}
$$

Therefore, we have:

$$
\lambda \phi_j = \sum_i T_{ij}^{(t)} 
\leftrightarrow 
\sum_j \lambda \phi_j = \sum_{ij} T_{ij}^{(t)} 
\leftrightarrow
\lambda = J
$$
$$
\phi_j^{t+1} = \dfrac{1}{J} \sum_{i} T_{ij}^{(t)} 
$$


### Estimate $\mu_{j}$

$$
\begin{aligned}
    \mu_{j}^{(t+1)} 
    &= \argmax\limits_{\mu_{j}} \mathbb{E}_{C|X, \theta^{(t)}} [\log L(\theta; X, C)]\\
    &\Rightarrow \argmin\limits_{\mu_{j}} \sum_i T_{ij}^{(t)}
        (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
\end{aligned}
$$

Let $g(\mu_j) = \sum_i T_{ij}^{(t)} (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)$, take the derivative of $g$ with respect to $\mu_j$ is

$$
\dfrac{\partial g}{\partial \mu_j} 
= -2 \sum_{i} T_{ij}^{(t)} \Sigma_{j}^{-1} (x_i - \mu_j) = 0
$$

Thefore, the minimum $\mu_j$ is

$$
\mu_j^{(t+1)} 
= \dfrac{\sum_i T_{ij}^{(t)} x_i}{\sum_i T_{ij}^{(t)}}
$$

### Estimate $\Sigma_{j}$

$$
\begin{aligned}
    \Sigma_j^{(t+1)} 
    &= \argmax\limits_{\Sigma_j} \mathbb{E}_{C|X, \theta^{(t)}} [\log L(\theta; X, C)]\\
    &\Rightarrow \argmin\limits_{\Sigma_j} \sum_{i}^{I} T_{ij}^{(t)} \left(
        \log |\Sigma_j| + 
        (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j)
\right)
\end{aligned}
$$

Let $g(\Sigma_j) = \sum_{i}^{I} T_{ij}^{(t)} \left( \log |\Sigma_j| + (\mathbf{x}_i - \boldsymbol{\mu}_j)^T \Sigma_j^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_j) \right)$, and take the partial derivative of it, we have

$$
\dfrac{\partial g}{\partial \Sigma_j}
= \sum_i T_{ij}^{(t)} \left(
    \Sigma_j^{-T}
    - \left(
        (\mathbf{x}_i -  \boldsymbol{\mu}_j)
        \Sigma_j^{-1}
    \right)^T 
    \left(
      (\mathbf{x}_i - \boldsymbol{\mu}_j) \Sigma_j^{-1}
    \right)
\right)
= 0
$$

Therefore, we have

$$
\Sigma_j^{(t+1)} = 
\dfrac{
    \sum_i T_{ij}^{(t)} 
    (\mathbf{x}_i -  \boldsymbol{\mu}_j)^T
    (\mathbf{x}_i - \boldsymbol{\mu}_j)
}{
    \sum_i T_{ij}^{(t)}
}
$$

### Summize

1. E-step

$$
T_{ij}^{(t)} = \dfrac{
    \phi_{j} \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \Sigma_j)
}{
    \sum_k^{J} \phi_{k} \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \Sigma_k)
}
$$

2. M-Step

$$
\begin{aligned}
    \phi_j^{t+1} 
    &= \dfrac{1}{J} \sum_{i} T_{ij}^{(t)}\\
    \mu_j^{(t+1)}
    &= \dfrac{\sum_i T_{ij}^{(t)} x_i}{\sum_i T_{ij}^{(t)}}\\
    \Sigma_j^{(t+1)} 
    &= \dfrac{
        \sum_i T_{ij}^{(t)}
        (\mathbf{x}_i -  \boldsymbol{\mu}_j)^T
        (\mathbf{x}_i - \boldsymbol{\mu}_j)
    }{
        \sum_i T_{ij}^{(t)}
    }
\end{aligned}
$$

