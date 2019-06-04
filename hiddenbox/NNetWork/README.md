# The Layers

## RNN

## sigmoidCrossEntropyWithLogits

x = logits, z = targets

$$
\begin{aligned}
Y
&= - z \log(\sigma(x)) - (1-z)\log(1 - \sigma(x)) \\
&= - z \log\left( \dfrac{1}{1 + e^{-x}} \right) - (1-z)\log\left(1 - \dfrac{1}{1 + e^{-x}} \right) \\
&= z \log(1 + e^{-x}) + (1-z)(x + \log(1 + e^{-x})) \\
&= x - zx + \log(1 + e^{-x})
\end{aligned}
$$

$$
\begin{aligned}
\dfrac{d Y}{d x}
&= 1 - z + \dfrac{1}{1 + e^{-x}} (-e^{-x})\\
&= 1 - \dfrac{e^{-x}}{1 + e^{-x}} - z \\
&= \dfrac{1}{1 + e^{-x}} - z \\
&= \sigma(x) - z
\end{aligned}
$$

## Least Square

x = logits, z = targets
Define:

$$y = (x - z)^2$$

then:

$$\frac{dy}{dx} = 2(x-z)$$

