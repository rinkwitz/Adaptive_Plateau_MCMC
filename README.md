# Adaptive_Plateau_MCMC

## Introduction

The Markov Chain Monte Carlo (MCMC) sampling tries to investigate properties of
distributions by drawing random samples. This is a sequential process, where the drawing
of a random sample at any given time depends only on its immediate predecessor. In this
project a variant of the MCMC algorithm from the paper [Lau and Krumscheid, 2019] is
reimplemented and an attempt is made to reproduce the results of the experiments. The
paper under consideration builds on the ideas of [Metropolis et al., 1953, Geyer, 1992,
Liu et al., 2000] and [Yang et al., 2019].

## Adaptive Component-wise Multiple-Try Metropolis Algorithm
The core algorithm
    of the paper [Lau and Krumscheid, 2019] consists of a MCMC algorithm which fulfills three properties. First, the algorithm suggests
    several suggestions from different plateau distributions during sampling. This happens independently for all
    components of a sample. Lastly, plateau distributions adapt their shape depending on the frequency of the accepted sample proposals.

### Plateau Proposal Distributions

The MCMC algorithm in [Lau and Krumscheid, 2019] uses *non-overlapping plateau proposal distributions* for sampling.
	The underlying probability density function <img src="https://latex.codecogs.com/gif.latex?f" /> is a combination of the density of a uniform distribution
	with exponential decaying tails. Here the density <img src="https://latex.codecogs.com/gif.latex?f" /> is constant around a mean value <img src="https://latex.codecogs.com/gif.latex?\mu" /> in the closed interval <img src="https://latex.codecogs.com/gif.latex?[\mu-\delta,\mu+\delta]" /> with <img src="https://latex.codecogs.com/gif.latex?\delta>0" />.
	Outside this interval the distribution follows an exponential decay with its tail width determined by <img src="https://latex.codecogs.com/gif.latex?\sigma_i>0" />,
	depending on whether you are on the left or right side of the interval <img src="https://latex.codecogs.com/gif.latex?[\mu-\delta,\mu+\delta]" />. If you define
	an unnormalized density function

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}\tilde{f}(y;\mu,\delta,\sigma_1,\sigma_2)&=\begin{cases}\exp\left(-\frac{1}{2\sigma_1^2}[y-(\mu-\delta)]^2\right),&\quad\,y<\mu-\delta\\1,&\quad\mu-\delta\leq\,y\leq\mu&plus;\delta\\\exp\left(-\frac{1}{2\sigma_2^2}[y-(\mu&plus;\delta)]^2\right),&\quad\,y>\mu&plus;\delta\end{cases}\end{align*}" title="\begin{align*}\tilde{f}(y;\mu,\delta,\sigma_1,\sigma_2)&=\begin{cases}\exp\left(-\frac{1}{2\sigma_1^2}[y-(\mu-\delta)]^2\right),&\quad\,y<\mu-\delta\\1,&\quad\mu-\delta\leq\,y\leq\mu+\delta\\\exp\left(-\frac{1}{2\sigma_2^2}[y-(\mu+\delta)]^2\right),&\quad\,y>\mu+\delta\end{cases}\end{align*}" /></p>

and then calculate the following integral

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;C(\delta,\sigma_1,\sigma_2)&=\int\limits_{-\infty}^\infty\tilde{f}(y;\mu,\delta,\sigma_1,\sigma_2)&space;dy\\&space;&=&space;\int\limits_{-\infty}^{\mu-\delta}\exp\left(&space;-\frac{1}{2\sigma_1^2}[y-(\mu-\delta)]^2&space;\right)dy&plus;&space;\int\limits_{\mu-\delta}^{\mu&plus;\delta}1dy&plus;&space;\int\limits_{\mu&plus;\delta}^{\infty}\exp\left(&space;-\frac{1}{2\sigma_2^2}[y-(\mu&plus;\delta)]^2&space;\right)dy\\&space;&=\frac{\sqrt{2\pi\sigma_1^2}}{2}&plus;2\delta&plus;\frac{\sqrt{2\pi\sigma_2^2}}{2}&space;\end{align*}" />
</p>
    
as the sums of 2 half gaussian integrals and one integral over a constant function, then the normalized probability density function is <img src="https://latex.codecogs.com/gif.latex?f(y;\mu,\delta,\sigma_1,\sigma_2)=C(\delta,\sigma_1,\sigma_2)^{-1}\tilde{f}(y;\mu,\delta,\sigma_1,\sigma_2)" /> [Lau and Krumscheid, 2019].
    Using <img src="https://latex.codecogs.com/gif.latex?f" /> you can define the plateau probability density distributions <img src="https://latex.codecogs.com/gif.latex?T_{j,k}" />, <img src="https://latex.codecogs.com/gif.latex?j\in\{1,\dots,M\}" /> for the trial proposals of the <img src="https://latex.codecogs.com/gif.latex?k" />-th component as

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;T_{j,k}(x,y)&=\begin{cases}&space;f(y;x,\delta_1,\sigma,\sigma),&j=1\\&space;\frac&space;12&space;f(y;x-(2j-3)\delta-\delta_1,\delta,\sigma,\sigma)&plus;\frac&space;12&space;f(y;x&plus;(2j-3)\delta&plus;\delta_1,\delta,\sigma,\sigma),&j=2,\dots&space;M-1\\&space;\frac&space;12&space;f(y;x-(2M-3)\delta-\delta_1,\delta,\sigma_0,\sigma)&plus;\frac&space;12&space;f(y;x&plus;(2M-3)\delta&plus;\delta_1,\delta,\sigma,\sigma_1),&j=M&space;\end{cases}&space;\end{align*}" title="\begin{align*} T_{j,k}(x,y)&=\begin{cases} f(y;x,\delta_1,\sigma,\sigma),&j=1\\ \frac 12 f(y;x-(2j-3)\delta-\delta_1,\delta,\sigma,\sigma)+\frac 12 f(y;x+(2j-3)\delta+\delta_1,\delta,\sigma,\sigma),&j=2,\dots M-1\\ \frac 12 f(y;x-(2M-3)\delta-\delta_1,\delta,\sigma_0,\sigma)+\frac 12 f(y;x+(2M-3)\delta+\delta_1,\delta,\sigma,\sigma_1),&j=M \end{cases} \end{align*}" />
</p>

with values <img src="https://latex.codecogs.com/gif.latex?\delta_1,\delta,\sigma,\sigma_0,\sigma_1>0" />. In figure \ref{trial_proposals}
    you can see the trial proposal propability density distributions for the parameters <img src="https://latex.codecogs.com/gif.latex?M=5,\delta_1=\delta=1,\sigma=0.05,\sigma_0=\sigma_1=0.5" />. One can see that the
    distributions overlap only at their exponential decaying tails. The outer tails decrease with the larger <img src="https://latex.codecogs.com/gif.latex?\sigma_0,\sigma_1" /> values
    shallower than the remaining tails. In the paper [Lau and Krumscheid, 2019], the authors consistently use the values <img src="https://latex.codecogs.com/gif.latex?\delta=\delta_1=2,\sigma=0.05,\sigma_0=\sigma_1=3" />.

<p align="center">
<img src="/figs/fig_2b.png" width="500" alt="trial proposal propability density distributions"><br>
Trial proposal propability density distributions for <img src="https://latex.codecogs.com/gif.latex?M=5" />
</p>

### Component-wise Multiple-Try Metropolis
The component-wise multiple-try Metropolis algorithm [Lau and Krumscheid, 2019, Algorithm 1],
    which forms the basis of the sampling procedure, starts from a starting position
	<img src="https://latex.codecogs.com/gif.latex?x_0\in\mathbb{R}^d" />. For each MCMC realization <img src="https://latex.codecogs.com/gif.latex?x_n" /> with <img src="https://latex.codecogs.com/gif.latex?n\in\{1,\dots,N\}" /> the following procedure is used for each one of the <img src="https://latex.codecogs.com/gif.latex?d" /> components. If <img src="https://latex.codecogs.com/gif.latex?x=(x_1,\dots,x_d)" /> is the last sampled candidate of the MCMC algorithm, then the algorithm proposes trials <img src="https://latex.codecogs.com/gif.latex?z_j" />
    for <img src="https://latex.codecogs.com/gif.latex?i=1,\dots,M" /> by sampling it from the distributions <img src="https://latex.codecogs.com/gif.latex?T_{j,k}(x_k,\cdot)" />. In my reimplementation
    I use a rejection sampling procedure [Peng, 2018]. For this purpose I use an uniform distribution to generate the samples over the following intervals

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;I_j&space;=&space;\begin{cases}&space;[x_k-\delta_1-t_1,x_k&plus;\delta_1&plus;t_1]\\\text{&space;for&space;}j=1\text{&space;with&space;}t_1&space;=&space;\sqrt{-2\sigma^2\log(0.0001C(\delta_1,\sigma,\sigma))},\vspace{.25cm}\\&space;[x_k-2(j&plus;1)\delta-\delta_1-t_2,x-2j\delta-\delta_1&plus;t_2]\cup[x_k&plus;2j\delta&plus;\delta_1-t_2,x&plus;2(j&plus;1)\delta-\delta_1&plus;t_2]\\\text{&space;for&space;}j=2,\dots,M-1\text{&space;with&space;}t_2=\sqrt{-2\sigma^2\log(0.0002C(\delta,\sigma,\sigma))},\text{&space;or}\vspace{.25cm}\\&space;[x_k-2(M&plus;1)\delta-\delta_1-t_{32},x_k-2M\delta-\delta_1&plus;t_{31}]\cup[x_k&plus;2M\delta&plus;\delta_1-t_{31},x_k&plus;2(M&plus;1)\delta&plus;\delta_1&plus;t_{32}]\\\text{&space;for&space;}j=M\text{&space;with&space;}t_{31}=\sqrt{-2\sigma^2\log(0.0002C(\delta,\sigma,\sigma_0))},t_{32}=\sqrt{-2\sigma_0^2\log(0.0002C(\delta,\sigma,\sigma_0))}.&space;\end{cases}&space;\end{align*}" title="\begin{align*} I_j = \begin{cases} [x_k-\delta_1-t_1,x_k+\delta_1+t_1]\\\text{ for }j=1\text{ with }t_1 = \sqrt{-2\sigma^2\log(0.0001C(\delta_1,\sigma,\sigma))},\vspace{.25cm}\\ [x_k-2(j+1)\delta-\delta_1-t_2,x-2j\delta-\delta_1+t_2]\cup[x_k+2j\delta+\delta_1-t_2,x+2(j+1)\delta-\delta_1+t_2]\\\text{ for }j=2,\dots,M-1\text{ with }t_2=\sqrt{-2\sigma^2\log(0.0002C(\delta,\sigma,\sigma))},\text{ or}\vspace{.25cm}\\ [x_k-2(M+1)\delta-\delta_1-t_{32},x_k-2M\delta-\delta_1+t_{31}]\cup[x_k+2M\delta+\delta_1-t_{31},x_k+2(M+1)\delta+\delta_1+t_{32}]\\\text{ for }j=M\text{ with }t_{31}=\sqrt{-2\sigma^2\log(0.0002C(\delta,\sigma,\sigma_0))},t_{32}=\sqrt{-2\sigma_0^2\log(0.0002C(\delta,\sigma,\sigma_0))}. \end{cases} \end{align*}" />
</p>

These intervals enable effective sampling in the range of <img src="https://latex.codecogs.com/gif.latex?T_{j,k}(x_k,\cdot)" /> where the probability density function is greater than <img src="https://latex.codecogs.com/gif.latex?0.0001" />.
    To do this, one samples a <img src="https://latex.codecogs.com/gif.latex?u\sim\,U(0,1)" /> and a <img src="https://latex.codecogs.com/gif.latex?y\sim\,U(I_i)" /> until

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;u&space;<&space;\frac{T_{j,k}(x_k,y)}{c|I_i|}&space;\end{align*}" title="\begin{align*} u < \frac{T_{j,k}(x_k,y)}{c|I_i|} \end{align*}" />
</p>

is fulfilled where <img src="https://latex.codecogs.com/gif.latex?|I_j|" /> is the width of the used interval <img src="https://latex.codecogs.com/gif.latex?I_j" />, <img src="https://latex.codecogs.com/gif.latex?g_j" /> is the probability density function of the uniform distribution over <img src="https://latex.codecogs.com/gif.latex?I_j" /> and <img src="https://latex.codecogs.com/gif.latex?c_j" /> has the following values situation dependent

* <img src="https://latex.codecogs.com/gif.latex?j=1:\quad\,c_1=\sup\limits_{y\in&space;I_1}\frac{T_{1,k}(x_k,y)}{g_1(y)}=\frac{|I_1|}{C(\delta_1,\sigma,\sigma)}" title="j=1:\quad\,c_1=\sup\limits_{y\in&space;I_1}\frac{T_{1,k}(x_k,y)}{g_1(y)}=\frac{|I_1|}{C(\delta_1,\sigma,\sigma)}" />,
* <img src="https://latex.codecogs.com/gif.latex?j=2:\quad\,c_j=\sup\limits_{y\in&space;I_j}\frac{T_{j,k}(x_k,y)}{g_j(y)}=\frac{|I_j|}{2C(\delta,\sigma,\sigma)}" title="j=2:\quad\,c_j=\sup\limits_{y\in I_j}\frac{T_{j,k}(x_k,y)}{g_j(y)}=\frac{|I_j|}{2C(\delta,\sigma,\sigma)}" />, and
* <img src="https://latex.codecogs.com/gif.latex?j=M:\quad\,c_M=\sup\limits_{y\in&space;I_M}\frac{T_{M,k}(x_k,y)}{g_M(y)}=\frac{|I_M|}{2C(\delta,\sigma,\sigma_0)}" title="j=M:\quad\,c_M=\sup\limits_{y\in I_M}\frac{T_{M,k}(x_k,y)}{g_M(y)}=\frac{|I_M|}{2C(\delta,\sigma,\sigma_0)}" />.

Afterwards the weights associated with the trials are

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;w_{j,k}&=\pi((z_j;x_{[-k]}))T_{j,k}(x_k,z_j)\lambda_{j,k}(x_k,z_j),\quad\text{for&space;}j=1,\dots,M&space;\end{align*}" title="\begin{align*} w_{j,k}&=\pi((z_j;x_{[-k]}))T_{j,k}(x_k,z_j)\lambda_{j,k}(x_k,z_j),\quad\text{for }j=1,\dots,M \end{align*}" />
</p>

are calculated whereby <img src="https://latex.codecogs.com/gif.latex?(z;x_{[-i]})\in\mathbb{R}^d" /> denotes the vector, which is identical to <img src="https://latex.codecogs.com/gif.latex?x" /> in all entries except the <img src="https://latex.codecogs.com/gif.latex?i" />-th one where it takes on the value <img src="https://latex.codecogs.com/gif.latex?z" />.
    In addition, the non-negative and symmetric function

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\lambda_{j,k}(x,y)&=|y-x|^{2.5}\end{align*}" />
</p>

is used. Here the authors of [Lau and Krumscheid, 2019] refer to the results of [Yang et al., 2019]. Thus, suggestions <img src="https://latex.codecogs.com/gif.latex?(z_j;x_{[-k]})" /> have a high weight, which have a high probability with regard to
    the target distribution <img src="https://latex.codecogs.com/gif.latex?\pi" />, whose new proposal <img src="https://latex.codecogs.com/gif.latex?z_j" /> for the <img src="https://latex.codecogs.com/gif.latex?k" />-th component regarding <img src="https://latex.codecogs.com/gif.latex?T_{j,k}(x_k,\cdot)" /> is very likely and where the distance function <img src="https://latex.codecogs.com/gif.latex?\lambda_{j,k}" /> is far enough away from <img src="https://latex.codecogs.com/gif.latex?x_k" />.
    Proportionally to the weights <img src="https://latex.codecogs.com/gif.latex?w_{1,k},\dots,w_{M,k}" /> a <img src="https://latex.codecogs.com/gif.latex?y\in\{z_1,\dots,z_M\}" /> is then randomly drawn and for <img src="https://latex.codecogs.com/gif.latex?j=1,\dots,M-1" /> the algorithm samples
    <img src="https://latex.codecogs.com/gif.latex?x_j^*\sim\,T_{j,k}(y,\cdot)" />. Finally one calculates

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\alpha&=\min\left\{&space;1,\frac{w_{1,k}(z_1,x)&plus;\dots&plus;w_{M,k}(z_M,x)}{w_{1,k}(x_1^*,(y;x_{[-k]}))&plus;\dots&plus;w_{M-1,k}(x_{M-1}^*,(y;x_{[-k]}))&plus;w_{M,k}(x_k,(y;x_{[-k]}))}&space;\right\}&space;\end{align*}" title="\begin{align*} \alpha&=\min\left\{ 1,\frac{w_{1,k}(z_1,x)+\dots+w_{M,k}(z_M,x)}{w_{1,k}(x_1^*,(y;x_{[-k]}))+\dots+w_{M-1,k}(x_{M-1}^*,(y;x_{[-k]}))+w_{M,k}(x_k,(y;x_{[-k]}))} \right\} \end{align*}" />
</p>

and with that probability <img src="https://latex.codecogs.com/gif.latex?\alpha" /> the algorithm accepts the new proposal and sets <img src="https://latex.codecogs.com/gif.latex?x_n=(y;x_{[-k]})" />. Otherwise the algorithm keeps the old proposal and sets <img src="https://latex.codecogs.com/gif.latex?x_n=x" />.

### Adaption of Proposal Distributions
In [Lau and Krumscheid, 2019] it is proposed to adapt the widths <img src="https://latex.codecogs.com/gif.latex?\delta" /> and <img src="https://latex.codecogs.com/gif.latex?\delta_0" /> of the plateau distributions.
    In my implementation every <img src="https://latex.codecogs.com/gif.latex?L" /> iterations it measures how high the number of selected suggestions <img src="https://latex.codecogs.com/gif.latex?c_{j,k}" /> from the plateau distributions <img src="https://latex.codecogs.com/gif.latex?T_{j,k}" /> are in this interval.
    If the middle plateau distribution is called much more than the average, that is <img src="https://latex.codecogs.com/gif.latex?c_{j,k}>L\eta_1" /> with <img src="https://latex.codecogs.com/gif.latex?\eta_1\in(0,1)" />, then
    the algorithm assumes that the plateaus are too wide and halves <img src="https://latex.codecogs.com/gif.latex?\delta" /> and <img src="https://latex.codecogs.com/gif.latex?\delta_1" /> accordingly for
    the following iterations. If, on the other hand, the outermost plateau distributions are selected much more than the average, in especially
    <img src="https://latex.codecogs.com/gif.latex?c_{M,k}>\eta_2L" /> with <img src="https://latex.codecogs.com/gif.latex?\eta_2\in(0,1)" />, then the algorithm assumes that the plateaus are too small and doubles
    <img src="https://latex.codecogs.com/gif.latex?\delta" /> and <img src="https://latex.codecogs.com/gif.latex?\delta_1" />. The adaptations only take place with a constantly decreasing probability of <img src="https://latex.codecogs.com/gif.latex?\max(0.99^{n-1},1/\sqrt{n})" />. This implementation of the adaptation procedure is based on [Lau and Krumscheid, 2019, Algorithm 2].

## References

* [Lau and Krumscheid, 2019] Lau, F. D.-H. and Krumscheid, S. (2019). Plateau proposal
distributions for adaptive component-wise multiple-try metropolis.
* [Metropolis et al., 1953] Metropolis, Nicholas, W., A., Rosenbluth, N., M., H., A., Teller,
and Edward (1953). Equation of state calculations for fast computing machines.
Journal of Chemical Physics 6, 21:1087–.
* [Yang et al., 2019] Yang, J., Levi, E., Craiu, R. V., and Rosenthal, J. S. (2019). Adapt-
ive component-wise multiple-try metropolis sampling. Journal of Computational and
Graphical Statistics, 28(2):276–289.
* [Peng, 2018] Peng, R. D.(2018). Advanced statistical computing.
https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html.
Accessed: 2020-03-17.

## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements:

The formulas of this README were create using:
* [Codecogs online Latex editor](https://www.codecogs.com/latex/eqneditor.php)
