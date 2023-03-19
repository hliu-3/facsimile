---
title: "494 Notes"
author: "Tom Liu"
date: '2023-03-19'
output: html_document
---



## Study Designs


#### Forward Propagation Problems:

Input Parameter `\(\vec{x}\)` `\(\to\)` Model a Functional `\(\mathcal{F}\)` `\(\to\)` Quantity of interest `\(y = \mathcal{F}(\vec{x})\)`. 

Note; The input of `\(\mathcal{F}\)` could be function themselves. For example, `\(\mathcal{F}\)` could be a differential operator with `\(\vec{x}\)` as a coefficient. 

e.g.1

$$ \mathcal{F} (f) = \frac{\partial f}{\partial z} \vec{x} $$
e.g.2

The Fourier Transform Operator (which is incidentally also a linear operator )


Question: If `\(\vec{x}\)` is stochastic, what can we say about `\(y = \mathcal{F}(\vec{x})\)`?

3 Pillars of solving Forward Propagation Problems:

 + Sampling Methods (Gibbs Sampling, MCMC, etc.)
 + Stochastic Collocation
 + Spectral Methods 

#### Monte Carlo

Broadly, Monte Carlo methods are those such that one acquires information about a system by simulating it with random sampling.




e.g.Nagel-SchreKenberg Traffit Modelling: Some traffic jams have no obvious cause. An aerial videos show that traffic jams spontaneously appear, move "backwards" (against traffic flow), and eventually disappears. 


Model Assumptions:

 + Single Lane
 + M chuncks of road labelled 0,...,M-1
 + Discrete time steps of identical length
 + One way
 + Each hypothetical car occupies one space. 
 + Constant flow of cars 
 + Car with velocity `\(v\geq 0\)` move `\(v\)` steps foward in `\(1\)` time step 
 + Some speed limit $v_{\max} > 0 $


Update Rules For a Single Car: 

Input: 
  + position `\(x\in\{0,1,\cdots, M-1\}\)`
  + Distance `\(d>0\)` between it and the car in front of it
  + Given probability `\(0<p<1\)` of randomly slowing down ($p$ is a constant, same for all cars ). 
  
  Update: 
  + If the car's velocity is less that `\(v_{\max}\)`, increase `\(v\)` by `\(1\)` unit, because drivers want to go. 
  + If `\(v*1(\text{time unit}) \geq d\)`, set `\(v\)` to `\(d-1\)` unit to avoid collision.
  + With probability `\(p\)`, slow down by 1 unit, set `\(v = \max(v-1,0)\)` with probability `\(p\)`. 
  
  Post-Update: Move Forward `\(v\)` spots. 
  
  Initial Positions: `\(N\)` cars put in `\(M\)` slots by samping `\(N\)` total values from `\(\{0,\cdots,M-1\}\)` uniformly without replacement. 
  
  Intitial Velocity: every car starts with Velocity 0, with a "burn in" perior (about 3000 steps if p = 0.5)
  
  Review: A continuous random variable `\(X\)` is a continuous map `\(X: \Omega \to \mathbb{R}\)`, where
    + `\(\Omega\)` is any non empty set (event space) representing "abstract elementary events" (something in the real world we actually observe that doesn't immediately have an associated numerical value)
    + `\(X\)` is effectively a probability measure
    + "Randomness" comes from not knowing the input event `\(\omega\in\Omega\)`.
    + We can assign a probability associated with certain outcomes:
    
    $$ \mathbb{P}(X\leq a) = \mathbb{P}(\{\omega\in\Omega|X(\omega)\leq a\})$$
    

    
A probability density function of a random variable `\(X\)` is a map

$$f_x \to \mathbb{R}\geq 0 $$
The property of a probability density function is derived naturally from the property of measures:

(1). For `\(a,b\in \mathbb{R}\)`, we have 

$$ P(a\leq X \leq b) = \int_{[a,b]} f_X(s) d\,s  $$

(2). The probability of the sample space is `\(1\)`, namely:

$$\int_{\Omega} f_X(s) d\,s = 1  $$

(3). `\(f_X \geq 0\)`

e.g.1. `\(X\)` conforms to a standard normal distribution `\(X\sim \mathcal{N}(0,1)\)`.

$$f_X: \mathbb{R} \to \mathbb{R} \geq 0 $$

`$$f_X(y) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{y^2}{2})$$`

e.g.1. `\(X\)` conforms to a uniform distribution `\(X\sim \mathcal{U}(c,d)\)`.

$$f_X: \mathbb{R} \to \mathbb{R} \geq 0 $$

`$$f_X(y) = \frac{1}{d-c}$$`

Mean and Variance

For a continuous random variable `\(X\)` with probability distribution function `\(f_X(x)\)`, define the mean as 

$$ \mathbb{E}[X] = \int_{-\infty}^{\infty} yf_X(y) dy $$

$$ \text{Var}[X] = \int_{-\infty}^{\infty} (y-\mathbb{E}[X] )^2 f_X(y) dy $$


Theorem: Consider some independent and identically distributed random variables `\(\{Y_1,Y_2,\cdots,Y_n\}\)`. If `\(\mathbb{E}[Y_i]<\infty\)`,

then $$ s_n := \frac{1}{n} \sum_{i=1}^n Y_i $$

converges in "some sense" to the true mean `\(\mathbb{E}[Y_i]\)`. 

In particular, the **weak law of large number** states that for all `\(\epsilon>0\)`, 

`$$\lim_{n\to\infty} \mathbb{P} (|s_n - \mathbb{E}(y_i)|>\epsilon) = 0$$`

(This is called convergence in probability)

Simple Monte Carlo:

+ Express some quantity of interest as the expected value of some RV Y

+ Generate values `\(Y_1,\cdots,Y_n\)` independently from the distribution of `\(Y\)`

+ Take the average $s_n = \displaystyle \frac{1}{n} \sum_{i=1}^n Y_i $

e.g.1 Suppose we want to evaluate `\(\pi\)`. We randomly generate vectors in `\(z \in [-1,1]\times[-1,1]\)` and see how many of them satisfies `\(\|z\|\leq 1\)`. As `\(N\to\infty\)`, the ratio approximates `\(\pi/4\)`


```
## [1] 3.14056
```

$$ \mathbb{E}[Y_i] =  \mathbb{E}[Y_i|\vec{X^{(i)}}\in \mathcal{R}] \mathbb{P} [\vec{X^{(i)}}\in \mathcal{R}] + \mathbb{E}[Y_i|\vec{X^{(i)}}\notin \mathcal{R}] \mathbb{P} [\vec{X^{(i)}}\notin \mathcal{R}] = \mathbb{P} [\vec{X^{(i)}}\in \mathcal{R}]  $$

$$ = \int_{\mathcal{R}} f_{\vec{X^{(i)}}} (x,y) d\,y d\,x  $$

Generate $\vec{X^{(i)}} \sim (\mathcal{U}(0,1),\mathcal{U}(0,1))  $ and compute the ratio of the dots in the area over total number of dots. 

Pro:

+ This estimate is unbiased because `\(\mathbb{E}[S_n] = \mathbb{E[Y]}\)`

Con:

+ We do not know how big `\(n\)` should be such that the error 

`$$\epsilon = \|\mathbb{E}[s_n] - \mathbb{E}[Y_i] \|$$`

If `\(\text{Var}[Y]=\sigma^2 <\infty\)`, 

$$ \text{Var}[S_n] = \text{Var} \left[\sum_{i=1}^n \frac{1}{n} Y_i\right] =  \sum_{i=1}^n \text{Var} \left[\frac{1}{n} Y_i\right] = \frac{1}{n^2}\sum_{i=1}^n \text{Var} \left[ Y_i\right] = \frac{\sigma^2}{n}$$

Root Mean Square Error:

For `\(p>1\)`, 

$$\text{p-th RMSE} = \left( \mathbb{E}[(s_n - \mathbb{E}(s_n))^p]^{\frac{1}{p}} \right) $$
Notably when `\(p=2\)` and `\(n\to\infty\)`,

`$$\text{2-nd RMSE} = \left( \mathbb{E}[(s_n - \mathbb{E}(s_n))^2]^{\frac{1}{2 }} \right)  = \left( \mathbb{E}[(s_n - \mathbb{E}(Y))^2]^{\frac{1}{2 }} \right)  = \sqrt{\text{Var}(S_n)}$$`

The simple MC has convergence on the order of `\(n^{-\frac12}\)` because `\(\sqrt{\text{Var}(S_n)} = \frac{\sigma}{\sqrt{n}}\)`. 

This is slow, but independent of the dimensional `\(\alpha\)`



If `\(\text{Var}[Y]=\sigma^2 <\infty\)`, we can apply the Strong Law of Large Numbers, namely:




Quadrature Rules/ Methods:


$$\mathbb{E}[\mathcal{F}(X)] = \int_{\mathbb{R}} \mathcal{F}(y) f_Y(y) \,dy \simeq \sum_{k=1}^n \nu_k \mathcal{F} (\lambda_{k} ) $$

+ Gaussian Quadrature:

  * Summation is exact for polynomials of a certain max degree. 
  
+ Orthogonal Polynomials + GQ:
  * Let `\(\mathbb{P}\)` be the space of real polynomials on `\(\mathbb{R}\)`, and `$$\mathbb{P}_n = \{p\in \mathbb{P}|\deg(p)\leq n\}$$`
  
  * Let `\(\{\pi_k\}_0^\infty\)` be a set of polynomials that are orthogonal w.r.t. the PDF `\(f\)` of the function `\(\mathcal{F}\)`. (Example: The Legendre polynomials )
  
  * If `\(X\sim N(0,1)\)`, then `\(\{\pi_k\}_0^\infty\)` are called the Hermite polynomials. The "probabilist's Hermite polynomials" are given by $$
{\displaystyle { {He}}_{n}(x)=(-1)^{n}e^{\frac {x^{2}}{2}}{\frac {d^{n}}{dx^{n}}}e^{-{\frac {x^{2}}{2}}} }$$`
  
  * If `\(X\sim U[0,1]\)`, then `\(\{\pi_k\}_0^\infty\)` are called the Legendre polynomials.
  

We say that `\(\pi_i,\pi_j \in \mathbb{P}(\mathbb{R})\)` are orthogonal wrt a PDF `\(f\)` if 

$$ \int_{\mathbb{R}} \pi_i(y) \pi_j(y) f(y) \,dy = \delta_{i,j} $$

where the Kronecker's Delta 

$$\delta_{i,j} = \begin{cases} c \phantom{--} i = j \\ 0 \phantom{--} i \neq j\end{cases} $$

Alternatively, because `\(\pi_i(s),\pi_j(s)\in\mathbb{P}(\mathbb{R})\)`,

we can think of the L-2 inner product, namely:

$$ \langle \pi_i, \pi_j \rangle_{L^2(f)} $$
  
It is known that orthogonal polynomials satisfy a 3-term recurrence:

`$$\beta_{k+1} \pi_{k+1} (s) = (s-\alpha_k)\pi_k(s) - \beta_k \pi_{k-1}(s)$$`

where `\(k\in \mathbb{N}, \{\alpha_k\},\{\beta_k\}\subset \mathbb{R}\)`

For edge cases, let `\(\pi_{-1} = 0\)` and `\(\pi_0 = 1\)`.

Denote

$$\vec{\pi}(s) = \begin{pmatrix} \pi_0(s) \\ \pi_1(s) \\ \vdots \\ \pi_{n-1}(s) \end{pmatrix} $$

and `\(\vec{e}_n\)` = nth standard basis vector, then for `\(k\in\{0,1,\cdots,n-1\}\)` we can rewrite as:

$$ s\vec{\pi}(s) = \mathcal{J}_n \vec{\pi}(s) + \beta_n \pi_n(s) \vec{e}_n $$

where 

$$ \mathcal{J}_n =\begin{bmatrix} \alpha_0  & \beta_1 & \cdots \\ \beta_1 & \alpha_1 & \beta_2 \cdots \\ \phantom{-} & \ddots & \ddots&  \\ \phantom{-} & \ddots & \ddots & \beta_{n-1} \\ \phantom{-} & \cdots & \beta_{n-1}  & \alpha_{n-1}\end{bmatrix} $$

For proof consider induction. 

Now multiply 

$$ \pi_k: \mathbb{R} \to \mathbb{R} , \deg(\pi_k) = k , \pi_k (s) = \sum_{i=0}^k c_is^i$$

Multiply through by `\(\pi_k(s)\)` for fixed but arbitraty `\(k\)`,

`$$s\pi_k^2(s) = \beta_{k+1} \pi_{k+1}(s) \pi_{k} (s) +\alpha_k\pi^2_k(s)+ \beta_k \pi_{k-1}(s)\pi_{k+1}(s)$$`

Now consider 

$$ \int_{\mathcal{D}} s\pi_k^2(s) f(s) \,ds = \langle \text{id}, \pi_k^2\rangle $$

Note that 

$$ \int_{\mathcal{D}} \beta_{k+1} \pi_{k+1}(s) \pi_{k} (s) +\alpha_k\pi^2_k(s)+ \beta_k \pi_{k-1}(s)\pi_{k+1}(s) \,ds = \alpha_k\langle \pi_k,\pi_k \rangle_f $$

Thus

$$\alpha_k = \frac{\langle \text{id}, \pi_k^2\rangle_f}{\langle \pi_k, \pi_k\rangle_f} $$


Given that the polynomials can be orthonormal ($\|\pi_k\|_{L^2(\mathcal{C})} = 1$) a general form of `\(\alpha_i\)` and `\(\beta_i\)`, namely

`$$\begin{cases} & \alpha_i = \\ & \beta_i = \\ \end{cases}$$`
  
  
  
  
  
Inner product:

Let `\(V\)` be a vetor space with an underlying field `\(\mathcal{F}\)` for scalars,

Then an inner product is a map `\(\langle \cdot , \cdot  \rangle: V\times V \ to \mathcal{F}\)` s.t. for all `\(x,y,z, \in V\)` and `\(a\in\mathcal{F}\)`,

+ Conjugate Symmetry: `\(\langle x,y \rangle = \overline{\langle y,x \rangle}\)`

+ Lineararity: $\langle ax+y,z \rangle = a\langle x,z \rangle + \langle y,z \rangle $

+ Positive Definiteness: `\(x\neq 0_v \iff \langle x, x\rangle > 0\)` (works for matrix- just put the transpose to Hermitian )

Let `\(\xi\)` be a RV with PDF `\(f:\mathbb{R} \to \mathbb{R}\)`. Let `\(\mathcal{D}\)` be the support of `\(f\)`. 

Let `\(V = \mathcal{C}(\mathcal{D},\mathbb{R}) = \{g: \mathcal{D}\to \mathbb{R} | g \text{ is continuous}\}\)`

+ `\(V\)` is closed under linearity.

+ `\(0_v\)` exists 

Not necessarily but: `\(V\)` is complete under `\(\|\|_2\)`. That is, for any `\(u_n \subset V\)`, `$$\lim_{n\to\infty} u_n \in V.$$` ?

$$\langle g,h \rangle_f = \int_\mathcal{D} g(s)h(s) f(s) \,ds $$

`\(g,h,\in V\)`:

+ `\(\langle h,g \rangle  = \langle g,h \rangle\)` is immediate. 

+ `$$\begin{aligned} \langle ag+h,p \rangle_f  &= \int_\mathcal{D} [ag(s)+h(s)] p(s)f(s) \,ds \\ &= a \int_\mathcal{D} g(s)p(s)f(s) \,ds +  \int_\mathcal{D} h(s)p(s)f(s) \,ds \\ &= a\langle g,p\rangle_f + \langle h,p \rangle_f\end{aligned}$$`

+ Assume that `\(g\neq 0_v\)`. This implies `\(\exists s^*\in \mathcal{D}\)` s.t. `\(g^2(s^*)>0\)`. Due to the definition of the support, `\(f(s^*)>0\)`. 

$$\begin{aligned} \langle g,g \rangle_f  &= \int_\mathcal{D} g^2(s) f(s) \,ds >0 \end{aligned} $$

For two functions `\(g,h \in V=\mathcal{C}(\mathcal{D},\mathbb{R})\)`  to be orthogonal wrt the pdf of `\(\xi\)` , we need `\(\langle g,h \rangle_f = 0\)` if 

$$\begin{aligned} \langle g,h \rangle_f  &= \int_\mathcal{D} g(s)h(s) f(s) \,ds \\ &= 0  \end{aligned} $$

Recall that Let `\(\{\pi_k\}_0^\infty\)` be a set of polynomials that are orthogonal w.r.t. the PDF `\(f\)` of the function `\(\xi \in \mathcal{F}\)`, with 
`$$\pi_k: \mathcal{D}\to\mathbb{R} \text{ s.t. } \deg(\pi_k) = k$$`. (Example: The Legendre polynomials )

We say that `\(\pi_i,\pi_j \in \mathbb{P}(\mathbb{R})\)` are orthogonal wrt a PDF `\(f\)` for some continuous `\(\xi\)` if 

$$ \int_{\mathbb{R}} \pi_i(y) \pi_j(y) f(y) \,dy = c_i \delta_{i,j} $$

where the Kronecker's Delta 

$$\delta_{i,j} = \begin{cases} 1 \phantom{--} i = j \\ 0 \phantom{--} i \neq j\end{cases} $$


e.g.  `\(\{\pi_k\}_0^\infty\)` are Hermite `\(\to\)` `\(c_k=k!\)`

`\(\xi\)` distribution    |$\pi_k$ family 
----------------------|--------------------
`\(N(0,1)\)`              | Hermite
----------------------|--------------------
`\(U(-1,1)\)`             | Legendre
----------------------|--------------------
`\(\text{Beta}(\alpha,\beta)\)`  `\(\alpha>-1\)` | Jacobi
----------------------|--------------------
`\(\Gamma(\alpha,\beta)\)` `\(\alpha,\beta>-1\)`  | Laguerre



Fact:

If `\(\xi\)` has all finite moments, namely

$$ \int_\mathcal{D} s^k f(s) \,ds <\infty, k\in\mathbb{N}^+ $$

Then,


$$\text{span}( \{\pi_k\}_0^\infty )\subseteq  \mathcal{C}(\mathcal{D},\mathbb{R}) $$

(Sidenote: `\(\text{span}( \{\pi_k\}_0^\infty )\)` is equivalent to $L^2_\mathcal{C}(\mathcal{D},\mathbb{R}) $)

This is defined as `\(L^2_\mathcal{C}(\mathcal{D},\mathbb{R}\)`, also called the space of random variables with finite second moments (when `\(f\)`) is a pdf:

`$$\mathbb{E}(g(\xi)^2) = \int_{\mathcal{D}} g(x)^2 f(x) \, dx = \langle g,g \rangle_f$$`


Cauchy-Schwartz Inequality: for two elements `\(u,v\)` in a vector space, 

`$${\displaystyle \left|\langle \mathbf {u} ,\mathbf {v} \rangle \right|^{2}\leq \langle \mathbf {u} ,\mathbf {u} \rangle \cdot \langle \mathbf {v} ,\mathbf {v} \rangle }$$`
If `\(\xi\)` has all finite moments,

$$ \text{span} (\{\pi_k\}_{k=0}^\infty) = L^2_c(\mathcal{D},\mathbb{R},f)$$

 

SUbspace Theorem: if the subspace is closed under linear operations and contains the zero vector, then the subspace is itself a vector space. 

Note that in particular,

`$$\vec{\pi}(s) = \mathcal{J}_n \vec{
\pi}(s) + \beta_n \pi_n(s) \vec{e}_n$$`


And given that `\(\lambda_n\)` is a root of `\(\pi_n\)`,

$$ \lambda_n \vec{\pi} (\lambda_n) = \mathcal{J}_n \vec{\pi} (\lambda_n)  $$

which implies that 

`\(\vec{\pi} (\lambda_n)\)` is an eigenvector of `\(\mathcal{J}_n\)` with eigenvalue `\(\lambda_n\)`

Def: Let `\(A\in \mathbb{R}^{n\times n}\)`. We say that `\(v\in \mathbb{R}^{n} \neq \vec{0}\)` is an eigenvector of `\(A\)` if 

$$ Av= \lambda v $$ 

for some scalar `\(\lambda\)`. Then `\(v\)` is called the eigen vector of `\(A\)` with associated eigenvalue `\(\lambda\)`.


Consider `$$s\vec{\pi}(s) = \mathcal{J}_n \vec{\pi}(s) + \beta_n \pi_n(s) \vec{e}_n$$`

where 

$$\vec{\pi}(s) = \begin{pmatrix} \pi_0(s) \\ \pi_1(s) \\ \vdots \\ \pi_{n-1}(s) \end{pmatrix} $$

and 

$$ \mathcal{J}_n =\begin{bmatrix} \alpha_0  & \beta_1 & \cdots \\ \beta_1 & \alpha_1 & \beta_2 \cdots \\ \phantom{-} & \ddots & \ddots&  \\ \phantom{-} & \ddots & \ddots & \beta_{n-1} \\ \phantom{-} & \cdots & \beta_{n-1}  & \alpha_{n-1}\end{bmatrix} $$

Let `\(\lambda_j\)` be a root of the `\(\deg n\)` polynomial `\(\pi_n\)`. Thus `\(\pi_n(\lambda_j) = 0\)`. Substitute `\(s=\lambda_j\)` nn   

`$$\lambda_j \vec{\pi}(j) = \mathcal{J}_n \vec{\pi}(j) + \beta_n \pi_n(\lambda_j) \vec{e}_n$$`

And due to `\(\pi_n(\lambda_j) = 0\)`,

`$$\lambda_j \vec{\pi}(\lambda_j) = \mathcal{J}_n \vec{\pi}(\lambda_j)$$`

Is `\(\vec{\pi}(\lambda_j)\)` non-zero? `\(\pi_0 = 1\)` By definition, so yes, non-zero.

Thus $$\vec{\pi}(\lambda_j) = \begin{pmatrix} \pi_0(\lambda_j) \\ \pi_1(\lambda_j) \\ \vdots \\ \pi_{n-1}(\lambda_j) \end{pmatrix} $$ is an eigenvector of `\(\mathcal{J}_n\)` with eigenvalue `\(\lambda_j\)`.

Two important corrolaries:

+ `\(\lambda_j\)` are real-valued, since `\(\mathcal{J}_n\)` is symmetric. 

+ `\(\lambda_j\)` are distinct. (why? Proof of contradiction by class)

Gaussian Quadratures:

$$\mathbb{E}[\mathcal{F}(X)] = \int_{\mathbb{R}} \mathcal{F}(y) f_Y(y) \,dy \approx\sum_{k=1}^n \nu_k \mathcal{F} (\lambda_{k} ) $$

How "approximate"? 

Def: The Lagrange cardinal function are `\(\ell_i: \mathbb{R} \to \mathbb{R}\)` s.t.

`$$\ell_i(s) = \prod_{j\neq i} \frac{s-\lambda_j}{\lambda_i - \lambda_j }$$`

Corrolary: `\(\ell_j(\lambda_j) = \delta_{ij}\)`

Def: An `\(n\)`-point quadrature rule is exact of degree `\(m\)` if

$$\int_{\mathcal{D} }\mathcal{G}(y) f_Y(y) \,dy \approx\sum_{k=1}^n \nu_k \mathcal{G} (\lambda_{k} ) $$

where

$$ \nu_k = \int_{\mathcal{D} }  \ell_k (s) f(s) \, ds$$

The coefficients `\(\nu_k\)` are positive.

Pf：Consider `\(l_k^2\)`, as `\(\deg(l_k^2)=2n-2\)`.

By exactness (which we will soon show), the equation holds. 

$$
`\begin{aligned}
\langle l_k,l_k\rangle_f &=
\int_{\mathcal{D} }  \ell^2_k (s) f(s) \, ds \\ &= \sum_{k=1}^n \nu_k \ell^2_k  (\lambda_{k} ) \\
&= \sum_{k=1}^n \nu_k (\delta_{ik})^2 \\
&=\nu_k 
\end{aligned}`
$$

Note that `\(\langle l_k,l_k\rangle_f = \|l_k\|_f >0\)`. 

for all polynomials `\(\mathcal{G}\in \mathbb{P}(\mathcal{D},\mathbb{R})\)`.

Thm: Gaussian quadrature is exact of degree `\(2n-1\)`. 

+ For general quadratures, the quadrature points `\(\{\lambda_j\}_1^n \subset \mathbb{R}\)` are distinct. 

+ For Gaussian quadratures, the quadrature points `\(\{\lambda_j\}_1^n \subset \mathbb{R}\)` are picked to be the roots of `\(\{\pi_j\}_1^n\)`.

Suppose `\(\mathcal{G}\in \mathbb{P}^{n-1}\)`, then we will define the interpolation polynomial `\(Q^\mathcal{G}:\mathbb{R}\to\mathbb{R}\)` s.t. 

$$Q^\mathcal{G}(x) = \sum_{k=1}^n \mathcal{G}(\lambda_k) \ell_k(x) $$

Then `\(\deg(Q^\mathcal{G}) \leq n-1\)` and for all `\(i=1,2,\cdots n\)`, 

$$ Q^\mathcal{G}(\lambda_i) = \sum_{k=1}^n \mathcal{G}(\lambda_k) \ell_k(\lambda_i) = \sum_{k=1}^n \mathcal{G}(\lambda_k) \delta_{ik} =  \mathcal{G}(\lambda_i)$$

Now consider `\(\mathcal{G}^\prime = \mathcal{G}-Q^\mathcal{G} \in \mathbb{P}^{n-1}(\mathcal{D})\)`, then

`$$\mathcal{G}^\prime(\lambda_k) = g(\lambda_k) -Q^g(\lambda_k)  = 0$$`

So `\(\{\lambda_j\}_1^n \subset \mathbb{R}\)` are roots of `\(g-Q^g\)`. By the Fundamental Theorem of Algebra, `\(g-Q^g\)` must be the zero polynomial. 

Thus, `\(\mathcal{G}=Q^\mathcal{G}\)`.


Ultimately,

$$
`\begin{aligned}
& \int_{\mathcal{D} }\mathcal{G}(y) f_Y(y) \,dy \\
&= \int_{\mathcal{D} }Q^\mathcal{G}(y) f_Y(y) \,dy \\
&= \int_{\mathcal{D} } \left(\sum_{k=1}^n \mathcal{G}(\lambda_k) \ell_k(\lambda_i)\right) f_Y(y) \,dy \\
&=  \sum_{k=1}^n \mathcal{G}(\lambda_k) \left(\int_{\mathcal{D} }  \ell_k(y) f_Y(y) \,dy  \right)
\end{aligned}`
$$

Where we define 

$$
\nu_k = \int_{\mathcal{D} }  \ell_k(y) f_Y(y) \,dy  
$$

If `\(\mathcal{D}=[a,b]\)`, then `\(\nu_k\)` has a stable and easy formula.



The exactness is in fact up to `\(\deg(2n-1)\)`. 

If `\(\mathcal{G}\in\mathbb{P}^{2n-1}(\mathcal{D})\)`, there exists `\(q,r \in\mathbb{P}^{n-1}(\mathcal{D})\)` s.t. 

$$  \mathcal{G} = \pi_nq  + r$$

Then `\(r\in \in\mathbb{P}^{n-1}(\mathcal{D})\)`

$$
r(\lambda_k) =  \mathcal{G}(\lambda_k) - \pi_n(\lambda_k) q(\lambda_k) = \mathcal{G}(\lambda_k)
$$

And thus `\(r=Q^\mathcal{G}\)` through a similar argument as before. 

$$
`\begin{aligned}
& \int_{\mathcal{D} }\mathcal{G}(y) f_Y(y) \,dy \\
&= \int_{\mathcal{D} }(\pi_nQ^\mathcal{G}+r)(y) f_Y(y) \,dy \\
&= \int_{\mathcal{D} }\pi_nQ^\mathcal{G}(y) f_Y(y) \,dy + \int_{\mathcal{D} }r(y) f_Y(y) \,dy \\
&= \int_{\mathcal{D} } \langle\pi_n,Q\rangle + \int_{\mathcal{D} }r(y) f_Y(y) \,dy \\
&=\int_{\mathcal{D} }r(y) f_Y(y) \,dy \quad\quad\left(\langle\pi_n,Q\rangle= 0\quad \text{due to orthogonality}\right)\\
&= \sum_{k=1}^n \mathcal{G}(\lambda_k)\nu_k
\end{aligned}`
$$

Newton-Cotes Quadrature:

+ `\(\mathcal{D}=[a,b]\)`

+ `\(\lambda_k=a+k\Delta x\)` (equally spaced)

+ Don't do it. 

Now that

$$
 \int_{\mathcal{D}}\mathcal{G}(y) f_Y(y) \,dy = \sum_{k=1}^n \mathcal{G}(\lambda_k)\nu_k
$$

Statement: If `\(m\geq 2n\)`, `\(\exists \mathcal{G}\in \mathbb{P}^m(\mathcal{D})\)` s.t.

$$ I \neq \sum_{k=1}^n \nu_k \mathcal{G} (\lambda_k)$$

Proof: Let `\(\mathcal{G} = \pi_n^2\)`. Then `\(\deg(\mathcal{G}) = 2n\)`

Thus

$$ I = \int_{\mathcal{D}} \pi^2_n(s) f(s) \,ds = \langle \pi_n, \pi_n  \rangle_f >0$$

Now consider 

$$\sum_{k=1}^n\nu_k \mathcal{G} (\lambda_k) = \sum_{k=1}^n\nu_k [\pi_n (\lambda_k)]^2 = 0 $$
This is a proof by contradiction. 

Now consider `\(\mathcal{G} = \pi_nq+r\)`.

$$ \int_{\mathcal{D}} \mathcal{G}(s) f(s) \,ds = \int_{\mathcal{D}} \pi_n(s)q(s) f(s) \,ds +\int_{\mathcal{D}} r(s) f(s) \,ds $$.

Note that `\(\deg(r) <n\)`. Thus, 

$$\int_{\mathcal{D}} r(s) f(s) \,ds = \sum_{k=1}^n \nu_kr(\lambda_k) $$

So if `\(\deg(\mathcal{G})<2n\)`, `\(\deg(q)<n\)`, and the equation holds due to orthogonality. 

Consider the interpolated polynomial

$$Q^\mathcal{G}(x) = \sum_{k=1}^n \mathcal{G}(\lambda_k) \ell_k(x) $$

GQ is exact of `\(\deg(2n-1)\)`.  Large `\(n\)` would get better approximation. 

Suppose `\(g\)` is some numerically defined function that takes a long time to evaluate for any `\(g(\lambda_k)\)`.

Calculating 

`$$\sum_{k=1}^n \nu_k \mathcal{G}(\lambda_k)$$`

, while certainly more efficient to trapezoids, is still slow. One might want to "reuse" quadrature points:

If `\(\lambda_k^{(n)}\)` is a root of `\(\pi_n\)`, is `\(\lambda_k^{(n)}\)` also a root of `\(\pi_{m}\)` if `\(m>n\)`? Unfortunately, no for GQ.

Suppose there exists `\(q\in\mathbb{P}^n(\mathcal{D})\)`, such that `\(\pi_{n+1}=q(x)(x-\lambda_k^{(n)})\)`


Thm: Let `\(\{\lambda_i^{(n)}\}_1^n\)` be the roots of `\(\pi_n\)`. WLOG let 

`$$\lambda_1^{(n)}<\lambda_2^{(n)}<\lambda_3^{(n)}<\cdots<\lambda_n^{(n)}$$`

Then for `\(n>2\)`, every consecutive pair `\(\lambda_k^{(n)}\)` and `\(\lambda_{k+1}^{(n)}\)`, there is at least one root of `\(\pi_m\)` for any `\(m>n\)`.

That is for any `\(\pi_m\)` s.t. `\(m>n\)`, `\(\exists \lambda_i^{(m)} \in \left(\lambda_k^{(n)},\lambda_{k+1}^{(n)}\right)\)`

Pf: Let `\(a=\lambda_k^{(n)}\)` and `\(b=\lambda_{k+1}^{(n)}\)`,

Define

$$ g(x)=c^2 (x-a)(x-b) \prod_{j\neq k,j\neq k+1} \left(\frac{x-\lambda_j}{\lambda_k-\lambda_j}\right)^2$$

Incidentally, 

`$$\pi_n = c(x-\lambda_1^{(n)})(x-\lambda_2^{(n)})\cdots(x-a)(x-b)\cdots (x-\lambda_n^{(n)})$$`

Then `\(g(x)\)` is exact given GQ. But also, `\(m\)`-point GQ is exact for any `\(m>n\)`. Thus,
$$
`\begin{aligned}
\int_{\mathcal{D}} g(s)f(s) \,ds &= \int_{\mathcal{D}} c^2 (x-a)(x-b) \prod_{j\neq k,j\neq k+1} \left(\frac{x-\lambda_j}{\lambda_k-\lambda_j}\right)^2f(s) \,ds \\
&= \int_{\mathcal{D}} \left [ c (x-a)(x-b) \prod_{j\neq k,j\neq k+1} \left(\frac{x-\lambda_j}{\lambda_k-\lambda_j}\right) \right ]\cdot \left [ c \prod_{j\neq k,j\neq k+1} \left(\frac{x-\lambda_j}{\lambda_k-\lambda_j}\right) \right ] f(s) \,ds \\
&= \int_{\mathcal{D}}\pi_n\cdot \left [ c \prod_{j\neq k,j\neq k+1} \left(\frac{x-\lambda_j}{\lambda_k-\lambda_j}\right) \right ] f(s) \,ds \\
&= 0
\end{aligned}`
$$

where the last equality holds due to orthogonality. However, suppose for contradiction that `\((a,b)\)` has no roots for some `\(\pi_m\)` s.t. `\(m>n\)`: 

For some `\(x \notin(a,b)\)`, then either

+ `\(x\leq a<b\)`:
  `\(x-a\leq 0\)`, `\(x-b<0\)`. Thus `\((x-a)(x-b)\geq 0\)` .

+ `\(a<b\leq x\)`
  `\(x-b\geq 0\)`, `\(x-a>0\)`. Thus `\((x-a)(x-b)\geq 0\)` .
  
So `\(g(x)>0\)` almost everywhere with exception that `\(g(\lambda_k^{(n)})\)`. Recall that a property holds almost everywhere if it is only violate at a set with measure zero. 

Now let `\(\lambda_1^{(m)} < \lambda_2^{(m)} < \cdots < \lambda_m^{(m)}\)` be the roots of `\(\pi_m\)`. Since `\(m>n\)`, ther eis at least one root of `\(\pi_m\)` that is not a root of `\(\pi_n\)`, denoted as `\(\lambda_{i*}^{(m)}\)`. 

Thus

$$
`\begin{aligned}
\int_{\mathcal{D}} g(s) f(s) \,ds &= \sum_{k=1}^m \nu_k g(\lambda^{(m)}_{k}) \\
&=\nu_{i*}g(\lambda^{(m)}_{i*}) + \sum_{k=1,k\neq i*}^m \nu_k g(\lambda^{(m)}_{k}) \\
\end{aligned}`
$$

which is strictly positive due to `\(\nu_{i*}g(\lambda^{(m)}_{i*})>0\)`. However, exactness implies that 

`$$\int_{\mathcal{D}} g(s) f(s) \,ds =0$$`



## Stochastic Process

Let $y(x,t;\xi) \in L^2_c(\mathbb{R};f) $ for all `\((x,t)\in \mathbb{R}_x\times\mathbb{R}_t \geq 0\)`.

Then we can write the polynomial chaos expansion (PCE) when `\(\xi\sim \mathcal{N}(0,1)\)`

$$y(x,t;\xi) = \sum_{k=0}^\infty y_k(x,t) H_k(\xi) $$

where `\(y_k:\mathbb{R}_x\times\mathbb{R}_t \to \mathbb{R}\)`, and `\(\{H_k\}\)` are Hermite orthogonal polynomials wrt `\(\xi\sim \mathcal{N}(0,1)\)`.This inevitably involves sampling. 


Example: Solving a PDE with uncertain initial condition (inviscid Burger's Equation):

$$ \frac{\partial y}{\partial t} + y \frac{\partial y}{\partial x}  = 0$$

We suspect the form to be `\(y(x,t=0;\xi) = \xi \sin(x)\)`

The general PDE solution is thus a stochastic process $y(x,t;\xi) $ for all `\((x,t)\in \mathbb{R}_x\times\mathbb{R}_t \geq 0\)`.

We introduce the truncated PCE to avoid sampling: 

$$y^{(M)}(x,t;\xi) = \sum_{k=0}^M y_k(x,t) H_k(\xi) $$

By converging properties, 

$$ \lim_{M\to\infty} y^{(M)} \to y$$

Substitute `\(y^{(M)}\)` in to the original PDE:

$$
`\begin{aligned}
&\frac{\partial}{\partial t} \left[\sum_{i=0}^M y_i(x,t) H_i(\xi)\right] + \left(\sum_{i=0}^M y_i(x,t) H_i(\xi)\right) \frac{\partial}{\partial x} \left[\sum_{i=0}^M y_i(x,t) H_i(\xi)\right] \\ 
&= \sum_{i=0}^M \frac{\partial y_i(x,t)}{\partial t} H_i(\xi) + \sum_{i=0}^M\sum_{j=0}^M y_i(x,t)\frac{\partial y_j(x,t)}{\partial t} H_i(\xi)H_j(\xi) \\
&=0
\end{aligned}`
$$

Multiply through by `\(H_k(\xi)\)` for some fixed `\(k\in[0,M]\)`:

$$
`\begin{aligned}
\sum_{i=0}^M \frac{\partial y_k(x,t)}{\partial t}H_i(\xi) H_k(\xi) + \sum_{i=0}^M\sum_{j=0}^M y_i(x,t)\frac{\partial y_j(x,t)}{\partial t} H_i(\xi)H_j(\xi) H_k(\xi) &= 0  \\
\sum_{i=0}^M \frac{\partial y_k(x,t)}{\partial t}\int_{\mathbb{R}} H_i(\eta) H_k(\eta) f(\eta) \,d\eta + \sum_{i=0}^M\sum_{j=0}^M y_i(x,t)\frac{\partial y_j(x,t)}{\partial t} \int_{\mathbb{R}}  H_i(\eta)H_j(\eta) H_k(\eta) \,d\eta&= 0  \\

\sum_{i=0}^M \frac{\partial y_k(x,t)}{\partial t}\langle H_i,H_k \rangle_f + \sum_{i=0}^M\sum_{j=0}^M y_i(x,t)\frac{\partial y_j(x,t)}{\partial t} \langle H_iH_j,H_k \rangle_f&= 0  \\
\end{aligned}`
$$

For Hermite `\(\{H_k\}\)`, `\(\langle H_k,H_k \rangle_f = k!\)` and 


$$
\langle H_iH_j,H_k \rangle_f =
`\begin{cases}
\frac{i!j!k!}{(s-i)!(s-j)！(s-k)!}, \text{ where } s=\frac{i+j+k}{2}, \max(i,j,k)<s \\
0, \text{ otherwise,}
\end{cases}`
$$


Thus the equation above simplifies to


$$
`\begin{aligned}
\frac{\partial y_k(x,t)}{\partial t} k! + \sum_{i=0}^M\sum_{j=0}^M y_i(x,t)\frac{\partial y_j(x,t)}{\partial t} \langle H_iH_j,H_k \rangle_f&= 0
\end{aligned}`
$$

+ This becomes a deterministic system of PDEs with size `\(M+1\)`. 

+ A solution `\(\{y_k\}_0^M\)`, you can then plug your coefficient functions into your solution approximation

`$$y^M = \sum_{k=0}^M y_k(x,t) H_k(\xi) \approx y$$`


Example: `\(M=1\)`,

$$
`\begin{cases}
\frac{\partial y_0}{\partial t} + y_0 \frac{\partial y_0}{\partial t} + y_1 \frac{\partial y_1}{\partial t} = 0 \\

\frac{\partial y_1}{\partial t} + y_0 \frac{\partial y_0}{\partial t} + y_1 \frac{\partial y_0}{\partial t} = 0 \\
\end{cases}`
$$


2 partial differential equations, 2 unknowns. 

Can numerically solve for both of them via scipy.ode

Now consider the example of the heat diffusion question:

+ `\(t\in\mathbb{R}^+\)` time

+ `\(x\in [0,L]\)` position

+ PDE solution

+ `\(u(x,y;\xi)\)` is the temperature at a given point `\(x\)` and time `\(t\)`, with `\(\xi\sim \mathcal{N}(0,1)\)`

Deterministic version:

`$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$`

with initial value 

+ `\(u(x,0)=\sin(x)\)`

+ `\(u(0,t)=u(L,t)\)`

Stochastic Version:

`$$\frac{\partial u}{\partial t} = \xi \frac{\partial^2 u}{\partial x^2}$$`

with initial value and constraint

+ `\(u(x,0)=\sin(x)\)`

+ `\(u(0,t)=u(L,t)\)`

+ `\(\xi\sim \mathcal{N}(0,1)\)`

Strategy: Galerkin Projection

+ Substitute a truncated PCE `\(u^{(M)}\)` for the true solution of the original PDE, where

`$$u^{(M)} = \sum_{k=0}^M u_k(x,t)H_k(\xi)$$`

+ Multiply through by an `\(H_k(\xi)\)` for some fixed but arbitrary `\(k\in \{0,\cdots,M\}\)`.

Integrate with respect to `\(\xi\)`, and (ideally) use orthogonality to simplify

`$$\sum_{k=0}^M \frac{\partial u_k(x,t)}{\partial t} H_k(\xi) = \xi \sum_{k=0}^M \frac{\partial ^2 u_k(x,t)}{\partial x^2} H_k(\xi)$$`

`$$\sum_{k=0}^M \frac{\partial u_k(x,t)}{\partial t} \int_{\mathbb{R}} H_i(\xi)H_k(\xi) \,d\xi=  \sum_{k=0}^M \frac{\partial ^2 u_k(x,t)}{\partial x^2} \int_{\mathbb{R}} H_i(\xi)H_k(\xi) \xi \,d\xi$$`

Recall that `$$\int_{\mathbb{R}} H_i(\xi)H_k(\xi) \,d\xi = \delta_{ik}k!$$`

Note that `\(\xi = H_1(\xi)\)`. Thus the equation simplifies to 

`$$\sum_{k=0}^M \frac{\partial u_k(x,t)}{\partial t} \delta_{ik}k!=  \sum_{k=0}^M \frac{\partial ^2 u_k(x,t)}{\partial x^2} \langle H_i(\xi),H_k(\xi)H_1(\xi)  \rangle$$`

with initial conditions

+ `\(u(x,0;\xi)=\sin(x)\)`

+ `\(u^{(M)}(x,0;\xi) = \displaystyle \sum_{k=0}^M u_k(x,0) H_k(\xi)\)`

`$$\sum_{k=0}^M u_k(x,0) \int_{\mathbb{R}} H_i(\xi)H_k(\xi) \,d\xi = \left[\int_{\mathbb{R}}H_i(\xi) \,d\xi\right]\sin(x) = \delta_{i,0}\sin(x)$$`

Thus `\(u_i(x,0) = \frac{1}{i!}\delta_{i,0}\sin(x)=\sin(x)\mathbf{1}_{i==0}\)`

For `\(i=0,1,\cdots,M\)`, the equations are deterministic, and thus easy to solve.




But what if our initial equation was `$$\frac{\partial u}{\partial t} = P^n(\xi) \frac{\partial^2 u}{\partial x^2}$$` where `\(P^n\)` is a polynomial of `\(n\)` degrees instead of `$$\frac{\partial u}{\partial t} = \xi \frac{\partial^2 u}{\partial x^2}$$`

Recall that `\(\mathbb{P}^n(\mathbb{R})=\operatorname{span}\{\pi_k\}_{0}^{n}\)`

For `\(\pi_k=H_k\)`, any polynomials of degree at most `\(n\)` in the variable `\(\xi\)`, namely any `$$p(\xi) = \sum_{j=0}^n c_j \xi^j$$` can be rewritten as `$$p(\xi) = \sum_{i=0}^n d_i H_i(\xi)$$`

### Hermite Polynomials:

Hermite Polynomials in `\(d\)` dimensions for `\(\Xi\sim\mathcal{N}(0_n,I_n)\)`,

That is, 

`$$\Xi = (\xi_1,\xi_2,\cdots , \xi_n)^T \sim \mathcal{N}(\vec\mu,\Sigma)$$`

where 

`$$\xi_i\sim\mathcal{N}(\mu_i,\Sigma_{ii})$$`

and

`$$\operatorname{cov}(\xi_i,\xi)j =\Sigma_{ij}$$`

`$$H_{\vec{\alpha}}^{(d)}(\Xi) = \frac{(-1)^{\|\vec{\alpha}\|_1}}{f(\Xi)} \partial ^{\vec{\alpha}}(f(\Xi))$$`

where `\(\vec{\alpha}\in \mathbb{N}_0^n\)` is a multi-index s.t.

+ `$$\|\vec{\alpha}\| = \sum_{i=1}^n \alpha_i$$`

+ `$$\partial^{\vec{\alpha}}= \partial^{\alpha_1} \partial^{\alpha^2}\cdots \partial^{\alpha_n}$$`

Operations:

+ `\(\alpha+\beta = (\alpha_1+\beta_1, \cdots, \alpha_n+\beta_n)\)`

+ `\(\alpha! = \prod\limits_{i=1}^n a_i!\)`

+ `$$\frac{\partial^{\vec{\alpha}}}{\partial x^{\vec{\alpha}}} = \frac{\partial^\alpha_1 }{\partial x_1^{\alpha_1} } \cdots \frac{\partial^\alpha_n }{\partial x_n^{\alpha_n} }$$`


Thus define the multivariate Hermite Polynomials as:

`$$H_{\vec{\alpha}}(\vec{x}) = \delta_{0,\|\alpha\|_1} + \frac{(-1)^{\|\vec{\alpha}\|_1}}{f(\vec{x})}\frac{\partial^{\vec{\alpha}}}{\partial x^{\vec{\alpha}}} [f(\vec{x})] (1-\delta_{0,\|\alpha\|_1})$$`

Now consider

$$
`\begin{aligned}
&\frac{(-1)^{\|\vec{\alpha}\|_1}}{f(\vec{x})}\frac{\partial^{\vec{\alpha}}}{\partial x^{\vec{\alpha}}} [f(\vec{x})] \\ 
&=\prod_{i=1}^n \left(\frac{(-1)^{\alpha_i}}{f_i(x_i)}\frac{\partial^{\alpha_i}}{ \partial x_i^{a_i}} f_i(x_i)\right) \\

&= \prod_{i=1}^n H_{\alpha_i(x_i)}
\end{aligned}`
$$

Now consider `\(H_\vec \alpha (\vec x)\)` and `\(H_\vec \beta (\vec x)\)`. Do we have `\(\langle H_\vec \alpha, H_\vec  \beta\rangle = 0\)` when `\(\alpha\neq\beta\)`?

By Fubini's Theorem,

`$$\int_{\mathbb{R}^n} H_\vec \alpha (\vec x)H_\vec \beta (\vec x) f(\vec x) \,d\vec x = \int_{\mathbb{R}}\int_{\mathbb{R}}\cdots\int_{\mathbb{R}} H_\vec \alpha (\vec x)H_\vec \beta (\vec x) f(\vec x) \,dx_1 \,dx_2 \cdots \,dx_n$$`

Now consider

$$
`\begin{aligned}
&\int_{\mathbb{R}}\int_{\mathbb{R}}\cdots\int_{\mathbb{R}} \  \prod_{i=1}^n H_{\alpha_i(x_i)}H_{\beta_i(x_i)} f(x_i)\,dx_i \\
&= \int_{\mathbb{R}}\int_{\mathbb{R}}\cdots\int_{\mathbb{R}}  \prod_{i=1}^n H_{\alpha_i(x_i)}H_{\beta_i(x_i)}  f(x_i)\,dx_i \\
&=\prod_{i=1}^n \int_{\mathbb{R}}   H_{\alpha_i(x_i)}H_{\beta_i(x_i)}  f(x_i)\,dx_i \\
&=\prod_{i=1}^n \int_{\mathbb{R}}   H_{\alpha_i(x_i)}H_{\beta_i(x_i)}  \,d \xi_i \\
&=\prod_{i=1}^n  \langle H_{\alpha_i(x_i)}H_{\beta_i(x_i)}  \rangle_{\xi_i}  = 0 \\
\end{aligned}`
$$


For `\(u(x,t;\Xi)\)`, a stochastic process indexed by `\((x,t)\in \mathbb{R}_x\times\mathbb{R}_t\)` with `\(\Xi\sim \mathcal{N}(0_n,I_n)\)`, if `\(\operatorname{Cov}_\xi (u(x,t;\Xi))<\infty\)` then we can write

`$$\sum_{\alpha\in\mathbb{N}_0^n} u_\alpha(x,y) H_{\vec \alpha} (\Xi) = u(x,t;\Xi)$$` 

### Simple Finite Difference of solving PDE:

Consider the pde `$$u^{\prime\prime}=g$$`, with `\(0\leq x \leq 1\)` with boundary conditions `\(u(0)=\alpha,u(1)=\beta\)`. 

+ Put uniform grid on domain,

+ `\(x_j = jh, j =0,1,\cdots,\frac 1n\)`.

+ `\(u_j := u(x_j)\)`

Locally, by Taylor expansion,

`$$u(x+h) = u(x) +u^\prime(x)h+u^{\prime\prime}$$`

For small `\(h\)`, 

`$$u_1^\prime (x) \approx \frac{u(x+h)-u(x)}{h}$$`

`$$u_2^\prime (x) \approx \frac{u(x)-u(x-h)}{h}$$`

`$$u_3^\prime (x) \approx \frac{u(x+h)-u(x-h)}{2h}$$`

The first order derivative of the first order derivative is the second order derivative:

`$$u^{\prime\prime}(x)\approx \frac{u^{\prime}(x)-u^{\prime}(x-h)}{h} \approx  \frac{\frac{u(x+h)-u(x)}{h} -\frac{u(x)-u(x-h)}{h} }{h}  = \frac{u(x+h)+u(x-h)-2u(x)}{2h^2}$$`

or, under the notation of `\(u_j\)`,

`$$\frac{u_{j+1}+u_{j-1}-2u_{j}}{2h^2}$$`

With Constraints and  Boundary Values

+ `\(j=1,2,\cdots,m-1\)`
+ `\(u_0=u(x_0)=\alpha\)`
+ `\(u_m=u(x_m)=\beta\)`

+ Other Methods: 

  + Runge-Kutla Methods
  Runge-Kutta methods are a family of numerical methods used to solve ordinary differential equations. They are based on the idea of approximating the solution of an initial value problem by a weighted sum of function values at different points. The most commonly used member of this family is the fourth-order Runge-Kutta method, which has the following formula:
  
$$
`\begin{aligned}
k_1 &= f(t_n, y_n) \\
k_2 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1) \\
k_3 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_2) \\
k_4 &= f(t_n + h, y_n + h k_3) \\
y_{n+1} &= y_n + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}`
$$

where `\(f\)` is the function defining the differential equation, `\(t_n\)` and `\(y_n\)` are the values of the independent and dependent variables at time step `\(n\)`, `\(h\)` is the step size, and `\(k_1\)`, `\(k_2\)`, `\(k_3\)`, and `\(k_4\)` are intermediate variables calculated at different points in the interval `\([t_n, t_{n+1}]\)`.

This method is called "fourth-order" because the error in the numerical solution is of order `\(O(h^4)\)`, where `\(h\)` is the step size. It is widely used due to its simplicity, accuracy, and efficiency.

+ so-called "shooting problems"

### Viscous Burger's with Stochastic Conditions:

$$ \frac{\partial u }{\partial t} + u\frac{\partial u}{\partial x} = (1+\xi_1^2)$$

with `\(u(x,t,\Xi) = \xi_1+\xi_2\sin(x)\)`

Assuming that `\(\operatorname{Var}[u(x,t,\Xi)\)` is finite. 

Multiple PCE:

`$$u^{(M)}(x,t;\Xi) = \sum_{|\vec{\alpha}|\leq M, \alpha=\{1,2\}} u_\alpha(x,t)H_\alpha(\Xi)$$`


Substitute that into the original PDE and get:

$$\frac{\partial}{\partial t}\left[\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t)H_\alpha(\Xi)\right] + \left[\sum_{|\vec{\alpha}|\leq M}\sum_{|\vec{\beta}|\leq M}u_\alpha(x,t)\frac{\partial u_\beta(x,t)}{\partial t}H_\alpha(\Xi)H_\beta(\Xi)\right] \\ \, =(1+\xi_1^2) \frac{\partial^2}{\partial x^2}\left[\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t)H_\alpha(\Xi)\right]  $$


Multiply by `\(H_\gamma(\Xi)\)`

$$\frac{\partial}{\partial t}\left[\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t)H_\alpha(\Xi)H_\gamma(\Xi)\right] + \left[\sum_{|\vec{\alpha}|\leq M}\sum_{|\vec{\beta}|\leq M}u_\alpha(x,t)\frac{\partial u_\beta(x,t)}{\partial t}H_\alpha(\Xi)H_\beta(\Xi)H_\gamma(\Xi)\right] \\ \, =(1+\xi_1^2) \frac{\partial^2}{\partial x^2}\left[\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t)H_\alpha(\Xi)H_\gamma(\Xi)\right]  $$

Now consider

$$\sum_{|\vec{\alpha}|\leq M}\frac{\partial}{\partial t}\iint_{\mathbb{R}^2}\left[u_\alpha(x,t)H_\alpha(\Xi)H_\gamma(\Xi)\right] \,d\Xi \\ + \sum_{|\vec{\alpha}|\leq M}\sum_{|\vec{\beta}|\leq M}u_\alpha(x,t)\frac{\partial u_\beta(x,t)}{\partial t} \iint_{\mathbb{R}^2} \left[H_\alpha(\Xi)H_\beta(\Xi)H_\gamma(\Xi)\right]  \,d\Xi \\ \, = \frac{\partial^2}{\partial x^2}\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t) \iint_{\mathbb{R}^2}\left[(1+\xi_1^2)H_\alpha(\Xi)H_\gamma(\Xi)\right] \,d\Xi  $$

Recall that integrals can calculate inner products. Namely,

`$$\langle  H_{\vec{\alpha}},H_{\vec{\gamma}}  \rangle_{\Xi} := \iint_{\mathbb{R}^2}\left[u_\alpha(x,t)H_\alpha(\Xi)H_\gamma(\Xi)\right] \,d\Xi$$`

`$$\langle  H_{\vec{\alpha}}H_{\vec{\beta}}, H_{\vec{\gamma}}  \rangle_{\Xi} := \iint_{\mathbb{R}^2}\left[u_\alpha(x,t)H_\alpha(\Xi)H_\beta(\Xi)H_\gamma(\Xi)\right] \,d\Xi$$`

And the original equation becomes

$$\sum_{|\vec{\alpha}|\leq M}\frac{\partial u_\alpha(x,t)}{\partial t}\langle  H_{\vec{\alpha}},H_{\vec{\gamma}}  \rangle_{\Xi} + \sum_{|\vec{\alpha}|\leq M}\sum_{|\vec{\beta}|\leq M}u_\alpha(x,t)\frac{\partial u_\beta(x,t)}{\partial t} \langle  H_{\vec{\alpha}}H_{\vec{\beta}}, H_{\vec{\gamma}}  \rangle_{\Xi}  \\ \, = \frac{\partial^2}{\partial x^2}\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t) \left[\langle  H_{\vec{\alpha}}H_{(2,0)}, H_{\vec{\gamma}}  \rangle_{\Xi} +2 \langle  H_{\vec{\alpha}},H_{\vec{\gamma}}  \rangle_{\Xi} \right]  $$

Which, due to the orthogonality of Hermite Polynomials previously discussed, simplifies to 

$$\frac{\partial u_\alpha(x,t)}{\partial t}\langle  \vec{\gamma}! + \sum_{|\vec{\alpha}|\leq M}\sum_{|\vec{\beta}|\leq M}u_\alpha(x,t)\frac{\partial u_\beta(x,t)}{\partial t} \langle  H_{\vec{\alpha}}H_{\vec{\beta}}, H_{\vec{\gamma}}  \rangle_{\Xi}  \\ \, = \frac{\partial^2}{\partial x^2}\sum_{|\vec{\alpha}|\leq M}u_\alpha(x,t) \left[\langle  H_{\vec{\alpha}}H_{(2,0)}, H_{\vec{\gamma}}  \rangle_{\Xi} +2 \langle  H_{\vec{\alpha}},H_{\vec{\gamma}}  \rangle_{\Xi} \right]  $$
