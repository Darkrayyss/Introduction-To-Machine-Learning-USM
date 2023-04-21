# 02 Supervised Learning

## Introduction to learning from examples

Just and introduction to the basic: We have features and a target with data, we want to predict the feature $y$ with the data of the other features $\textbf{x}$.

## Preliminary definitions

- Features can be either numerical or categorical.
- Let $\mathcal{X}$ be the feature space, and $\mathcal{Y}$ the output space.
- Our goal is to obtain a mapping $f : \mathcal{X} \rightarrow \mathcal{Y}$, (commonly called the
hypothesis or learner), $\mathcal{X} \subset \mathbb{R}^n$, $\mathcal{Y} \subset \mathbb{R}$.
- Let $\mathcal{S}$ the space that spans the possible samples, drawn from an
unknown distribution $P(\textbf{x}, y)$.
- A learning algorithm is a map from the space of train sets to the
hypothesis space H of possible functional solutions



### Learning process

- If the target is continuos, we have a regression problem.
- If the tarjet can take a finite number $k$ of discrete values, we have a classification problem.  In particular if $k = 2$ the problem is called binary
classification.
- Density estimation is another supervised task.
- It goal is to approximate the probability of a desired input. Here the target y ∈ [0, 1].

### Generalization

- A main challenge is to construct an automatic method or algorithm
able to estimate future examples based on the observed phenomenon
in the train set.
- This key property of an algorithm is known as the generalization
ability
- The algorithms that memorize the train samples but have poor
predictive performance with unknown examples, this undesirable
problem is well-known as overfitting.

### Loss function

- The quality of the algorithm $\mathcal{A}$ is measured by the loss function given
by $\ell: \mathbb{R} \times \mathcal{Y} \rightarrow [0, \infty)$, which quantifies the accuracy of the observed response $f(\textbf{x})$ with respect to the true or desired response $y$.
- This function does not penalize the exact predictions, i.e., $\ell(y,f(\textbf{x}))=0$ if and only if $y=f(\textbf{x})$.
- $\ell$ is a non-negative function, hence, the hypothesis will never profit from additional good predictions.
- In regression settings we use the quadratic loss function
$$
\ell(y,f(\textbf{x}))=(y-f(\textbf{x}))^2
$$
- While in classification we have the misclassification loss function
$$
\ell(f(\textbf{x}),y)=\left\{\begin{aligned}&0 \text{ if } f(\textbf{x})=y\\&1 \text{ if } f(\textbf{x})\neq y\end{aligned}\right.
$$
- However, in binary classification problem where ($y\in\{-1,1\}$), the margin $yf(\textbf{x})$ is introduced as a quality measure.
- This quality amount leads to several loss functinos such as the \textit{hinge loss}
$$
\ell(f(\textbf{x}),y)=\max{(1-yf(\textbf{x}),0)}=\lvert 1-yf(\textbf{x}) \rvert_+
$$
- The logistic loss $\ell(f(\textbf{x}),y)=\log_2(1+e^{-yf(\textbf{x})})$
- The exponential loss $\ell(f(\textbf{x}),y)=e^{-yf(\textbf{x})}$
- Note that the square loss can be arranged as $\ell(f(\textbf{x}),y)=(1-yf(\textbf{x}))^2$ taking into account that $y^2=1$.
- And the misclassification loss can be written as $\ell(f(\textbf{x}),y)=I(yf(\textbf{x})<0)$.
- Extra: Logistic Loss and Exponential Loss can be viewed as a continuous approximation of the misclassification function.

## Lineal model
- The space of linear functions mapping from $\mathcal{X}$ to $\mathcal{Y}$ is parametric.
- We define $x^{(0)}=1$ to define the lineal model in matrix form:
$$f(\textbf{x})=\sum_{i=0}^I\beta_ix^{(i)}=\beta^T\textbf{x}$$
- where $I$ is the number of features.
> I think that $I$ correspond an a hyperparameter.
- Thus, the goal is to learn a $I+1$ dimensional vector $\beta$ that defines a hyperplane minimizing an error criterion.

### How do we select the $\beta$'s?

- We can write the quadratic loss function over the train set as:
$$ J(\beta)=\frac{1}{2}\sum_{m=1}^M(f(x_m)-y_m)^2 $$
- where $x_m$ is the vector of features of the $m$-observation.
- We need to choose $\beta$ wich minimizes $J(\beta)$
> I will not write all the deduction, so doing optimization things
$$\hat\beta = (\textbf{X}^T\textbf{X})^{-1}\textbf{X}^T\textbf{Y}$$

### Bias of the LMS Algorithm

- Assuming that the linear model is correct, i.e., $\textbf{Y}=\textbf{X}\beta+\epsilon$ for some unknown $\beta$. Furthermore, we assume that $E[\epsilon]=0$ and $Cov(\epsilon)=\sigma^2I$ (uncorrelated noise). From the least squares solution, is easy to check that $\beta_{ls}$ is an unbiased estimator of $\beta$.
- Furthermore we have $Cov(\beta_{ls})=\sigma^2(\textbf{X}^T\textbf{X})^{-1}$
- Typically the variance $\sigma^2$ is estimated as
$$\hat \sigma^2=\frac{1}{N-I-1}\sum_{m=1}^M(y_m-\hat y_m)^2$$

### Gradient descent algorithm
- Let consider the gradient descent algorithm which start with some initial $\beta$ and repeatedly perfoms:
$$
\beta_i^{p+1}=\beta_i^p-\alpha\frac{\partial}{\partial\beta_i}J(\beta).
$$
> Note that the algorithm is based in the idea that if $\beta$ is a value that minimize the error, then the derivate of the error function in $\beta$ should be equal to 0, so we can create a function of fixed point that can be represented by each $i$
- Calculating the derivative of the loss function for a single train example $(\textbf{x}_m,y_m)$:
$$\frac{\partial}{\partial \beta_i}J(\beta)=(f(\textbf{x}_m)-y_m)\textbf{x}_m^{(i)}$$
- Note that the amount of the update is proportional to the error.
#### Batch gradient descent algorithm
> In resuming, chossing $\alpha$ properly, for a complete train set we can write the iteration of Batch gradiente descent algorithm as
$$\beta_i^{p+1}=\beta_i^p-\alpha\frac{1}{M}\sum_{m=1}^M(f(\textbf{x}_m)-y_m)\textbf{x}_m^{(i)},\quad \text{for every i -obviusly-}$$
> We use all the data to do the next step, so is slow and with a high computacional cost.

#### Stochastic gradiente descent algorithm
> Exactly same except for the part that we do not use all the data por every step, we only use a single random sample to do the iteration.
- This method is called stochastic or online gradient descent.
- Usually, this technique converge faster than batch gradient descent.
- However, using a fixed value for $\alpha$ it may never converge to the minimun of $J(\beta)$, oscillating around it.
- To avoid this behavior, it is recommended to slowly decrease $\alpha$ to zero along the iterations.

## How to select a model in practice
- We will assume we have a finite set of models $\mathcal{M}=\{M_1,\dots,M_d\}$
- In case that the parameter(s) is (are) continuous we can discretize it (them).
- But choosing the parameters which minimizes the train error does not guarantee generalization.

### Hold-out cross validation
- Obtain a $S_{train}$ (general the 75% of the data) and $S_{CV}$ (validation set with the rest).
- Then train every model $M_i$ in the $S_{train}$ to obtain the hypothesis $f_i$.
- Select the hypothesis with the smallest $\hat\varepsilon_{S_{CV}}(f_i)$ on the hold out cross validation set.
> Cons: We are not using all the data in the training and we estimate the generalization error only from one cross validation set.

### $K$-fold cross validation
> The same of the Hold-out cross validation but with more steps, instead of having only test to validation, we create a partition of $S$ into $K$ disjoint, then for every model $M_i$, create the hypothesis $f_{ij}\leftarrow \mathcal{A}(S\setminus S_j)$, then compute the error $\hat\varepsilon_{S_j}(f_{ij})$ as the error over the cardinal of the set $S_j$ to obtain an ponderate error, then we calculate the error $\hat\varepsilon_{M_i}$ as the average of the preovious errors, finally pick de model $M_i$ with the lowest $\hat\varepsilon_{M_i}$. So picking $f\leftarrow \mathcal{A_i}(S)$ we obtain the final hypothesis.
- Usually $K=5$ or $K=10$ are the general options.
- If we pick $K=M$ this method is called leave-one-out cross validation.
- In classification problems, sometimes the proportion of examples of each class are unequal. In this case, we can adapt cross validation in such a way that each fold has the same proportions. Thus, all folds will be equally unbalanced. This method is called stratified $K$-fold cross validation.

## Training error
- For every hypothesis $f$ and a loss function $\ell$. We define the training error, or empirical risk, or empirical error to be
$$R_{emp}(f)=\frac{1}{M}\sum_{m=1}^M\ell(y_m,f(\textbf{x}_m))$$
- For binary classification problem, $\ell$ is the misclassificatino function, then the empirical error results the fraction of training examples that $f$ misclassifies.

### Empirical Risk Minimization (ERM)
- Consider an hypothesis $f_\textbf{w}(x)$ which is parametrized by $\textbf{w}$. We would like to find which minimizes $\hat{\textbf{w}}$:
$$\hat{\textbf{w}}=argmin_\textbf{w}R_{emp}(f_{\textbf{w}})$$
- This is called empirical risk minimization (ERM).

### Generalization  error
- We define the generalization error to be
$$R(f)=\int\int\ell(y_m,f(\textbf{x}_m))P(\textbf{x},y)\,d\textbf{x}dy$$

- In particular, for binary classification problems $\ell$ is the misclassification function.
- Hence, the probability to misclassify a new example $(\textbf{x},y)$ from the distribution $\mathcal{D}$.

### Bias-variance tradeoff
> The same analysis from stadistics. Bias is understood as the set of points that cannot be well predicted by the model and the variance measure how different the predictions along the training samples are.

> Where $E[(y-f(\textbf{x}))^2]=MSE=bias^2(f)+var(f)$
- From the machine learning point of view, this trade-off is strongly related with the complexity of the learner $f$.
- A learner with low complexity has high bias covering the training points, which can lead to underfitting.
- While, if the complexity of the learner is too high, the prediction tends to be closer (lower bias) to the training data and consequently generates overfitting.
> Low complexity => High Bias and Low Variance, High complexity => Low Bias and High Variance.
> In other words, with low complexity we cannot predice correctly the values, but the excepted prediced values are closer, instead of with high complexity, we can achuntarle brigido to the exact point, but when we fail, we do it for long????.

### Types of errors

- $f^* = \argmin_fE[\ell(f(\textbf{x}),y)]$, the hypothesis that minimize the error.
> The "perfect hypothesis" in terms of the error function.
- $f_\mathcal{H}^* = \argmin_{f\in \mathcal{H}}E[\ell(f(\textbf{x}),y)]$, the "perfect" hypothesis from our hypothesis space.
- $\hat f_n = \argmin_{f\in \mathcal{H}}\frac{1}{M}\sum_{m=1}^M\ell(f(\textbf{x}_m),y_m)$, the approximation obtained from minimize the empirical error, this is not the real obtained, this is de "perfect" restricted to our data in terms of the empirical error.
- Approximation Error: $R(f_\mathcal{H})-R(f^*)$
> An error that come from our selection of $\mathcal{H}$, if we can use $\mathcal{H}$ as the colecction of ALL possible function, this erros is 0.
- is the penalty for restricting to $\mathcal{H}$ rather than all possible functions.
- Bigger $\lvert\mathcal{H}\rvert$ mean smaller approximation error.
- Estimation Error: $R(\hat f_n)-f(f_\mathcal{H})$.
- It is produced for choosing $f$ using finite training data (empirical risk rather than true risk).
> It is produced for the limitation in data, is the error inherent from the dataset.
- With smaller $\lvert\mathcal{H}\rvert$ we expected smaller estimation error.
> Because a smaller $\lvert\mathcal{H}\rvert$ mean less possible function, then the impact of the dataset is less.

#### Excess Risk
- Definition: The excess risk is the difference between the risk of $f$ and the optimal $f^*$:
$$\text{Excess Risk}(f)=R(f)-R(f^*)$$

#### Excess Risk decomposition for ERM
- The excess risk for the ERM $\hat f_n$ can be decomposed:
$$\text{Excess Risk}(\hat f_n)=R(\hat f_n)-R(f^*)=\underbrace{(R(\hat f_n)-R(f_\mathcal{H}))}_{\text{estimation error}}+\underbrace{(R(f_\mathcal{H})-R(f^*))}_{\text{approximation error}}$$

#### Optimization Error
- In practice, we don't find the ERM $\hat f_n\in \mathcal{H}$.
- We find $\tilde f_n\in \mathcal{H}$ due to our optimizer.
> Depends of the method to minimize the function in practice, if we can find the exact solution, we find $\hat f_n$.
- Thus we define the Optimization Error: $R(\hat f_n)-R(\tilde f_n)$
- So, decomposition of the Excess Risk is:
$$\text{Excess Risk}(\tilde f_n)=\underbrace{(R(\tilde f_n)-R(\hat f_n))}_{\text{Optimization eror}}+\underbrace{(R(\hat f_n)-R(f_\mathcal{H}))}_{\text{estimation error}}+\underbrace{(R(f_\mathcal{H})-R(f^*))}_{\text{approximation error}}$$

## VC dimension
- zzz...