# Apuntes Intro al Machine Learning

Please keep in mind that these are my personal notes with possible mistakes, written in a way that helps me understand the content and organize my knowledge. Most of this notes are copy directly from the ppts of the professor Carlos Valles.

## 01 What is Machine Learning

### What is Machine Learning?

Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed. In an abstract task, we need to provide the machine with some experience related to that task, and a metric to measure its ability to perform the task based on the experience.

### Learning Paradigms

In general, there are three principal paradigms in Machine Learning: 
- Supervised learning: Use labels in data, usefull in, for example, classification problems. 
- Unsupervised learning: Do not use labels in data, useful in, for example, to find structures in data and anomalies.
- Reinforced learning: The algorithm must perform a certain goal. It doesn’t know how
close to its goal is.

## 02 Supervised Learning

### Introduction to learning from examples

Just and introduction to the basic: We have features and a target with data, we want to predict the feature $y$ with the data of the other features $\textbf{x}$.

### Preliminary definitions

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

### Lineal model
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

### How to select a model in practice
-