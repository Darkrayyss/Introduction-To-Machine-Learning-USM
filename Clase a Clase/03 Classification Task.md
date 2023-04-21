# 03 Classification Task
> I need to go faster, soo.

## Motivation
- In binary classification we most focus in use $y\in \{-1,1\}$ more than $y\in \{0,1\}$ because of margin function (generaly).

## Other perfomance metrics for classification
We need better tools to measure the performace of a classification algorithm, so we introduce the notion of True positive, False Negative, False Positive and True Negative.
- Confusion matrix is deduced from above.
- Recall: Is the radio of total number of correctly classified positive examples divide the total number of positive examples:
$$\text{Recall}=\frac{TP}{TP+FN}$$
> In this, we focus on predict correctly the positive examples.
- Precision:
$$\text{Precision}=\frac{TP}{TP+FP}$$
> Is a measure of how reliable the positives examples are.
- F1-score: Is the harmonic mean of precision and recall:
$$F1=\frac{2*\text{Recall}*\text{Precision}}{\text{Recall}+\text{Precision}}$$

## Generative models (Bayes)
- In generative models we try to model $p(y\vert x)$ by using $p(x\vert y)$ and $p(y)$ the margin distributions or class priors.
- Strong assumption: We can directly model $p(x\vert y)$ and $p(y)$ from train data.

> Nota importante: En la sección anterior vimos que dado un modelo de aprendizaje, tenemos una función que toma valores en los posibles espacios de set de entrenamiento y nos entrega una hipótesis en función de esto, la selección del modelo de aprendizaje es la que nos define el espacio $\mathcal{H}$ de donde proviene el error de aproximación, ahora estamos viendo modelos generativos, los cuales son modelos de aprendizaje basados en el teorema de Bayes.

- Actually for supervised tasks discrminative models tend to be more efficient in practice (Model directly $p(y\vert \textbf{x})$)
- Is easy to check that to maximizing $p(y\vert x)$ is enough to maximize $p(x\vert y)p(y)$, so
$$\argmax_y p(y\vert x)=\argmax_y p(x\vert y)p(y)$$

### Lineal discriminant analysis (LDA)
- Assumptions: $x\vert y$ have a multivariate normal distribution for every possible value of $y$ and every gaussian have the same covariance matrix.
- Under this, que define the discriminant function as
$$\delta_k(x)=\log p_k +x^T\Sigma^{-1}\mu_k^T-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k$$
- Then the estimation of in which class $y$ is, is given by
$$\hat y=\argmax_k \delta_k(x)$$
- To estimate the parameters $p_k=p(y=k)$, $\mu_k$ is direct as usual. To estimate $\Sigma$, we use:
$$\hat \Sigma=\sum_{k=1}^K\sum_{y_m=k}(x_m-\hat \mu_k)(x_m-\hat \mu_k)^T/(M-K)$$

- Binary LDA: Is the same case than the preovious
- If the covariance matrices are different: QDA (Quadratic discriminant functions). We relax the assumption of equal covariance matrix, then we need to estimate more matrices, this method have better results but with high risk of overfitting and a high computacional cost.

### Naive Bayes
- We assume that features are mutual independent, then 
$$\hat p_k=\frac{1}{M}\sum_{m=1}^M I(y_m=k)$$
- and by the same way
$$\hat p(x^{(i)}\vert y=k)=\frac{1}{M_k}\sum_{m=1}^MI((y_m=k)\wedge(x_m^{(i)}=x^{(i)}))$$
- Note that we need a finite possible values of $x^{(i)}$, if this feature is continuous, we need to discrete it. A possible problem is when exists a possible value that is not in our data, then $p(x)=0$ thus $p(y=k\vert x)=0/0$, to solve that we use the Laplace smoothing: If $q$ es el number of extra possibles values, is enought to sum 1 to all numerators and sum $q$ to the denominator in the last formula.

## Discriminative models
- Lineal regression model might mask some classes

### Logistic regression
- Born as a model to solve this problem.
- It measures the relationship between the class and the input vector by estimating probabilities using a logistic function:
$$f_\beta{\textbf{x}}=g(\beta^T\textbf{x})=\frac{1}{1+e^{-\beta^T\textbf{x}}}$$
- where $g$ is the logistic or sigmoid function.
- Note that $f_\beta(\textbf{x})$ is clearly not lineal, but it is it decision boundary.
- $z=0$ create the decision threshold $0.5$.
- $g'(z)=g(z)(1-g(z))$
Obtaining $\beta$
- Assume that $P(y=1\vert \textbf{x};\beta)=f_\beta(\textbf{x})$ and $P(y=0\vert \textbf{x};\beta)=1-f_\beta(\textbf{x})$, thus $p(y\vert\textbf{x};\beta)=(f_\beta(\textbf{x}))^y(1-f_\beta{\textbf{x}})^{1-y}$.
- With that we calculate the log-likelihood to maximize it.
- Using Stochastic graidient ascent (we are scaling on the gradient $+\alpha$), we need to calculate the gradiente, where
$$\frac{\partial}{\partial \beta_i}\ell(\beta)=(f_\beta(\textbf{x})-y)\textbf{x}^{(i)}$$
- so the algorithm is based in
$$\beta^{p+1}=\beta^{p}+\alpha(f_\beta(\textbf{x})-y)\textbf{x}$$

##  Logistic regression vs LDA
- Logistic regression directly model $E[y\vert x]$ to define the decision boundary without making any assumption about $p(x\vert y)$.
- While LDA assumes that $p(x\vert y)$ follows a multivariate Gaussian, etc.
- If these assumptions are not met the performance of LDA will severely drop.
- In practice these assumptions are never correct, and often some of the components of $X$ are qualitative variables.
- Exists the idea that logistic regression is more robust than LDA.
> Datos atipicos en logistic regression son tirados a 0 por la funcion mientras que en LDA afectan gravemente a la matriz de covarianza.
- Thererfore, LDA is not robust to gross outliers.
