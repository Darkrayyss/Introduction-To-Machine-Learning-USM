# Feature Selection / Extraction
## Feature selection
> Buscamos elejir que features usar al momento de aplicar cualquier modelo, para sacar información que no sea realmente relevante, quitar correlaciones entre los features, simplificar el modelo para evitar overfitting, etc, dados $I$ features, hay $2^I$ forms to select the features. We do not have the real solution but we have heuristics.
- In our multilineal model, assuming $\epsilon$ follow a $N(0,\sigma^2)$, then we know the distribution of $\hat\beta$, so running hypothesis test over $\beta_i=0$ using the standarized coefficient or $Z$-score:
$$
z_i=\frac{\hat\beta_i}{\hat \sigma\sqrt{v_i}}~t_{M-I-1}
$$
- where $v_i$ is the $i$th diagonal ement of $(X^TX)^{-1}$.
> Desde ahora me dedicare a comentar más que copiar...
> The idea is extract the coefficients close to 0, but to this we use the data of the train set mediante estimations, but we do not like to relial the train data, that conduce to overfitting, another problem is that we need to extract one to one, because extrar a $x_i$ (related to $\beta_i=0$) change every estimation.
> Testing to differentes...