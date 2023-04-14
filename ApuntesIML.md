# Apuntes IML:

## Clase Viernes 31 de Marzo (Online)

### Logistic Regression

Clase 03 en pdf:

Tener en cuenta que:

* Se ocupa harta mate para la optimización.

* LDA y Logistic Regression son clasificadores lineales dados que sus fronteras decisión son lineales a pesar de ser funciones no lineales.

La frontera de decisión de la logistic regresion viene dada por $g(z)=0.5$, luego la solución en $x$ de $g(\beta^Tx)=0.5$ es efectivamente un hiperplano (lineal).

Proof: The logistic regression is given by the function
$$
g(\beta^Tx)=\frac{1}{1+e^{-\beta^Tx}}
$$
 Thus $g(z)=0.5$, we have
 $$
 \frac{1}{1+e^{-\beta^Tx}}=0.5\Rightarrow e^{-\beta^Tx}=1 \Rightarrow \beta^Tx=0
 $$