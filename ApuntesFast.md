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

 <hr>

## Clase 14-03-2023

- Recuerda siempre normalizar los datos y centrarlos en 0.

### Feature selection

- Los Z-score dependen de los datos, entonces los del mundo del Machine learning no le gusta ya que no se confia de los datos. La idea subyacente es eliminar una columna, luego recalcular los Z-score y seguir.

- Z-score depende de los datos mientras que el siguiente (3) ocupa los datos de entrenamiento, es decir, la capacidad de generalizar.

- PCA es principal component analysis, ortogonaliza el espacio en orden de los vectores que expliquen la mayor varianza, el tema de PCA es que no utiliza las clases, por lo tanto nunca debe ser usado para problemas de clasificación.

## Dudas para la clase 14-04-2023

### Sobre la tarea:

- Recomienda agrupar las clases de rented bike para tener menos clases y poder usar stratify?

- Hay un día que es function day = No y function day = 1 dependiendo de la hora del día, ¿Lo eliminamos? ¿Que trato le damos?

- Cuando function day = 0, se tiene  que rented bike = 0, luego estos días sólo entregan información deterministica que dice que cuando no se trabaja, no se vende, por lo tanto al eliminar esta columna como indica la tarea, estamos cesgando el modelo con días donde las ventas son 0 sin decirle la razón (estaba cerrado el trabajo), por lo tanto pensamos en eliminar esas observaciones y luego eliminar la columna, para así sólo tener datos de valor, i.e. aquellos que nos entregan rented bike en función de las carácteristicas de ese día.

- En la tarea se pregunta por "cuando se venden más bicicletas" y para esto, se requiere hacer una comparación justa entre las carácteristicas de los días utilizando el promedio de ventas según feature, pero luego se vuelve a preguntar por "promedio de ventas según feature" lo cuál nos hace pensar que estamos entendiendo mal las primeras preguntas.

- Preguntar sobre si hacer agrupaciones de clases, por ejemplo si rented bike = 666, entonces entra en la clase  $k_7 = [650,699]$ y se agrega una columna con una variable tarjet que corresponde a las k_i. (me explayo más en unas notas en el github de la tarea)

## Dudas para la clase 21-04-2023

- En las funciones de perdida, se nos dice que la función no penaliza los valores bien predichos, pero tanto como la exponential loss y la logistic loss si lo hacen ¿Esto no es una contradicción?
- En problemas de regresión ocupamos perdida cuadratica mientras que en clasificación binaria nos apoyamos de la margin $y(f(\textbf{x}))$ ¿Como se abordan los problemas con $k$ clases?
In linear regression, the number of features $I$ is a hyperparameter?
- Batch gradient descent algorithm. In general, gradient descent can reach a local minimum, what happend for $J$ the square loss function? In the notes say "$J$ is convex function. Thus, the optimization problem has only one global optimum."
- Why in K-fold cross validation the set must to be disjoint?
- Empirical Risk Minimization is another way to choose a model?
- How to extend the logistic regression to multi classes?
