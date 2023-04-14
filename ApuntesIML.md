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

## Dudas para la clase 14-03-2023

### Sobre la tarea:

- Recomienda agrupar las clases de rented bike para tener menos clases y poder usar stratify?

- Hay un día que es function day = No y function day = 1 dependiendo de la hora del día, ¿Lo eliminamos? ¿Que trato le damos?

- Cuando function day = 0, se tiene  que rented bike = 0, luego estos días sólo entregan información deterministica que dice que cuando no se trabaja, no se vende, por lo tanto al eliminar esta columna como indica la tarea, estamos cesgando el modelo con días donde las ventas son 0 sin decirle la razón (estaba cerrado el trabajo), por lo tanto pensamos en eliminar esas observaciones y luego eliminar la columna, para así sólo tener datos de valor, i.e. aquellos que nos entregan rented bike en función de las carácteristicas de ese día.

- En la tarea se pregunta por "cuando se venden más bicicletas" y para esto, se requiere hacer una comparación justa entre las carácteristicas de los días utilizando el promedio de ventas según feature, pero luego se vuelve a preguntar por "promedio de ventas según feature" lo cuál nos hace pensar que estamos entendiendo mal las primeras preguntas.

- Preguntar sobre si hacer agrupaciones de clases, por ejemplo si rented bike = 666, entonces entra en la clase  $k_7 = [650,699]$ y se agrega una columna con una variable tarjet que corresponde a las k_i. (me explayo más en unas notas en el github de la tarea)