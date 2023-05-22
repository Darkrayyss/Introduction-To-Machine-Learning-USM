# Separating Hyperplanes and Kernels

## Separating Hyperplanes

- Tirar un plano entre dos clases que las divida en dos grupos de forma perfecta.

### Primal perceptron algorithm: 

- Utiliza gradiente descendente para minimizar la función que suma los margin en signo negativo (maximiza el margin). Cada iteración sobre cada dato calcula una iteración todo el rato hasta converger. 

- Principal problema, asume que se trazar una recta que separe los grupos, si no se puede, el algoritmo nunca termina. Esta carácteristica sobre los datos de poder trazar una recta que los separe, se conoce como linealmente separable.

- La frontera de decisión viene dada por el producto $f(x)= w^T\textbf{x}+\beta=0$, si la función toma un signo, el valor pertenece a un grupo, con el otro signo, pertenece al otro grupo. Esto se fundamenta en que $w^T$ es el vector normal de la recta que separa a los datos. La distancia consigno es $f(x)$ sobre la norma de la derivada evaluada en el punto.



## Kernel function

### Comentarios de clase:

- La matriz de Kernel debe cumplir dos cosas principales:

    - Simétrica y semidefinida positiva. Esto se justifica con el teorema de Mercer (pide que sean datos discretos para funcionar).
