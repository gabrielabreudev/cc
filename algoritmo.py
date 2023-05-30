import numpy as np

def gauss_elimination(A, b):
    n = len(A)
    # Aplicar eliminação para transformar a matriz A em uma matriz triangular superior
    for i in range(n):
        # Pivôizar a linha atual
        max_row = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[max_row, i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Eliminar os coeficientes abaixo do pivô
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Resolver o sistema triangular superior
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

# Exemplo de sistema linear
A = np.array([[2, -1, 3],
              [1, 3, -2],
              [4, 2, -3]])
b = np.array([9, 8, 3])

# Resolver o sistema usando eliminação de Gauss
x = gauss_elimination(A, b)

print("Solução:")
print(x)
