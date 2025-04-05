from sympy import expand, simplify, MatrixSymbol, symbols, Rational, Matrix
from sympy.core import Mul, Pow


H_symbol = MatrixSymbol("H", 3, 3)
A_symbol = MatrixSymbol("A", 3, 3)

alpha, beta, gamma, delta = symbols("alpha beta gamma delta")

M1 = H_symbol + alpha * A_symbol
M2 = beta * A_symbol
M3 = gamma * A_symbol
M4 = delta * A_symbol

def commutator(X, Y):
    return X * Y - Y * X

C12 = simplify(expand(commutator(M1, M2)))
C13 = simplify(expand(commutator(M1, M3)))
C14 = simplify(expand(commutator(M1, M4)))

C112 = simplify(expand(commutator(M1, C12)))
C113 = simplify(expand(commutator(M1, C13)))

expr = (
    M1 - Rational(1, 6) * C12 + Rational(0, 60) * C113 - Rational(1, 60) * commutator(M2, C12)
    + Rational(1, 360) * commutator(M1, C112) + Rational(1, 140) * commutator(M2, C14)
    - Rational(1, 420) * commutator(M3, C13) - Rational(1, 210) * commutator(M4, C12)
    - Rational(1, 840) * commutator(M1, commutator(M1, C14)) + Rational(1, 504) * commutator(M2, C113)
    - Rational(1, 840) * commutator(M2, commutator(M2, C12)) + Rational(1, 2520) * commutator(M3, C112)
    - Rational(1, 2520) * commutator(M1, commutator(M1, C113))
    - Rational(1, 7560) * commutator(C12, C112) + Rational(1, 2520) * commutator(M2, commutator(M1, C112))
    - Rational(1, 15120) * commutator(M1, commutator(M1, commutator(M1, C112)))
)

simplified_expr = simplify(expand(expr))
#print(simplified_expr)

k, k_z, kappa = symbols('k k_z kappa')
H = Matrix([[0, -k, 0], [k, 0, 0], [0, 0, 0]])
A = Matrix([[0, k_z, 0], [-k_z, 0, -kappa], [0, kappa, 0]])

expr_with_matrices = expr.subs({H_symbol: H, A_symbol: A})
expr_simplified = simplify(expand(expr_with_matrices))

for i in range(3):
    for j in range(3):
        print(f"Omega[{i}, {j}] = {expr_simplified[i, j]}")