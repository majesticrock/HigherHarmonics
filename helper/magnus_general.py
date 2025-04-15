from sympy import expand, simplify, symbols, Rational, Matrix

from sympy.printing.str import StrPrinter

class CustomStrPrinter(StrPrinter):
    def _print_Pow(self, expr):
        base, exp = expr.args
        # Check for integer exponents
        if exp.is_Integer and exp > 0:
            return f"pow_{self._print(base)}[{exp - 2}]" # -2 because arrays begin at 0 and we do not need to save ^1
        return super()._print_Pow(expr)

alpha1, beta1, gamma1 = symbols("coeffs_1[0] coeffs_1[1] coeffs_1[2]")
alpha2, beta2, gamma2 = symbols("coeffs_2[0] coeffs_2[1] coeffs_2[2]")
alpha3, beta3, gamma3 = symbols("coeffs_3[0] coeffs_3[1] coeffs_3[2]")
alpha4, beta4, gamma4 = symbols("coeffs_4[0] coeffs_4[1] coeffs_4[2]")


M1 = Matrix([[0, -gamma1, beta1], [gamma1, 0, -alpha1], [-beta1, alpha1, 0]])
M2 = Matrix([[0, -gamma2, beta2], [gamma2, 0, -alpha2], [-beta2, alpha2, 0]])
M3 = Matrix([[0, -gamma3, beta3], [gamma3, 0, -alpha3], [-beta3, alpha3, 0]])
M4 = Matrix([[0, -gamma4, beta4], [gamma4, 0, -alpha4], [-beta4, alpha4, 0]])

def commutator(X, Y):
    return X * Y - Y * X

C12 = simplify(expand(commutator(M1, M2)))
C13 = simplify(expand(commutator(M1, M3)))
C14 = simplify(expand(commutator(M1, M4)))

C112 = simplify(expand(commutator(M1, C12)))
C113 = simplify(expand(commutator(M1, C13)))

C23 = simplify(expand(commutator(M2, M3)))

expr = (M1 
    - Rational(1, 6) * C12 
    + Rational(0, 60) * C113 
    - Rational(1, 60) * commutator(M2, C12)
    + Rational(1, 360) * commutator(M1, C112) 
    - Rational(1, 30) * C23 
    - Rational(1, 70) * commutator(M3, M4)
    + Rational(1, 140) * commutator(M2, C14)
    - Rational(1, 210) * commutator(M2, C23)
    - Rational(1, 420) * commutator(M3, C13)
    - Rational(1, 210) * commutator(M4, C12)
    - Rational(1, 840) * commutator(M1, commutator(M1, C14))
    - Rational(1, 504) * commutator(C12, C13)
    + Rational(1, 504) * commutator(M2, C113)
    - Rational(1, 840) * commutator(M2, commutator(M2, C12)) 
    + Rational(1, 2520) * commutator(M3, C112)
    - Rational(1, 2520) * commutator(M1, commutator(M1, C113))
    - Rational(1, 7560) * commutator(C12, C112) 
    + Rational(1, 2520) * commutator(M2, commutator(M1, C112))
    - Rational(1, 15120) * commutator(M1, commutator(M1, commutator(M1, C112)))
)

simplified_expr = simplify(expand(expr))

printer = CustomStrPrinter()

for i in range(3):
    for j in range(3):
        print(f"Omega[{i}, {j}] = {printer.doprint(simplified_expr[i, j])}\n")