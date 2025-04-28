from sympy import symbols, Matrix, I, sqrt, simplify, latex, fraction, expand, collect, rcollect, factor_terms, factor

# Define real symbols for phi components
alpha, beta = symbols('alpha beta', real=True)

# Define phi as a column vector
phi = Matrix([alpha, beta])

# Define Pauli matrices
sigma_x = Matrix([[0, 1], [1, 0]])
sigma_y = Matrix([[0, -I], [I, 0]])
sigma_z = Matrix([[1, 0], [0, -1]])

x, y, z = symbols('x y z', real=True)
#epsilon = symbols('epsilon', real=True)
epsilon = sqrt(x**2 + y**2 + z**2)
normalization = 1 / (2 * epsilon * (z + epsilon))
V = Matrix([[z + epsilon, -x + I * y], [x + I * y, z + epsilon]])

phi_dagger = phi.H  # Hermitian (conjugate transpose)
V_dagger = V.H  # Hermitian (conjugate transpose)

pattern = x**2 + y**2 + z**2
epsilon_symbol = symbols('epsilon', real=True, positive=True)

print("Test: ", simplify(normalization * V_dagger * V))

simga_results = []

for sigma_i, name in zip([sigma_x, sigma_y, sigma_z], ['sigma^x', 'sigma^y', 'sigma^z']):
    as_matrix = phi_dagger * V_dagger * sigma_i * V * phi
    expression = simplify(normalization * as_matrix[0])
    sub_expr = expression.subs(pattern, epsilon_symbol**2)
    n, d = fraction(sub_expr)
    
    simga_results.append(collect(expand(n), [alpha, beta]))
    print(f"\\epsilon (z + \\epsilon) \\langle \\{name} \\rangle &=", latex(simga_results[-1]), "\\\\")
print()    

norm = simga_results[0]**2 + simga_results[1]**2 + simga_results[2]**2

simp = expand(norm)
simp = simp.subs(pattern, epsilon_symbol**2)
### alpha^4
simp = collect(expand(simp), alpha**4 * epsilon_symbol**2)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), 2 * alpha**4 * epsilon_symbol * z)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), alpha**4 * z**2)
simp = simp.subs(pattern, epsilon_symbol**2)
### beta^4
simp = collect(expand(simp), beta**4 * epsilon_symbol**2)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), 2 * beta**4 * epsilon_symbol * z)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), beta**4 * z**2)
simp = simp.subs(pattern, epsilon_symbol**2)
### alpha^2 beta^2
simp = collect(expand(simp), 4 * alpha**2 * beta**2 * epsilon_symbol * z)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), 2 * alpha**2 * beta**2 * z**2)
simp = simp.subs(x**2 + 3 * y**2 + z**2, epsilon_symbol**2 + 2 * y**2)

simp = collect(expand(simp), 4 * alpha**2 * beta**2 * y**2)
simp = simp.subs(pattern, epsilon_symbol**2)

simp = collect(expand(simp), 2 * alpha**2 * beta**2 * epsilon_symbol**2)
simp = simp.subs(simplify(pattern + z**2), epsilon_symbol**2 + z**2)
### finalize
simp = collect(expand(simp), [alpha, beta, 2 * alpha**2 * beta**2])

simp = factor(simp)
norm = simplify(simp / (epsilon_symbol * (epsilon_symbol + z))**2)

print(r"| \langle \vec{\sigma}(0) \rangle |^2 = ", latex(norm))