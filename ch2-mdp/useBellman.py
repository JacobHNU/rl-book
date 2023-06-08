import sympy
from sympy import symbols
sympy.init_printing()
v_hungry, v_full = symbols('v_hungry v_full')
q_hungry_eat, q_hungry_none, q_full_eat, q_full_none = \
     symbols('q_hungry_eat q_hungry_none q_full_eat  q_full_none')
alpha, beta, x, y, gamma = symbols('alpha beta x y gamma')

# 求解Bellman expectation equation
system = sympy.Matrix((
    (1, 0, x-1, -x, 0, 0, 0),
    (0, 1, 0, 0, -y, y - 1, 0),
    (-gamma, 0, 1, 0, 0, 0, -2),
    ((alpha - 1) * gamma, -alpha * gamma, 0, 1, 0, 0, 4 * alpha - 3),
    (-beta * gamma, (beta - 1) * gamma, 0, 0, 1, 0, -4 * beta + 2),
    (0, -gamma, 0, 0, 0, 1, 1) ))  # 标准形式的系数矩阵
result = sympy.solve_linear_system(system, v_hungry, v_full,
                          q_hungry_none,q_hungry_eat,q_full_none,q_full_eat)  # solve_linear_system可以求解整理成标准形式系数矩阵的线性方程组

print(result)
# result :
# v_hungry: (-alpha*gamma*x*y + 3*alpha*gamma*x - 4*alpha*x + beta*gamma*x*y + 2*beta*gamma*y - gamma*x - 2*gamma + x + 2)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1),
# v_full: (-alpha*gamma*x*y - alpha*gamma*x + beta*gamma*x*y - 2*beta*gamma*y + 4*beta*y + gamma*y + gamma - y - 1)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1),
# q_hungry_none: (-alpha*gamma**2*x*y + alpha*gamma**2*x - 2*alpha*gamma*x + beta*gamma**2*x*y + 2*beta*gamma*y - gamma**2*x + gamma*x - 2*gamma + 2)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1),
# q_hungry_eat: (-alpha*gamma**2*x*y + alpha*gamma**2*x + alpha*gamma**2*y - alpha*gamma**2 - 2*alpha*gamma*x - alpha*gamma*y + 5*alpha*gamma - 4*alpha + beta*gamma**2*x*y - beta*gamma**2*y + 3*beta*gamma*y - gamma**2*x + gamma**2 + gamma*x - 4*gamma + 3)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1),
# q_full_none: (-alpha*gamma**2*x*y + alpha*gamma**2*x - 2*alpha*gamma*x + beta*gamma**2*x*y - beta*gamma**2*x - beta*gamma**2*y + beta*gamma**2 + beta*gamma*x + 3*beta*gamma*y - 5*beta*gamma + 4*beta + gamma**2*y - gamma**2 - gamma*y + 3*gamma - 2)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1),
# q_full_eat: (-alpha*gamma**2*x*y - alpha*gamma*x + beta*gamma**2*x*y - beta*gamma**2*y + 3*beta*gamma*y + gamma**2*y - gamma*y + gamma - 1)/(alpha*gamma**2*x - alpha*gamma*x + beta*gamma**2*y - beta*gamma*y - gamma**2 + 2*gamma - 1)


# 求解 Bellman Optimal Equation
xy_tuples = ((0, 0), (1, 0), (0, 1), (1, 1))
for x, y in xy_tuples:
    system = sympy.Matrix((
            (1, 0, x-1, -x, 0, 0, 0),
            (0, 1, 0, 0, -y, y-1, 0),
            (-gamma, 0, 1, 0, 0, 0, -2),
            ((alpha-1)*gamma, -alpha*gamma, 0, 1, 0, 0, 4*alpha-3),
            (-beta*gamma, (beta-1)*gamma, 0, 0, 1, 0, -4*beta+2),
            (0, -gamma, 0, 0, 0, 1, 1) ))
    result = sympy.solve_linear_system(system,
            v_hungry, v_full,
            q_hungry_none, q_hungry_eat, q_full_none, q_full_eat, simplification=True)
    msgx = 'v(饿) = q(饿,{}吃)'.format('' if x else '不')
    msgy = 'v(饱) = q(饱,{}吃)'.format('不' if y else '')
    print('==== {}, {} ==== x = {}, y = {} ===='.format(msgx, msgy, x, y))
    print(result)
# result :
# ==== v(饿) = q(饿,不吃), v(饱) = q(饱,吃) ==== x = 0, y = 0 ====
# {v_hungry: 2/(gamma - 1), v_full: -1/(gamma - 1), q_hungry_none: 2/(gamma - 1), q_hungry_eat: (alpha*gamma - 4*alpha - gamma + 3)/(gamma - 1), q_full_none: (-beta*gamma + 4*beta + gamma - 2)/(gamma - 1), q_full_eat: -1/(gamma - 1)}
# ==== v(饿) = q(饿,吃), v(饱) = q(饱,吃) ==== x = 1, y = 0 ====
# {v_hungry: (3*alpha*gamma - 4*alpha - 3*gamma + 3)/(alpha*gamma**2 - alpha*gamma - gamma**2 + 2*gamma - 1), v_full: -1/(gamma - 1), q_hungry_none: (alpha*gamma**2 - 2*alpha*gamma - gamma**2 - gamma + 2)/(alpha*gamma**2 - alpha*gamma - gamma**2 + 2*gamma - 1), q_hungry_eat: (3*alpha*gamma - 4*alpha - 3*gamma + 3)/(alpha*gamma**2 - alpha*gamma - gamma**2 + 2*gamma - 1), q_full_none: (alpha*gamma**2 - 2*alpha*gamma - 4*beta*gamma + 4*beta - gamma**2 + 3*gamma - 2)/(alpha*gamma**2 - alpha*gamma - gamma**2 + 2*gamma - 1), q_full_eat: -1/(gamma - 1)}
# ==== v(饿) = q(饿,不吃), v(饱) = q(饱,不吃) ==== x = 0, y = 1 ====
# {v_hungry: 2/(gamma - 1), v_full: (-2*beta*gamma + 4*beta + 2*gamma - 2)/(beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_hungry_none: 2/(gamma - 1), q_hungry_eat: (4*alpha*gamma - 4*alpha - beta*gamma**2 + 3*beta*gamma + gamma**2 - 4*gamma + 3)/(beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_full_none: (-2*beta*gamma + 4*beta + 2*gamma - 2)/(beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_full_eat: (-beta*gamma**2 + 3*beta*gamma + gamma**2 - 1)/(beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1)}
# ==== v(饿) = q(饿,吃), v(饱) = q(饱,不吃) ==== x = 1, y = 1 ====
# {v_hungry: (2*alpha*gamma - 4*alpha + 3*beta*gamma - 3*gamma + 3)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), v_full: (-2*alpha*gamma - beta*gamma + 4*beta + 2*gamma - 2)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_hungry_none: (-2*alpha*gamma + beta*gamma**2 + 2*beta*gamma - gamma**2 - gamma + 2)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_hungry_eat: (2*alpha*gamma - 4*alpha + 3*beta*gamma - 3*gamma + 3)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_full_none: (-2*alpha*gamma - beta*gamma + 4*beta + 2*gamma - 2)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1), q_full_eat: (-alpha*gamma**2 - alpha*gamma + 3*beta*gamma + gamma**2 - 1)/(alpha*gamma**2 - alpha*gamma + beta*gamma**2 - beta*gamma - gamma**2 + 2*gamma - 1)}

