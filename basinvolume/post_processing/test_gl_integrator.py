from gauss_lobatto import *
from scipy.integrate import fixed_quad, quad

if __name__ == "__main__":
    order = 22
    x = Gauss_Lobatto_abscissas(order)()
    f = [np.sin(xi)*np.sin(xi) for xi in x]
    our_integral = calculate_GL_integral(f)
    fixed_integral = fixed_quad(lambda(x) : np.sin(x)*np.sin(x),-1,1, n=order)
    quad_integral = quad(lambda(x) : np.sin(x)*np.sin(x),-1,1)
    print "our_integral:"
    print our_integral
    print "fixed_integral: "
    print fixed_integral
    print "quad_integral: "
    print quad_integral