import numpy as np
from basinvolume.post_processing import *
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
    print "fixed_integral:"
    print fixed_integral
    print "quad_integral:"
    print quad_integral
    integral_range_1 = calculate_GL_integral_range(lambda(x) : np.sin(x)*np.sin(x),-1,1,n=order)
    print "integral_range_1:"
    print integral_range_1
    quad_integral_2 = quad(lambda(x) : np.sin(x)*np.sin(x),-2,2)
    print "quad_integral_2:"
    print quad_integral_2
    integral_range_2 = calculate_GL_integral_range(lambda(x) : np.sin(x)*np.sin(x),-2,2,n=order)
    print "integral_range_2:"
    print integral_range_2