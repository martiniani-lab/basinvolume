print sim.success_list
print sim.nrattlers_list
print sim.pressure_list
print sim.energy_list
x, y = np.log(sim.energy_list), np.log(sim.pressure_list)
plt.scatter(x, y)
from scipy.optimize import curve_fit
def ff(x, a, b):
    return a * x + b
popt, pcov = curve_fit(ff, x, y)
plt.plot(x, ff(x, popt[0], popt[1]), label=r"$\ln P = {}\ln E + {}$".format(popt[0], popt[1]))
print "fit: ", popt
plt.legend()
plt.xlabel(r"$\ln E$")
plt.ylabel(r"$\ln P$")
plt.show()