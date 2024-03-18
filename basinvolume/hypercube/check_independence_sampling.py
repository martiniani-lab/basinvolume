import random
import numpy as np
from basinvolume.monte_carlo import IndependenceSampling


def main():
    number_replicas = 20
    nexchanges = number_replicas**5
    print(nexchanges)
    i32max = np.iinfo(np.int32).max
    seed_exchanges = random.randint(0, i32max)

    independence_sampling = IndependenceSampling(seed_exchanges)
    dxs = np.array([random.random() for _ in range(number_replicas)])
    betas = np.array([random.random() for _ in range(number_replicas)])

    exchange_pattern = np.empty(number_replicas, dtype="int32")
    naccept = independence_sampling.exchange(exchange_pattern, dxs, betas, nexchanges)

    independence_sampling_new = IndependenceSampling(seed_exchanges)
    energies = np.array([0.5 * beta * dx * dx for beta in betas for dx in dxs])
    exchange_pattern_new = np.empty(number_replicas, dtype="int32")
    naccept_new = independence_sampling_new.exchange_energies(exchange_pattern_new, energies, nexchanges)

    print(exchange_pattern)
    print(naccept)
    print(exchange_pattern_new)
    print(naccept_new)

    print(np.all(exchange_pattern == exchange_pattern_new))
    print(naccept == naccept_new)


if __name__ == '__main__':
    main()
