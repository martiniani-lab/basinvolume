import numpy as np
from pymbar import MBAR


class MBARWrapper(MBAR):
    def __init__(self, permutation, u_kn, n_k, *args, **kwargs):
        # The permutation argument gives the rearrangement of the windows.
        # It must be a permutation of the integers 0 to len(N_k) - 1.
        self.permutation = permutation
        assert np.all(np.sort(permutation) == np.arange(len(n_k)))
        # See https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
        self.idx = np.empty_like(permutation)
        self.idx[permutation] = np.arange(len(permutation))
        permuted_u_kn = u_kn[self.idx, :]
        permuted_n_k = n_k[self.idx]
        if "initial_f_k" in kwargs and kwargs["initial_f_k"] is not None:
            kwargs["initial_f_k"] = kwargs["initial_f_k"][self.idx]
        x_kindices = np.zeros(u_kn.shape[1], dtype=np.int64)
        n_sum = 0
        for k in range(len(n_k)):
            x_kindices[n_sum: n_sum + n_k[k]] = self.permutation[k]
            n_sum += n_k[k]
        assert "x_kindices" not in kwargs
        super().__init__(permuted_u_kn, permuted_n_k, *args, x_kindices=x_kindices, **kwargs)

    class _ResultWrapper(object):
        def __init__(self, result_array, permutation):
            self.result_array = result_array
            self.permutation = permutation

        def __getitem__(self, key):
            if isinstance(key, (int, np.integer)):
                if len(self.result_array.shape) > 1:
                    return type(self)(self.result_array[self.permutation[key]], self.permutation)
                else:
                    return self.result_array[self.permutation[key]]
            elif isinstance(key, tuple):
                new_key = tuple(self.permutation[i] for i in key)
                if len(new_key) < len(self.result_array.shape):
                    return type(self)(self.result_array[new_key], self.permutation)
                else:
                    return self.result_array[new_key]
            elif isinstance(key, slice):
                if len(self.result_array.shape) > 1:
                    raise TypeError("Invalid key type")
                return self.result_array[self.permutation[key]]
            else:
                raise TypeError("Invalid key type")

        def __neg__(self):
            self.result_array = -self.result_array
            return self

    def compute_free_energy_differences(self, *args, **kwargs):
        result_dict = super().compute_free_energy_differences(*args, **kwargs)
        wrapped_result_dict = {}
        for key, value in result_dict.items():
            wrapped_result_dict[key] = self._ResultWrapper(value, self.permutation)
        return wrapped_result_dict

    def compute_perturbed_free_energies(self, *args, **kwargs):
        result_dict = super().compute_perturbed_free_energies(*args, **kwargs)
        wrapped_result_dict = {}
        for key, value in result_dict.items():
            wrapped_result_dict[key] = self._ResultWrapper(value, self.permutation)
        return wrapped_result_dict

    def get_f_k(self):
        # TODO: TEST RUN_BS
        print("HALLO")
        return self._ResultWrapper(self.f_k, self.permutation)
