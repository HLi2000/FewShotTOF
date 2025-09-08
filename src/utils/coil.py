import numpy as np
import dask.array as da
import time

class CoilCompressorSVD:
    """SVD-based coil compression.
    coil_axis = 0

    Args:
      coil_axis: An `int`. Defaults to -1.
      out_coils: An `int`. The desired number of virtual output coils. Cannot be
        used together with `variance_ratio`.
      variance_ratio: A `float` between 0.0 and 1.0. The percentage of total
        variance to be retained. The number of virtual coils is automatically
        selected to retain at least this percentage of variance. Cannot be used
        together with `out_coils`.
    """

    def __init__(self, out_coils=None, variance_ratio=None, is_3D=False):
        if out_coils is not None and variance_ratio is not None:
            raise ValueError("Cannot specify both `out_coils` and `variance_ratio`.")
        self._out_coils = out_coils
        self._variance_ratio = variance_ratio
        self._singular_values = None
        self._explained_variance = None
        self._explained_variance_ratio = None
        self._energy_retained = None
        self.is_3D = is_3D

    def compress(self, kspace):
        """Fits the coil compression matrix.

            Args:
              kspace: A NumPy array. The multi-coil k-space data. Must have type
                `complex64` or `complex128`.

            Returns:
              The fitted `CoilCompressorSVD` object.
            """
        # print(f"SVD-based coil compression:")
        t = time.time()
        eps = np.finfo(np.float32).eps
        kspace = np.asarray(kspace)

        if not self.is_3D:
            encoding_dimensions = kspace.shape[2:]
            num_coils = kspace.shape[0]

            self._explained_variance_ratio = []
            self._energy_retained = []
            res = []
            A_x_1 = None
            for x in range(kspace.shape[1]):
                # print(x)
                kspace_x = kspace[:,x]

                # Flatten the encoding dimensions.
                kspace_x = kspace_x.reshape([num_coils, -1])
                num_samples = kspace_x.shape[1]

                # Compute singular-value decomposition.
                U, S, VT = da.linalg.svd(da.asarray(kspace_x).persist())
                U, S, VT = np.asarray(U), np.asarray(S), np.asarray(VT)

                # Get variance.
                explained_variance = S ** 2 / (num_samples - 1)
                total_variance = np.sum(explained_variance)
                self._explained_variance_ratio.append(explained_variance / (total_variance + eps))

                out_coils = self._out_coils

                self._energy_retained.append(np.sum(self._explained_variance_ratio[x][:out_coils]) / (np.sum(
                    self._explained_variance_ratio[x]) + eps))

                # Compress
                if isinstance(out_coils, int):
                    A_x = U[:, :out_coils].conj().T
                    if A_x_1 is not None:
                        C_x = A_x @ A_x_1.conj().T
                        U_c, S_c, VT_c = da.linalg.svd(da.asarray(C_x).persist())
                        U_c, S_c, VT_c = np.asarray(U_c), np.asarray(S_c), np.asarray(VT_c)
                        A_x = (VT_c.conj().T @ U_c.conj().T) @ A_x
                    A_x_1 = A_x
                    # kspace_x = (np.diag(S[:out_coils]) @ VT[:out_coils, :])
                    kspace_x = (A_x @ kspace_x)

                # Restore data shape.
                res.append(kspace_x.reshape((out_coils,) + encoding_dimensions))

            kspace = np.stack(res, axis=1)
            print(f"Compressed {num_coils} coils to {out_coils} virtual coils")
            print(f"Percentage of energy retained: {np.mean(self._energy_retained) * 100:.2f}%")

        else:
            # Flatten the encoding dimensions.
            encoding_dimensions = kspace.shape[1:]
            num_coils = kspace.shape[0]
            kspace = kspace.reshape([num_coils, -1])
            num_samples = kspace.shape[1]

            # Compute singular-value decomposition.
            U, S, VT = da.linalg.svd(da.asarray(kspace).persist())
            U, S, VT = np.asarray(U), np.asarray(S), np.asarray(VT)

            # Get variance.
            self._singular_values = S
            self._explained_variance = S ** 2 / (num_samples - 1)
            total_variance = np.sum(self._explained_variance)
            self._explained_variance_ratio = self._explained_variance / (total_variance + eps)

            # Get output coils from variance ratio.
            if self._variance_ratio is not None:
                cum_variance = np.cumsum(self._explained_variance_ratio)
                self._out_coils = np.count_nonzero(cum_variance <= self._variance_ratio)
            out_coils = self._out_coils
            # print(f"Compressed {num_coils} coils to {out_coils} virtual coils")

            self._energy_retained = np.sum(self._explained_variance_ratio[:out_coils]) / (np.sum(self._explained_variance_ratio) + eps)
            # print(f"Percentage of energy retained: {self._energy_retained * 100:.2f}%")

            # Compress
            if isinstance(out_coils, int):
                # kspace = (np.diag(S[:out_coils]) @ VT[:out_coils, :])
                kspace = (U[:, :out_coils].conj().T @ kspace)

            # Restore data shape.
            kspace = kspace.reshape((out_coils,) + encoding_dimensions)

        print('Elapsed: {:.2f}s'.format(time.time() - t))
        return kspace
