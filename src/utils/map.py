import time

import sigpy as sp
import numpy as np
import numba as nb
import dask.array as da
from scipy import ndimage, stats
from scipy.signal.windows import hann


def AdaptiveCombine(kspace, calib, smoothing=5, niter=5, thresh=1e-3,
                             verbose=False):
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : ndarray
        Input images, [coil, x, y] or [coil, x, y, z].
    smoothing : int or ndarray-like
        Smoothing block size(s) for the spatial axes.
    niter : int
        Maximal number of iterations to run.
    thresh : float
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all ``niter`` iterations will be performed.
    verbose : bool
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : ndarray
        Relative coil sensitivity maps, [coil, y, x] or [coil, z, y, x].
    coil_combined : ndarray
        The coil combined image volume, [y, x] or [z, y, x].

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's ``coil_map_3d_Inati_Iter`` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """

    kspace = np.asarray(kspace)
    if kspace.ndim < 3 or kspace.ndim > 4:
        raise ValueError("Expected 3D [ncoils, nx, ny] or 4D "
                         " [ncoils, nz, ny, nz] input.")

    kspace = mask_calib(kspace, calib)

    if kspace.ndim == 3:
        im = sp.ifft(kspace, axes=[1, 2])
        # pad to size 1 on z for 2D + coils case
        images_are_2D = True
        im = im[:, :, :, np.newaxis]
    else:
        im = sp.ifft(kspace, axes=[1, 2, 3])
        images_are_2D = False

    # convert smoothing kernel to array
    if isinstance(smoothing, int):
        smoothing = np.asarray([smoothing, ] * 3)
    smoothing = np.asarray(smoothing)
    if smoothing.ndim > 1 or smoothing.size != 3:
        raise ValueError("smoothing should be an int or a 3-element 1D array")

    if images_are_2D:
        smoothing[2] = 1  # no smoothing along z in 2D case

    # smoothing kernel is size 1 on the coil axis
    smoothing = np.concatenate(([1, ], smoothing), axis=0)

    ncha = im.shape[0]

    try:
        # numpy >= 1.7 required for this notation
        D_sum = im.sum(axis=(1, 2, 3))
    except:
        D_sum = im.reshape(ncha, -1).sum(axis=1)

    v = 1/np.linalg.norm(D_sum)
    D_sum *= v
    R = 0

    for cha in range(ncha):
        R += np.conj(D_sum[cha]) * im[cha, ...]

    eps = np.finfo(im.real.dtype).eps * np.abs(im).mean()
    for it in range(niter):
        if verbose:
            print("Coil map estimation: iteration %d of %d" % (it+1, niter))
        if thresh > 0:
            prevR = R.copy()
        R = np.conj(R)
        coil_map = im * R[np.newaxis, ...]
        coil_map_conv = smooth(coil_map, box=smoothing)
        D = coil_map_conv * np.conj(coil_map_conv)
        R = D.sum(axis=0)
        R = np.sqrt(R) + eps
        R = 1/R
        coil_map = coil_map_conv * R[np.newaxis, ...]
        D = im * np.conj(coil_map)
        R = D.sum(axis=0)
        D = coil_map * R[np.newaxis, ...]
        try:
            # numpy >= 1.7 required for this notation
            D_sum = D.sum(axis=(1, 2, 3))
        except:
            D_sum = im.reshape(ncha, -1).sum(axis=1)
        v = 1/np.linalg.norm(D_sum)
        D_sum *= v

        imT = 0
        for cha in range(ncha):
            imT += np.conj(D_sum[cha]) * coil_map[cha, ...]
        magT = np.abs(imT) + eps
        imT /= magT
        R = R * imT
        imT = np.conj(imT)
        coil_map = coil_map * imT[np.newaxis, ...]

        if thresh > 0:
            diffR = R - prevR
            vRatio = np.linalg.norm(diffR) / np.linalg.norm(R)
            if verbose:
                print("vRatio = {}".format(vRatio))
            if vRatio < thresh:
                break

    coil_combined = (im * np.conj(coil_map)).sum(0)

    if images_are_2D:
        # remove singleton z dimension that was added for the 2D case
        coil_combined = coil_combined[0, :, :]
        coil_map = coil_map[:, :, :, 0]

    return coil_map, coil_combined

def smooth_calib(calib):
    if calib.ndim == 2:
        # Generate a Hann window
        hann_x, hann_y = hann(calib.shape[0])[:, np.newaxis], hann(calib.shape[1])[:, np.newaxis]
        # Apply the Hann window along both dimensions for a 2D array
        smoothed_calib = hann_x**0.3 * hann_y.T**0.3 * calib

    elif calib.ndim == 3:
        hann_x, hann_y = hann(calib.shape[0])[:, np.newaxis, np.newaxis], hann(calib.shape[1])[:, np.newaxis, np.newaxis]
        hann_z = hann(calib.shape[2])[:, np.newaxis, np.newaxis]
        # Apply the Hann window along each dimension for a 3D array
        smoothed_calib = hann_x**0.3 * hann_y.T**0.3 * hann_z.T**0.3 * calib

    return smoothed_calib

def mask_calib(kspace, calib):
    if kspace.ndim == 3:
        nx, ny = kspace.shape[1:3]
        mask = np.zeros_like(kspace)

        calib_reg = np.ones(calib)
        calib_reg = smooth_calib(calib_reg)

        # Add calibration region
        mask[:, int(nx / 2 - calib[0] / 2):int(nx / 2 + calib[0] / 2),
        int(ny / 2 - calib[1] / 2):int(ny / 2 + calib[1] / 2)] = calib_reg

    else:
        nx, ny, nz = kspace.shape[1:4]
        mask = np.zeros_like(kspace)

        calib_reg = np.ones((calib[0],)+calib)
        calib_reg = smooth_calib(calib_reg)

        # Add calibration region
        mask[:, int(nx / 2 - calib[0] / 2):int(nx / 2 + calib[0] / 2),
        int(ny / 2 - calib[0] / 2):int(ny / 2 + calib[0] / 2),
        int(nz / 2 - calib[1] / 2):int(nz / 2 + calib[1] / 2)] = calib_reg

    return kspace * mask

def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''

    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
    ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag

    return simg


def poisson(img_shape, accel, slope=1000, calib=(12, 6), dtype=complex,
            crop_corner=True, seed=0, max_attempts=20, tol=0.1):
    """Generate variable-density Poisson-disc sampling pattern.

    The function generates a variable density Poisson-disc sampling
    mask with density proportional to :math:`1 / (1 + s |r|)`,
    where :math:`r` represents the k-space radius, and :math:`s`
    represents the slope. A binary search is performed on the slope :math:`s`
    such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol`
    determines how much they can deviate.

    Args:
        img_shape (tuple of ints): length-1 or length-2 image shape.
        accel (float): Target acceleration factor. Must be greater than 1.
        slope (int): Slope for density variation.
        calib (tuple of ints): length-1 or length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        seed (int): Random seed.
        max_attempts (int): maximum number of samples to reject in Poisson
           disc calculation. 30
        tol (float): Tolerance for how much the resulting acceleration can
            deviate from `accel`.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.

    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")

    if seed is not None:
        rand_state = np.random.get_state()

    if len(img_shape) == 1 or img_shape[0] == 1 or img_shape[1] == 1:
        # Handle 1D case
        length = max(img_shape)
        r = np.maximum(abs(np.arange(length) - length / 2) - calib[0] / 2, 0)
        r /= r.max()

        pow = 1
        actual_accel = 0
        ratio = 1 / np.log10(max(slope, 10)) ** 0.5
        while abs(actual_accel - accel) > tol:
            radius = np.clip((1 + r * slope) ** pow * length / length, 1, None)
            mask = _poisson_1d(length, max_attempts, radius, calib, seed)

            actual_accel = length / np.sum(mask)

            scale = accel / actual_accel
            pow *= scale ** ratio
            ratio = ratio ** 1.001

        if abs(actual_accel - accel) >= tol:
            raise ValueError(f"Cannot generate mask to satisfy accel={accel}.")

        if seed is not None:
            np.random.set_state(rand_state)

        return mask.reshape(img_shape).astype(dtype)

    else:
        # Handle 2D case
        ny, nx = img_shape
        y, x = np.mgrid[:ny, :nx]
        x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
        x /= x.max()
        y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
        y /= y.max()
        r = np.sqrt(x ** 2 + y ** 2)

        pow = 1
        actual_accel = 0
        ratio = 1 / np.log10(max(slope, 10)) ** 0.5
        while abs(actual_accel - accel) > tol:
            radius_x = np.clip((1 + r * slope) ** pow * nx / max(nx, ny), 1, None)
            radius_y = np.clip((1 + r * slope) ** pow * ny / max(nx, ny), 1, None)
            mask = _poisson(img_shape[-1], img_shape[-2], max_attempts, radius_x, radius_y, calib, seed)
            if crop_corner:
                mask *= r < 1

            actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

            scale = accel / actual_accel
            if actual_accel < accel:
                pow *= scale ** ratio
            else:
                pow *= scale ** ratio

            ratio = ratio ** 1.001

        if abs(actual_accel - accel) >= tol:
            raise ValueError(f"Cannot generate mask to satisfy accel={accel}.")

        if seed is not None:
            np.random.set_state(rand_state)

        return mask.reshape(img_shape).astype(dtype)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, max_attempts, radius_x, radius_y, calib, seed=None):
    """2D Poisson-disc sampling implementation."""
    mask = np.zeros((ny, nx))

    # Add calibration region
    mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
         int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    if seed is not None:
        np.random.seed(int(seed))

    # initialize active list
    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    num_actives = 1
    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        px = pxs[i]
        py = pys[i]
        rx = radius_x[py, px]
        ry = radius_y[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate point randomly from r and 2 * r
            v = (np.random.random() * 3 + 1)**0.5
            t = 2 * np.pi * np.random.random()
            qx = px + v * rx * np.cos(t)
            qy = py + v * ry * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                startx = max(int(qx - rx), 0)
                endx = min(int(qx + rx + 1), nx)
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1
                            and (((qx - x) / radius_x[y, x])**2 +
                                 ((qy - y) / (radius_y[y, x]))**2 < 1)):
                            done = False
                            break

            k += 1

        # Add point if done else remove from active list
        if done:
            pxs[num_actives] = qx
            pys[num_actives] = qy
            mask[int(qy), int(qx)] = 1
            num_actives += 1
        else:
            pxs[i] = pxs[num_actives - 1]
            pys[i] = pys[num_actives - 1]
            num_actives -= 1

    return mask


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson_1d(length, max_attempts, radius, calib, seed=None):
    """1D Poisson-disc sampling implementation with initial points outside calibration area."""
    mask = np.zeros(length)

    # Add calibration region
    calib_start = int(length / 2 - calib[0] / 2)
    calib_end = int(length / 2 + calib[0] / 2)
    mask[calib_start:calib_end] = 1

    if seed is not None:
        np.random.seed(int(seed))

    # Initialize active list with two points outside the calibration area on each side
    ps = np.empty(length, np.int32)

    # Point near the left side of calibration area
    ps[0] = np.random.randint(0, calib_start)
    # Point near the right side of calibration area
    ps[1] = np.random.randint(calib_end, length)
    num_actives = 2

    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        p = ps[i]
        r = radius[p]

        # Attempt to generate point within the range [r, 2r]
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate a candidate point within the range [r, 2r]
            v = np.random.random() + 1  # v will be in the range [1, 2]
            q = p + int(v * r) * np.random.choice(np.asarray([-1, 1]))

            # Reject if outside grid or close to other points
            if 0 <= q < length:
                start = max(int(q - r), 0)
                end = min(int(q + r + 1), length)

                done = True
                for x in range(start, end):
                    if mask[x] == 1 and abs(q - x) < r:
                        done = False
                        break

            k += 1

        # Add point if done else remove from active list
        if done:
            ps[num_actives] = q
            mask[q] = 1
            num_actives += 1
        else:
            ps[i] = ps[num_actives - 1]
            num_actives -= 1

    return mask


def undersample_mask(imsize, r, slope=10, type='ber', n_dim=2, seed=None, calib=(1,1)):
    # Generates sampling masks based on specified sampling types

    if r < 1.0:
        raise ValueError('Invalid undersampling ratio delta in generateSamplingMask')
    elif r == 1:
        return np.full(imsize, True, dtype=bool)
    else:
        if seed == None:
            seed = np.random.get_state()[1][0]

        if n_dim == 1:
            size = (1, imsize[1])
        elif n_dim == 2:
            size = imsize[0:2]
        else:
            size = imsize[1:3]

        # Selects the specified sampling type
        if type == 'ber': # Bernoulli
            indices = stats.bernoulli.rvs(size=(size), p=1/r)
            mask = np.ma.make_mask(indices)
        elif type == 'poiss':  # variable-density Poisson-disc
            mask = poisson(size, r, slope=slope, calib=calib, seed=seed, tol=0.01)
        else:
            raise ValueError('Invalid sampling type')

        if n_dim == 1:
            mask = np.tile(mask, (imsize[0], 1)).astype(bool)
            return np.stack([mask] * imsize[2], axis=2).astype(bool)
        elif n_dim == 2:
            return np.stack([mask] * imsize[2], axis=2).astype(bool)
        else:
            return np.stack([mask] * imsize[0], axis=0).astype(bool)

class EspiritCalib(sp.app.App):
    """ESPIRiT calibration.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
        calib (tuple of ints): calibration shape.
        thresh (float): threshold for the calibration matrix.
        kernel_width (int): kernel width for the calibration matrix.
        max_power_iter (int): maximum number of power iterations.
        device (Device): computing device.
        crop (int): cropping threshold.

    Returns:
        array: ESPIRiT maps of the same shape as ksp.

    References:
        Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
        John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
        ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
        Where SENSE meets GRAPPA.
        Magnetic Resonance in Medicine, 71:990-1001 (2014)

    """

    def __init__(
        self,
        ksp,
        calib_sz=(24,24),
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        max_iter=100,
        device=sp.cpu_device,
        output_eigenvalue=False,
        show_pbar=True,
    ):
        self.device = sp.Device(device)
        self.output_eigenvalue = output_eigenvalue
        self.crop = crop

        img_ndim = ksp.ndim - 1
        num_coils = len(ksp)
        with sp.get_device(ksp):
            # Get calibration region
            if len(calib_sz) == img_ndim:
                calib_shape = (num_coils,) + tuple(calib_sz)
            elif len(calib_sz) == 2 and img_ndim == 3:
                calib_shape = (num_coils,) + (calib_sz[0],) + tuple(calib_sz)
            calib = sp.resize(ksp, calib_shape)
            if 2*kernel_width > min(calib_sz):
                min_index, min_value = min(enumerate(calib_sz), key = lambda n: n[1])
                calib_sz = list(calib_sz)
                calib_sz[min_index] = 2*kernel_width
                calib_shape = (num_coils,) + (calib_sz[0],) + tuple(calib_sz)
                calib = sp.resize(calib, calib_shape)
            calib = sp.to_device(calib, device)

        xp = self.device.xp
        with self.device:
            # Get calibration matrix.
            # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
            mat = sp.array_to_blocks(
                calib, [kernel_width] * img_ndim, [1] * img_ndim
            )
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])

            # Perform SVD on calibration matrix
            _, S, VH = xp.linalg.svd(mat, full_matrices=False)
            VH = VH[S > thresh * S.max(), :]

            # Get kernels
            num_kernels = len(VH)
            kernels = VH.reshape(
                [num_kernels, num_coils] + [kernel_width] * img_ndim
            )
            img_shape = ksp.shape[1:]

            # Get covariance matrix in image domain
            AHA = xp.zeros(
                img_shape[::-1] + (num_coils, num_coils), dtype=ksp.dtype
            )
            for kernel in kernels:
                img_kernel = sp.ifft(
                    sp.resize(kernel, ksp.shape), axes=range(-img_ndim, 0)
                )
                aH = xp.expand_dims(img_kernel.T, axis=-1)
                a = xp.conj(aH.swapaxes(-1, -2))
                AHA += aH @ a

            AHA *= sp.prod(img_shape) / kernel_width**img_ndim
            self.mps = xp.ones(ksp.shape[::-1] + (1,), dtype=ksp.dtype)

            def forward(x):
                with sp.get_device(x):
                    return AHA @ x

            def normalize(x):
                with sp.get_device(x):
                    return (
                        xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5
                    )

            alg = sp.alg.PowerMethod(
                forward, self.mps, norm_func=normalize, max_iter=max_iter
            )

        super().__init__(alg, show_pbar=show_pbar)


    def _output(self):
        xp = self.device.xp
        with self.device:
            # Normalize phase with respect to first channel
            mps = self.mps.T[0]
            mps *= xp.conj(mps[0] / xp.abs(mps[0]))

            # Crop maps by thresholding eigenvalue
            max_eig = self.alg.max_eig.T[0]
            mps *= max_eig > self.crop

        if self.output_eigenvalue:
            return mps, max_eig
        else:
            return mps

def espirit(
        ksp,
        kernel_width=6,
        calib_sz=(24,24),
        thresh=0.02,
        crop=0.95,
        device=sp.cpu_device,
):
    """
    Derives the ESPIRiT operator.

    Arguments:
      ksp: Multi channel k-space data. Expected dimensions are (nc, sx, sy, sz), where (sx, sy, sz) are volumetric
         dimensions and (nc) is the channel dimension.
      kernel_width: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel
         will have dimensions (1, k, k, 8)
      calib_sz: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      thresh: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      crop: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """
    t = time.time()

    img_ndim = ksp.ndim - 1
    num_coils = len(ksp)

    # Get calibration region
    if len(calib_sz) == img_ndim:
        calib_shape = (num_coils,) + tuple(calib_sz)
    elif len(calib_sz) == 2 and img_ndim == 3:
        calib_shape = (num_coils,) + (calib_sz[0],) + tuple(calib_sz)
    C = sp.resize(ksp, calib_shape)
    if 2 * kernel_width > min(calib_sz):
        min_index, min_value = min(enumerate(calib_sz), key=lambda n: n[1])
        calib_sz = list(calib_sz)
        calib_sz[min_index] = 2 * kernel_width
        calib_shape = (num_coils,) + (calib_sz[0],) + tuple(calib_sz)
        C = sp.resize(C, calib_shape)
    C = sp.to_device(C, device)

    ksp = np.moveaxis(ksp, 0, -1)
    C = np.moveaxis(C, 0, -1)

    sx = ksp.shape[0]
    sy = ksp.shape[1]
    sz = ksp.shape[2] if img_ndim == 3 else None
    nc = num_coils

    # Construct Hankel matrix.
    xmax = max(1, C.shape[0] - kernel_width + 1)
    ymax = max(1, C.shape[1] - kernel_width + 1)
    zmax = max(1, C.shape[2] - kernel_width + 1) if img_ndim == 3 else 1
    A = np.zeros([xmax * ymax * zmax, kernel_width ** img_ndim * nc]).astype(np.complex64)

    idx = 0
    if img_ndim == 3:
        for xdx in range(xmax):
            for ydx in range(ymax):
                for zdx in range(zmax):
                  # numpy handles when the indices are too big
                  block = C[xdx:xdx + kernel_width, ydx:ydx + kernel_width, zdx:zdx + kernel_width, :].astype(np.complex64)
                  A[idx, :] = block.flatten()
                  idx = idx + 1
    else:
        for xdx in range(xmax):
            for ydx in range(ymax):
                # numpy handles when the indices are too big
                block = C[xdx:xdx + kernel_width, ydx:ydx + kernel_width, :].astype(np.complex64)
                A[idx, :] = block.flatten()
                idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = da.linalg.svd(da.asarray(A).persist())
    U, S, VH = np.asarray(U), np.asarray(S), np.asarray(VH)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= thresh * S[0])
    V = V[:, 0:n]

    kxt = (sx // 2 - kernel_width // 2, sx // 2 + kernel_width // 2 + kernel_width % 2) if (sx > 1) else (0, 1)
    kyt = (sy // 2 - kernel_width // 2, sy // 2 + kernel_width // 2 + kernel_width % 2) if (sy > 1) else (0, 1)
    kzt = (sz // 2 - kernel_width // 2, sz // 2 + kernel_width // 2 + kernel_width % 2) if img_ndim == 3 else None

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(ksp), n)).astype(np.complex64)
    if img_ndim == 3:
        kerdims = [(sx > 1) * kernel_width + (sx == 1) * 1, (sy > 1) * kernel_width + (sy == 1) * 1, (sz > 1) * kernel_width + (sz == 1) * 1, nc]
        for idx in range(n):
            kernels[kxt[0]:kxt[1], kyt[0]:kyt[1], kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)
    else:
        kerdims = [(sx > 1) * kernel_width + (sx == 1) * 1, (sy > 1) * kernel_width + (sy == 1) * 1, nc]
        for idx in range(n):
            kernels[kxt[0]:kxt[1],kyt[0]:kyt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    # Take the iucfft
    axes = (0, 1, 2) if img_ndim == 3 else (0, 1)
    kerimgs = np.zeros(np.append(np.shape(ksp), n)).astype(np.complex64)
    if img_ndim == 3:
        for idx in range(n):
            for jdx in range(nc):
                ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
                kerimgs[:,:,:,jdx,idx] = sp.fft(ker, axes=axes) * np.sqrt(sx * sy * sz)/np.sqrt(kernel_width ** img_ndim)
    else:
        for idx in range(n):
            for jdx in range(nc):
                ker = kernels[::-1, ::-1, jdx, idx].conj()
                kerimgs[:,:,jdx,idx] = sp.fft(ker, axes=axes) * np.sqrt(sx * sy)/np.sqrt(kernel_width ** img_ndim)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.append(np.shape(ksp), nc)).astype(np.complex64)
    if img_ndim == 3:
        for idx in range(0, sx):
            for jdx in range(0, sy):
                for kdx in range(0, sz):

                    Gq = kerimgs[idx,jdx,kdx,:,:]

                    u, s, vh = da.linalg.svd(da.asarray(Gq).persist())
                    u, s, vh = np.asarray(u), np.asarray(s), np.asarray(vh)
                    for ldx in range(0, nc):
                        if (s[ldx]**2 > crop):
                            maps[idx, jdx, kdx, :, ldx] = u[:, ldx]
    else:
        for idx in range(0, sx):
            for jdx in range(0, sy):
                Gq = kerimgs[idx, jdx, :, :]

                u, s, vh = da.linalg.svd(da.asarray(Gq).persist())
                u, s, vh = np.asarray(u), np.asarray(s), np.asarray(vh)
                for ldx in range(0, nc):
                    if (s[ldx] ** 2 > crop):
                        maps[idx, jdx, :, ldx] = u[:, ldx]

    print('ESPRiT elapsed: {:.2f}s'.format(time.time() - t))
    return np.moveaxis(maps, [-1, -2], [0, 1])