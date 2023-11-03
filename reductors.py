import numpy as np
import scipy.linalg as spla
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel, _poles_b_c_to_lti
from pymor.reductors.h2 import IRKAReductor

from tools import cauchy_index, lti_with_better_cond


class IRKACauchyIndexReductor(IRKAReductor):
    def __init__(self, fom, mu=None):
        super().__init__(fom, mu=mu)
        self.logger.setLevel('INFO')

    def _clear_lists(self):
        super()._clear_lists()
        self.cauchy_indices = []

    def _compute_error(self, rom, it, compute_errors):
        super()._compute_error(rom, it, compute_errors)
        if rom.dim_input == rom.dim_output == 1:
            self.cauchy_indices.append(cauchy_index(rom))


class AlphaInterpReductor(BasicObject):
    """Do interpolation of an affine combination.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    rom
        Reduced-order |LTIModel|.
    sigma
        Interpolation points.
    b
        Right tangential directions.
    c
        Left tangential directions.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, rom, sigma, b, c, mu=None):
        assert isinstance(fom, LTIModel)
        self.__auto_init(locals())
        self._V = None
        self._W = None
        self._dE = None
        self._dA = None
        self._dB = None
        self._dC = None
        self.logger.setLevel('INFO')

    def _compute_gradients(self):
        Pt, Qt = solve_sylv_schur(
            self.fom.A,
            self.rom.A,
            self.fom.E,
            self.rom.E,
            self.fom.B,
            self.rom.B,
            self.fom.C,
            self.rom.C,
        )
        Pr = self.rom.gramian('c_dense')
        Qr = self.rom.gramian('o_dense')
        Er = to_matrix(self.rom.E, format='dense')
        Ar = to_matrix(self.rom.A, format='dense')
        Br = to_matrix(self.rom.B, format='dense')
        Cr = to_matrix(self.rom.C, format='dense')
        solve = lambda A, b: spla.solve(A, b)
        self._dE = Er - solve(Qr, solve(Pr, self.fom.E.apply2(Qt, Pt).T).T)
        self._dA = Ar - solve(Qr, solve(Pr, self.fom.A.apply2(Qt, Pt).T).T)
        self._dB = Br - solve(Qr, self.fom.B.apply_adjoint(Qt).to_numpy())
        self._dC = Cr - solve(Pr, self.fom.C.apply(Pt).to_numpy()).T

    def reduce(self, alpha):
        """Interpolate alpha * fom + (1 - alpha) * rom.

        Parameters
        ----------
        alpha
            Positive scalar.

        Returns
        -------
        Reduced-order |LTIModel|.
        """
        if any(mat is None for mat in [self._dE, self._dA, self._dB, self._dC]):
            self._compute_gradients()

        Er = to_matrix(self.rom.E, format='dense') - alpha * self._dE
        Ar = to_matrix(self.rom.A, format='dense') - alpha * self._dA
        Br = to_matrix(self.rom.B, format='dense') - alpha * self._dB
        Cr = to_matrix(self.rom.C, format='dense') - alpha * self._dC

        return LTIModel.from_matrices(Ar, Br, Cr, E=Er)


class IRKAWithLineSearchReductor(IRKACauchyIndexReductor):
    """Iterative Rational Krylov Algorithm with line search.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, mu=mu)
        self.alphas = []
        self.logger.setLevel('INFO')

    def reduce(
        self,
        rom0_params,
        alpha_min=1e-20,
        tol=1e-4,
        maxit=100,
        num_prev=1,
        force_sigma_in_rhp=False,
        conv_crit='sigma',
        compute_errors=False,
    ):
        r"""Reduce using IRKA with step size alpha.

        Parameters
        ----------
        rom0_params
            Can be:

            - order of the reduced model (a positive integer),
            - initial interpolation points (a 1D |NumPy array|),
            - dict with `'sigma'`, `'b'`, `'c'` as keys mapping to
              initial interpolation points (a 1D |NumPy array|), right
              tangential directions (|NumPy array| of shape
              `(len(sigma), fom.dim_input)`), and left tangential directions
              (|NumPy array| of shape `(len(sigma), fom.dim_input)`),
            - initial reduced-order model (|LTIModel|).

            If the order of reduced model is given, initial
            interpolation data is generated randomly.
        alpha_min
            Minimum step size.
        tol
            Tolerance for the convergence criterion.
        maxit
            Maximum number of iterations.
        num_prev
            Number of previous iterations to compare the current
            iteration to. Larger number can avoid occasional cyclic
            behavior of IRKA.
        force_sigma_in_rhp
            If `False`, new interpolation are reflections of the current
            reduced-order model's poles. Otherwise, only poles in the
            left half-plane are reflected.
        conv_crit
            Convergence criterion:

            - `'sigma'`: relative change in interpolation points
            - `'h2'`: relative :math:`\mathcal{H}_2` distance of
              reduced-order models
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of
            intermediate reduced-order models be computed.

            .. warning::
                Computing :math:`\mathcal{H}_2`-errors is expensive. Use
                this option only if necessary.

        Returns
        -------
        rom
            Reduced |LTIModel| model.
        """
        if self.fom.sampling_time > 0:
            raise NotImplementedError

        self._clear_lists()
        self.alphas = []
        sigma, b, c = self._rom0_params_to_sigma_b_c(rom0_params, force_sigma_in_rhp)
        rom = _poles_b_c_to_lti(-sigma, b, c)
        self._store_sigma_b_c(sigma, b, c)
        self._check_common_args(tol, maxit, num_prev, conv_crit)

        self.logger.info('Starting IRKA with line search')
        self._conv_data = (num_prev + 1) * [None]
        if conv_crit == 'sigma':
            self._conv_data[0] = sigma

        Pt = solve_sylv_schur(self.fom.A, rom.A, self.fom.E, rom.E, self.fom.B, rom.B)
        error_old = rom.h2_norm() ** 2 - 2 * np.trace(
            self.fom.C.apply(Pt).to_numpy().T @ rom.C.matrix.T
        )

        for it in range(maxit):
            msg = ''
            alpha = 1
            alpha_interp = AlphaInterpReductor(self.fom, rom, sigma, b, c, mu=self.mu)
            while alpha >= alpha_min:
                self.logger.info(f'{msg}Attempting alpha={alpha:.2e}')
                rom_new = alpha_interp.reduce(alpha)
                rom_new = lti_with_better_cond(rom_new)
                if self.fom.dim_input == self.fom.dim_output == 1:
                    if cauchy_index(rom) != cauchy_index(rom_new):
                        msg = 'Cauchy index changed; '
                        alpha /= 2
                        continue
                if rom_new.poles().real.max() < 0:
                    Pt_new = solve_sylv_schur(
                        self.fom.A,
                        rom_new.A,
                        self.fom.E,
                        rom_new.E,
                        self.fom.B,
                        rom_new.B,
                    )
                    error_new = rom_new.h2_norm() ** 2 - 2 * np.trace(
                        self.fom.C.apply(Pt_new).to_numpy().T @ rom_new.C.matrix.T
                    )
                    if error_new < error_old:
                        rom = rom_new
                        break
                    else:
                        msg = 'Error did not reduce; '
                else:
                    msg = 'ROM is unstable; '
                alpha /= 2
            else:
                self.logger.info('alpha too small')
                break
            sigma, b, c = self._rom_to_sigma_b_c(rom, force_sigma_in_rhp)
            self._store_sigma_b_c(sigma, b, c)
            self._update_conv_data(sigma, rom, conv_crit)
            self._compute_conv_crit(rom, conv_crit, it)
            self._compute_error(rom, it, compute_errors)
            self.alphas.append(alpha)
            if self.conv_crit[-1] < alpha * tol:
                break
            error_old = error_new

        return rom
