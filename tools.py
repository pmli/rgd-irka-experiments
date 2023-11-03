import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.logger import set_log_levels
from pymor.models.iosys import LTIModel, _lti_to_poles_b_c


def settings():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['lines.markersize'] = 8

    set_log_levels(
        {
            'pymor.algorithms.gram_schmidt.gram_schmidt': 'ERROR',
            'pymor.reductors.basic.LTIPGReductor': 'ERROR',
        }
    )


def cauchy_index(m):
    """Compute Cauchy index.

    Parameters
    ----------
    m : LTIModel
        Model.

    Returns
    -------
    index : int
        Cauchy index.
    """
    if m.dim_input > 1 or m.dim_output > 1:
        raise ValueError('The model has to be SISO')

    poles, b, c = _lti_to_poles_b_c(m)
    residues = b * c

    index = 0
    for p, r in zip(poles, residues):
        if p.imag == 0:
            if r > 0:
                index += 1
            else:
                index -= 1

    return index


def lti_with_better_cond(m, cond_tol=1e4):
    """Return an LTIModel with a better condition numbers.

    Parameters
    ----------
    m : LTIModel
        Model.
    cond_tol : float
        Condition number tolerance.

    Returns
    -------
    lti : LTIModel
        New model.
    """
    E = to_matrix(m.E)
    U, s, Vh = spla.svd(E, lapack_driver='gesvd')
    if s[0] / s[-1] < cond_tol:
        return m
    A = to_matrix(m.A)
    B = to_matrix(m.B)
    C = to_matrix(m.C)
    s_sqrt = np.sqrt(s)
    A = U.T @ A @ Vh.T / s_sqrt / s_sqrt[:, np.newaxis]
    B = U.T @ B / s_sqrt[:, np.newaxis]
    C = C @ Vh.T / s_sqrt
    lti = LTIModel.from_matrices(A, B, C)
    return lti


def plot_with_inf(ax, start, y, color=None):
    """Plot a list of values containing infinities.

    Parameters
    ----------
    ax : matploxlib Axes
        Axis to draw on.
    start : int
        Starting index for the list.
    y : list
        List of values to draw.
    color : str, optional
        Color to use for plotting finite values
        (red crosses are used for infinities).
    """
    ax.plot(range(start, start + len(y)), y, '.-', color=color)
    for i, yi in enumerate(y):
        if not np.isfinite(yi):
            ax.plot(start + i, 1, 'rx')


def plot_fom(fom, w):
    """Plot FOM properties.

    Parameters
    ----------
    fom : LTIModel
        LTI system to plot.
    w : tuple, list, np.ndarray
        Frequencies to use for frequency-domain plots.

    Returns
    -------
    fig : matplotlib Figure
        Resulting figure.
    axs : array of matplotlib Axes
        Resulting axes.
    """
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    ax = axs[0]
    ax.plot(fom.poles().real, fom.poles().imag, '.')
    _ = ax.set_title('FOM poles')

    ax = axs[1]
    _ = fom.transfer_function.mag_plot(w, ax=ax)
    _ = ax.set_title('FOM magnitude plot')

    return fig, axs


def plot_rom(fom, rom, w, reductor, reductor_name):
    """Plot ROM properties.

    Parameters
    ----------
    fom : LTIModel
        Full-order model.
    rom : LTIModel
        Reduced-order model.
    w : tuple, list, np.ndarray
        Frequencies to use for frequency-domain plots.
    reductor : pyMOR reductor
        The reductor instance that was used to build the ROM.
    reductor_name : str
        The name of the reductor.

    Returns
    -------
    fig : matplotlib Figure
        Resulting figure.
    axs : array of matplotlib Axes
        Resulting axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(rom.poles().real, rom.poles().imag, '.')
    _ = ax.set_title(f'{reductor_name} ROM poles')

    ax = axs[0, 1]
    plot_with_inf(ax, 0, reductor.conv_crit)
    ax.set_yscale('log')
    ax.set_title(f'{reductor_name} convergence criterion')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axs[1, 0]
    _ = fom.transfer_function.mag_plot(w, ax=ax)
    _ = rom.transfer_function.mag_plot(w, ax=ax)
    _ = ax.set_title('FOM/ROM magnitude plot')

    ax = axs[1, 1]
    err = fom - rom
    _ = err.transfer_function.mag_plot(w, ax=ax)
    _ = ax.set_title('Error magnitude plot')

    ylim1 = axs[1, 0].get_ylim()
    ylim2 = axs[1, 1].get_ylim()
    ylim_tmp = tuple(zip(ylim1, ylim2))
    ylim = (min(ylim_tmp[0]), max(ylim_tmp[1]))
    axs[1, 0].set_ylim(ylim)
    axs[1, 1].set_ylim(ylim)

    return fig, axs


def plot_combined(irka, irka_rgd, name, h2_error_scale='linear', show_ci=True):
    """Plot results of IRKA and RGD-IRKA.

    Parameters
    ----------
    irka : IRKAReductor
        The IRKA reductor.
    irka_rgd : IRKAWithLineSearchReductor
        The RGD-IRKA reductor.
    name : str
        Name of the file to save the plot to.
    h2_error_scale : str
        Which scale to use for the y-axis in H2 error plots
        ('linear' or 'log').
    show_ci : bool
        Whether to show the Cauchy index plots (only for SISO systems).

    Returns
    -------
    fig : matplotlib Figure
        Resulting figure.
    axs : array of matplotlib Axes
        Resulting axes.
    """
    is_siso = irka.fom.dim_input == irka.fom.dim_output == 1
    show_ci = is_siso and show_ci

    subplots_opts = dict(sharex=True, sharey='row', constrained_layout=True)
    fig_width = 8
    if show_ci:
        fig_height = 7
        fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height), **subplots_opts)
    else:
        fig_height = 4
        fig, axs = plt.subplots(
            1, 2, figsize=(fig_width, fig_height), squeeze=False, **subplots_opts
        )

    ax = axs[0, 0]
    plot_with_inf(ax, 1, irka.errors)
    ax.set_yscale(h2_error_scale)
    ax.set_ylabel(r'Relative $\mathcal{H}_2$ Error')
    ax.set_title('IRKA')

    if show_ci:
        ax = axs[1, 0]
        ci = irka.cauchy_indices
        ax.plot(range(1, len(ci) + 1), ci, '.-')
        ax.set_ylabel('Cauchy Index')
        ci_rgd = irka_rgd.cauchy_indices
        if ci[:-1] == ci[1:] and ci_rgd[:-1] == ci_rgd[1:] and ci[0] == ci_rgd[0]:
            ax.set_yticks([ci[0]])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axs[0, 1]
    plot_with_inf(ax, 1, irka_rgd.errors)
    ax.set_yscale(h2_error_scale)
    ax.set_title('RGD-IRKA')

    ax = ax.twinx()
    color = 'tab:green'
    plot_with_inf(ax, 1, irka_rgd.alphas, color=color)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\alpha_k$', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_minor_locator(LogLocator(base=2))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=2, labelOnlyBase=True))

    if show_ci:
        ax = axs[1, 1]
        ax.plot(range(1, len(ci_rgd) + 1), ci_rgd, '.-')

    for ax in axs[-1]:
        ax.set_xlabel('Iteration')
        if len(irka.errors) < 20:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(name, bbox_inches='tight', pad_inches=0)

    return fig, axs


def savetxt(fname, columns, names=None):
    """Save columns to a text file.

    Parameters
    ----------
    fname : str
        File name.
    columns : sequence of lists of floats
        Columns to save.
    names : sequence of str (optional)
        Column names to write in the header.
    """
    columns = [np.asarray(c) for c in columns]
    X = np.column_stack(columns)
    header = '' if names is None else ' '.join(names)
    fmt = []
    for c in columns:
        if np.issubdtype(c.dtype, np.integer):
            fmt.append('%d')
        else:
            fmt.append('%.5e')
    np.savetxt(fname, X, fmt=fmt, header=header, comments='')
