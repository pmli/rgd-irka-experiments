# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from pymor.models.iosys import LTIModel

from models import slicot_model
from reductors import IRKACauchyIndexReductor, IRKAWithLineSearchReductor
from tools import plot_combined, plot_fom, plot_rom, settings

settings()

# %% [markdown]
# # FOM

# %%
fom = slicot_model('CDplayer')

# %%
print(fom)

# %%
w = (1e-2, 1e4)
_ = plot_fom(fom, w)

# %% [markdown]
# # IRKA

# %%
r = 6
irka = IRKACauchyIndexReductor(fom)
A0 = np.diag(np.arange(-1, -r - 1, -1))
B0 = np.ones((r, fom.dim_input))
C0 = np.ones((fom.dim_output, r))
rom0 = LTIModel.from_matrices(A0, B0, C0)
rom = irka.reduce(rom0, compute_errors=True, conv_crit='h2')

# %%
_ = plot_rom(fom, rom, w, irka, 'IRKA')

# %%
print((fom - rom).h2_norm() / fom.h2_norm())

# %% [markdown]
# # IRKA with line search

# %%
irka_rgd = IRKAWithLineSearchReductor(fom)
rom_rgd = irka_rgd.reduce(rom0, compute_errors=True, conv_crit='h2')

# %%
_ = plot_rom(fom, rom_rgd, w, irka_rgd, 'RGD-IRKA')

# %%
print((fom - rom_rgd).h2_norm() / fom.h2_norm())

# %%
_ = plot_combined(irka, irka_rgd, 'cd_r6.pdf', h2_error_scale='log')
