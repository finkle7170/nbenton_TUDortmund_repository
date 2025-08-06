import numpy as np, gzip, pickle, sys, os
from pathlib import Path
import crflux.models as pm

# ---- MCEq setup ---------------------------------------------------------
import MCEq.config as config
from MCEq.core import MCEqRun
from MCEq.particlemanager import MCEqParticle   # noqa: F401

config.mceq_db_fname            = "mceq_db_fine_v150.h5"
config.decay_db_name            = "full_decays"
config.muon_helicity_dependence = True

mceq_run = MCEqRun(
    interaction_medium = "air",
    interaction_model  = "SIBYLL2.3d",
    primary_model      = (pm.HillasGaisser2012, "H3a"),
    theta_deg          = 0.0,
)
mceq_run.set_density_model(("MSIS00_IC", ("SouthPole", "October")))
mag = 3

# Re‑attach databases (safe for custom builds)
pman = mceq_run.pman
pman.set_cross_sections_db(mceq_run._int_cs)
pman.set_decay_channels(mceq_run._decays)
pman.set_interaction_model(mceq_run._int_cs, mceq_run._interactions)

# ---- zenith grid --------------------------------------------------------
cos_grid   = np.linspace(np.cos(np.deg2rad(90)), np.cos(np.deg2rad(0)), 21)
wanted_cos = (0.1, 0.5, 0.9)
tol        = 1e-6

# ---- containers ---------------------------------------------------------
tot_mu, tot_numu, tot_nue, tot_nutau         = [], [], [], []
conv_mu, conv_numu, conv_nue, conv_nutau     = [], [], [], []

data = {
    "avg": {k: None for k in (
        "mu","numu","nue","nutau",
        "mu_conv","numu_conv","nue_conv","nutau_conv")},
}
for c in wanted_cos:
    data[c] = {k: None for k in (
        "mu","numu","nue","nutau",
        "mu_conv","numu_conv","nue_conv","nutau_conv")}

# main loop that solves and fetches the data from mceq_run and stores it
for cosv in cos_grid:
    theta_deg = float(np.degrees(np.arccos(cosv)))
    mceq_run.set_theta_deg(theta_deg)
    mceq_run.solve()

    mu_t    = mceq_run.get_solution("total_mu+",   mag) + mceq_run.get_solution("total_mu-",   mag)
    numu_t  = mceq_run.get_solution("total_numu",  mag) + mceq_run.get_solution("total_antinumu",mag)
    nue_t   = mceq_run.get_solution("total_nue",   mag) + mceq_run.get_solution("total_antinue", mag)
    nutau_t = mceq_run.get_solution("total_nutau", mag) + mceq_run.get_solution("total_antinutau",mag)

    mu_c    = mceq_run.get_solution("conv_mu+",   mag) + mceq_run.get_solution("conv_mu-",   mag)
    numu_c  = mceq_run.get_solution("conv_numu",  mag) + mceq_run.get_solution("conv_antinumu",mag)
    nue_c   = mceq_run.get_solution("conv_nue",   mag) + mceq_run.get_solution("conv_antinue", mag)
    nutau_c = mceq_run.get_solution("conv_nutau", mag) + mceq_run.get_solution("conv_antinutau",mag)

    tot_mu.append(mu_t);       conv_mu.append(mu_c)
    tot_numu.append(numu_t);   conv_numu.append(numu_c)
    tot_nue.append(nue_t);     conv_nue.append(nue_c)
    tot_nutau.append(nutau_t); conv_nutau.append(nutau_c)

    for tgt in wanted_cos:
        if abs(cosv - tgt) < tol:
            d = data[tgt]
            d["mu"],       d["mu_conv"]       = mu_t,    mu_c
            d["numu"],     d["numu_conv"]     = numu_t,  numu_c
            d["nue"],      d["nue_conv"]      = nue_t,   nue_c
            d["nutau"],    d["nutau_conv"]    = nutau_t, nutau_c
            break

# ---- θ‑averages ---------------------------------------------------------
data["avg"]["mu"]        = np.mean(tot_mu,    axis=0)
data["avg"]["numu"]      = np.mean(tot_numu,  axis=0)
data["avg"]["nue"]       = np.mean(tot_nue,   axis=0)
data["avg"]["nutau"]     = np.mean(tot_nutau, axis=0)
data["avg"]["mu_conv"]   = np.mean(conv_mu,    axis=0)
data["avg"]["numu_conv"] = np.mean(conv_numu,  axis=0)
data["avg"]["nue_conv"]  = np.mean(conv_nue,   axis=0)
data["avg"]["nutau_conv"]= np.mean(conv_nutau, axis=0)

# ---- ratios -------------------------------------------------------------
for c in wanted_cos:
    for fl in ("mu","numu","nue","nutau"):
        data[c][f"{fl}_ratio"]       = data[c][fl]            / data["avg"][fl]
        data[c][f"{fl}_conv_ratio"]  = data[c][f"{fl}_conv"]  / data["avg"][f"{fl}_conv"]

# ---- energy grid --------------------------------------------------------
data["energy"] = mceq_run.e_grid

# ---- save ---------------------------------------------------------------
outfile = Path("FishPlot_data.pkl.gz")
with gzip.open(outfile, "wb") as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[OK] Saved fish‑plot data → {outfile.resolve()}")