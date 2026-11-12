# =============================================================================
# KINETIC MODEL OF AIR CO-GASIFICATION OF OLIVE POMACE AND RDF IN A BUBBLING FLUIDIZED BED (SINGLE CELL)
#   Bed: two perfectly mixed zones (bubble and emulsion phases) exchanging gas (Kunii-Levenspiel); homogeneous
#   O2-consuming reactions take place only in the bubble phase; volatiles are released in the emulsion; char inventory
#   and dolomite in the emulsion. Freeboard: isothermal plug-flow reactor. Devolatilization: instantaneous, with
#   elemental closure. Reaction network: 26 reactions checked against their primary sources (section 9).
#   Adopted model (LEVEL = "two_zone"): the gas-phase kinetics are not fitted; the only parameter fitted to gasification
#   data is the activity factor of the in-bed dolomite (eta = 1.5, Eq. D1), on the tar yields of the reference case.
#   The perfectly mixed bed (formulation A, LEVEL = "predictive" or "calibrated") is kept for the comparison of
#   Section 3.1 of the article. Options that are not part of the adopted model are switched off (section 2).
#   Sections 15-24: techno-economic and carbon-footprint assessment, optimization and uncertainty.
#   Section 25: figures of the article; section 26: tables and figures of the article (only output shown).
#   Experimental data are not modified.
# =============================================================================
# -----------------------------------------------------------------------------
# 0. IMPORTS
# -----------------------------------------------------------------------------
import json, time, warnings
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, nnls

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)

# -----------------------------------------------------------------------------
# 1. PATHS
# -----------------------------------------------------------------------------
def _find(fname):
    for p in [Path("data") / fname, Path("..") / "data" / fname, Path(fname)]:
        if p.exists():
            return p
    raise FileNotFoundError(fname)
DATA_OUT = Path("data") if Path("data").exists() else Path(".")

# --- Output control. ARTICLE_ONLY = True: the notebook shows only the tables and figures of the article
#     (section 26). Everything else that the cell prints goes to DATA_OUT/full_output.txt, and warnings are silenced.
#     ARTICLE_ONLY = False restores the full printed output. No calculation depends on this switch.
#     Figures: only the figures of the article are written (DATA_OUT/figures). SAVE_AUX = True also writes the
#     intermediate diagnostic figures of sections 11-24 to DATA_OUT/auxiliary.
ARTICLE_ONLY = True
SAVE_AUX = False
FIG_ART = DATA_OUT / "figures"; FIG_ART.mkdir(exist_ok=True)
AUX_OUT = DATA_OUT / "auxiliary"
from matplotlib.figure import Figure as _Figure
_savefig0 = _Figure.savefig
def _savefig_ctrl(self, fname, *a, **k):
    """Writes the figures of the article; the auxiliary figures only if SAVE_AUX is True."""
    p_ = Path(fname)
    if p_.parent.resolve() == FIG_ART.resolve():
        return _savefig0(self, p_, *a, **k)
    if not SAVE_AUX:
        return None
    if p_.parent.resolve() == DATA_OUT.resolve():
        p_ = AUX_OUT / p_.name
    p_.parent.mkdir(parents=True, exist_ok=True)
    return _savefig0(self, p_, *a, **k)
_Figure.savefig = _savefig_ctrl
if ARTICLE_ONLY:
    import sys as _sys
    _STDOUT = _sys.stdout
    _LOG = open(DATA_OUT / "full_output.txt", "w", encoding="utf-8")
    _sys.stdout = _LOG
    warnings.filterwarnings("ignore")
    plt.ioff()
    plt.show = lambda *a, **k: plt.close("all")

# -----------------------------------------------------------------------------
# 2. CONSTANTS AND CONFIGURATION DATACLASSES
# -----------------------------------------------------------------------------
# Normal conditions used everywhere: T_N = 273.15 K, P_N = 101325 Pa (1 Nm3 = 44.615 mol).
R_GAS = 8.314462618
P_ATM = 101325.0
NM3_T, NM3_P = 273.15, P_ATM
MOL_PER_NM3 = NM3_P / (R_GAS * NM3_T)
AIR_O2 = 0.21
AW = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}
LHV_NM3 = {"H2": 10.78, "CO": 12.63, "CH4": 35.80, "C2H4": 59.0, "C3H6": 87.6}

@dataclass
class FeedstockComponent:
    name: str
    ultimate_dry: Dict[str, float]
    moisture_ar: float
    VM_ar: float
    FC_ar: float
    ash_ar: float
    HHV_MJ_kg: float
    source: str = ""
    devol_surrogate: Optional[Dict[str, float]] = None
    f_vol_in_bed: float = 0.0
    char_steam: Optional[Dict[str, float]] = None

@dataclass
class FeedstockConfig:
    components: Dict[str, FeedstockComponent]
    mass_fractions: Dict[str, float]
    name: str = "blend"

@dataclass
class ReactorConfig:
    D_m: float
    H_total_m: float
    bed_mass_kg: float
    bed_material: str
    dp_bed_m: float
    rho_p_kg_m3: float
    eps_mf: float
    n_orifices: int
    tau_char_s: float
    name: str = "reactor"
    tar_catalyst: Optional[dict] = None

@dataclass
class OperatingPoint:
    T_C: float
    P_gauge_kPa: float
    feed_kg_h: float
    air_Nm3_h: float
    air_T_C: float = 20.0
    T_fb_out_C: Optional[float] = None    # measured temperature at the freeboard outlet (C), if reported
    fb_profile: Optional[tuple] = None    # measured profile: (heights above the distributor in m, temperatures in C)

@dataclass
class SimulationOptions:
    level: int = 3
    T_freeboard_offset_K: float = 0.0
    rtol: float = 1e-6
    t_end_factor: float = 12.0
    component_char_kinetics: bool = True
    devol_rdf_measured: bool = False   # measured SRF volatile distribution (Tokmurzin 2022) for the RDF components
    co_ox_bed_factor: float = 1.0      # calibrated level: factor of the CO oxidation law (H-R04) inside the bed only
    wgs_in_freeboard: bool = True      # False: no water-gas shift in the solids-free freeboard (option, off)
    measured_freeboard: bool = False   # measured freeboard temperatures (where published) instead of T_bed everywhere
    wood_char_kinetics: bool = False   # Senneca 2007 wood-chip char kinetics (CO2, O2) for pine and eucalyptus chars
    devol_wood_measured: bool = False  # measured wood pyrolysis yields (Heidari 2014) for pine and eucalyptus
    devol_olive_measured: bool = False # measured olive husk pyrolysis yields (Christoforou 2018) for olive pomace
    devol_hc_lump: bool = False        # literature CH4:C2 ratio for the hydrocarbon closure of olive residues
    devol_h2_closure: bool = True      # False: H2 is not an element-closure species of the devolatilisation
    fuel_char_kinetics: bool = False   # literature char kinetics per fuel (CHAR_LIT), where a source exists
    wgs_equilibrium: bool = True
    wgs_dT_K: float = 200.0
    o2_priority: float = 1.0
    fb_cool: Optional[tuple] = None    # (a, b): freeboard outlet T (C) = a * T_bed(C) + b, linear profile (measured thermocouples)
    wgs_final: bool = True             # impose WGS equilibrium again at the freeboard outlet
    bed_model: str = "cstr"            # "cstr" (perfectly mixed bed) | "two_phase" (1-D three-phase bed, not used) | "two_zone" (adopted)
    tz_volatiles_to: str = "emulsion"  # two_zone bed only: phase that receives the volatiles ("emulsion" | "bubble")
    tz_vol_emulsion_frac: float = -1.0 # two_zone bed only: share of the volatiles released in the emulsion; < 0 -> use tz_volatiles_to
    tz_kbe_factor: float = 1.0         # two_zone bed only: multiplier of the bubble-emulsion interchange coefficient (uncertainty analysis)
    bubble_homogeneous: bool = True    # 1-D three-phase bed only: homogeneous reactions also inside the bubble phases


# -----------------------------------------------------------------------------
# 3. SPECIES / NETWORK (stoichiometry from the JSON files; kinetic table below is the ONLY kinetic source)
# -----------------------------------------------------------------------------
with open(_find("active_network.json")) as f:
    NET = json.load(f)
with open(_find("species_registry.json")) as f:
    REG = json.load(f)

SPECIES = NET["species_list"]
IDX = {s: i for i, s in enumerate(SPECIES)}
NSP = len(SPECIES)
MW = np.array([NET["species_MW"][s] for s in SPECIES])
IS_SOLID = np.array([NET["species_phase"][s] == "solid" for s in SPECIES])
GAS = [s for s in SPECIES if not IS_SOLID[IDX[s]]]
I_CS = IDX["C_s"]
ELEMENTS = ["C", "H", "O", "N"]
ATOM = np.array([[REG["species_info"][s]["atoms"].get(e, 0) for e in ELEMENTS]
                 for s in SPECIES], dtype=float)
TAR_SPECIES = ["C6H6", "C7H8", "C6H6O", "C10H8", "Tar"]
PERMANENT_GAS = [s for s in GAS if s not in TAR_SPECIES + ["H2O"]]

RXN_IDS = list(NET["active_reactions"])
NU = np.array(NET["N_active_matrix"], dtype=float)

# H-R11 stoichiometric correction (JSON destroys C and H)
_i = RXN_IDS.index("H-R11")
NU[_i, :] = 0.0
NU[_i, IDX["Tar"]], NU[_i, IDX["C6H6"]] = -1.0, 1.0
NU[_i, IDX["C_s"]], NU[_i, IDX["H2"]] = 4.0, 1.0

# In-bed calcined-dolomite catalytic tar elimination (Orio 1997 Table 4)
DOL_RXNS = {"D-R01": ("C10H8", {"C10H8": -1, "H2O": -10, "CO": 10, "H2": 14}),
            "D-R02": ("Tar",   {"Tar": -1, "H2O": -10, "CO": 10, "H2": 14}),
            "D-R03": ("C7H8",  {"C7H8": -1, "H2O": -7, "CO": 7, "H2": 11}),
            "D-R04": ("C6H6O", {"C6H6O": -1, "H2O": -5, "CO": 6, "H2": 8})}
for _rid, (_sp, _st) in DOL_RXNS.items():
    RXN_IDS.append(_rid)
    _row = np.zeros(NSP)
    for _k, _v in _st.items():
        _row[IDX[_k]] = _v
    NU = np.vstack([NU, _row])


# Kinetic table (Sebastiani 2025 SI, Tables S5-S6; checked line by line against the SI).
# NOTE: 'kinetic_parameters' inside active_network.json is NOT used (exponent signs and rate expressions there
#       disagree with the SI); section 19 lists the mismatches and exports the network actually used.
LAW = {
    "H-R01": dict(A=2.778,     Tn=0, Ea=12560,  fwd={"CO": 1, "H2O": 1},       rev={"CO2": 1, "H2": 1}, keq=(0.022,  34730.0)),
    "H-R02": dict(A=4.916e-10, Tn=2, Ea=36150,  fwd={"H2O": 1, "CH4": 1},      rev={"CO": 1, "H2": 3},  keq=(3.106e14, -208800.0)),
    "H-R03": dict(A=1.08e10,   Tn=0, Ea=125525, fwd={"O2": 1, "H2": 1}),
    "H-R04": dict(A=1.78e10,   Tn=0, Ea=180032, fwd={"CO": 1, "H2O": 0.5, "O2": 0.25}),
    "H-R05": dict(A=1.58e10,   Tn=0, Ea=202641, fwd={"CH4": 0.7, "O2": 0.8}),
    "H-R06": dict(A=2.06e6,    Tn=0, Ea=202500, fwd={"CH4": 0.3, "O2": 1.3}),
    "H-R07": dict(A=1.58e12,   Tn=0, Ea=202641, fwd={"C6H6": 1, "O2": 1}),
    "H-R08": dict(A=4.4e5,     Tn=0, Ea=220000, fwd={"C6H6": 1}),
    "H-R09": dict(A=9.97e10,   Tn=0, Ea=324000, fwd={"C10H8": 1, "H2": 0.4}),
    "H-R11": dict(A=4.0e4,     Tn=0, Ea=76600,  fwd={"Tar": 1}),
    "H-R12": dict(A=2.4e12,    Tn=0, Ea=125590, fwd={"C2H4": 0.1, "O2": 1.65}),
    "H-R13": dict(A=2.4e12,    Tn=0, Ea=125590, fwd={"C2H4": 0.1, "O2": 1.65}),
    "H-R14": dict(A=0.12,      Tn=0, Ea=17111,  fwd={"C2H4": 1, "H2O": 1}),
    "H-R15": dict(A=5.0e11,    Tn=0, Ea=125590, fwd={"C3H6": -0.1, "O2": 1.85}),
    "H-R16": dict(A=1.3e12,    Tn=0, Ea=125590, fwd={"C3H6": 0.1, "O2": 1.65}),
    "H-R17": dict(A=0.12,      Tn=0, Ea=17111,  fwd={"C3H6": 1, "H2O": 1}),
    "H-R18": dict(A=3.3e10,    Tn=0, Ea=247000, fwd={"C7H8": 1, "H2": 0.5}),
    "H-R19": dict(A=1.0e7,     Tn=0, Ea=100000, fwd={"C6H6O": 1}),
    "H-R20": dict(A=1.0e7,     Tn=0, Ea=100000, fwd={"C6H6O": 1, "H2O": 3}),
    "X-R01": dict(A=4.89e7,    Tn=0, Ea=268000, fwd={"CO2": 1}, basis="char",
                  inhib=[(6.60e-2, 0.0, "CO2"), (1.2e-1, 25500.0, "CO")]),
    "X-R02": dict(A=2.39e2,    Tn=0, Ea=129000, fwd={"H2O": 1}, basis="char",
                  inhib=[(3.16e-2, 30100.0, "H2O"), (5.36e-3, 59800.0, "H2"), (8.25e-5, 96100.0, "CO")]),
    "X-R03": dict(A=5.957e2,   Tn=1, Ea=149440, fwd={"O2": 1.0}, basis="char_surface"),
}
# Oxygen coefficient of the fuel LHV formula (ISO 18125: LHV = HHV - 0.212 H - 0.0245 M - 0.0008 O, MJ/kg, % ar).
LHV_O_COEF = 0.0008
DOLOMITE_ORIO = dict(k0_m3_kg_h=1.51e6, Ea=100e3)
# In-bed dolomite tar conversion. Orio et al. (1997) measured the intrinsic rate on crushed
# dolomite (k0 = 1.51e6 m3/(kg h), Ea = 100 kJ/mol). The calcined dolomite of the bed (dp ~ 0.5-1 mm) works in the
# strong pore-diffusion regime, where the observed activation energy is half the intrinsic one (Levenspiel, Chemical
# Reaction Engineering, 3rd ed., 1999, ch. 18):  Ea_app = Ea / 2 = 50 kJ/mol.
#   k_app(T) = eta * k_Orio(T_ref) * exp[-Ea_app / R * (1/T - 1/T_ref)],  T_ref = 800 C            (Eq. D1)
# eta (activity factor of the bed dolomite relative to Orio's sample) is the only parameter fitted to gasification
# data: eta = 1.5 (least squares on the five tar yields of the reference case, Terrados Table 4; fit 1.49, tar RMSE
# 3.5 g/kg). It is frozen for every other dataset; Carmo-Calado (dolomite bed) is the independent test. Beds of sand,
# olivine or ofite have W_dol = 0 and are not affected. DOL_APPARENT = False uses the intrinsic law of Orio et al.
DOL_APPARENT = True
DOL_EA_APP, DOL_ETA, DOL_T_REF = 50.0e3, 1.5, 1073.15
for _rid, (_sp, _st) in DOL_RXNS.items():
    _A0, _E0 = DOLOMITE_ORIO["k0_m3_kg_h"] / 3600.0, DOLOMITE_ORIO["Ea"]
    if DOL_APPARENT:
        _A0, _E0 = DOL_ETA * _A0 * np.exp((DOL_EA_APP - _E0) / (R_GAS * DOL_T_REF)), DOL_EA_APP
    LAW[_rid] = dict(A=_A0, Tn=0, Ea=_E0, fwd={_sp: 1}, basis="dolomite")

CHAR_PROPS = {"dp_char_m": 1.0e-3, "rho_char_kg_m3": 400.0}
REVERSIBLE_MODE = "source_Keq"

def char_oxidation_alpha(T_K):
    fr = 4.72e-3 * np.exp(37737.0 / (R_GAS * T_K))
    return (1.0 + 2.0 * fr) / (1.0 + fr)

_I_XR03 = RXN_IDS.index("X-R03")
_I_XR02 = RXN_IDS.index("X-R02")
_I_XR01 = RXN_IDS.index("X-R01")
_I_HR11 = RXN_IDS.index("H-R11")

def nu_matrix(T_K):
    N = NU.copy()
    a = char_oxidation_alpha(T_K)
    N[_I_XR03, :] = 0.0
    N[_I_XR03, IDX["C_s"]] = -a
    N[_I_XR03, IDX["O2"]]  = -1.0
    N[_I_XR03, IDX["CO"]]  = 2.0 * (a - 1.0)
    N[_I_XR03, IDX["CO2"]] = 2.0 - a
    return N

@dataclass
class KineticParameterSet:
    A: np.ndarray
    T_exp: np.ndarray
    Ea: np.ndarray
    multiplier: np.ndarray
    reversible_mode: str = "source_Keq"
    char_steam: Optional[list] = None
    char_steam_fallback_w: float = 1.0
    fwd_tab: Optional[tuple] = None     # corrected reaction-order table (None = orders of the LAW table)
    char_lit: Optional[dict] = None     # fuel-specific literature char kinetics {'CO2': [...], 'O2': [...]}
    char_lit_fallback_w: float = 1.0    # share of the fixed carbon that keeps the generic (Table S6) kinetics

def build_kinetic_set():
    return KineticParameterSet(
        np.array([LAW[r]["A"] for r in RXN_IDS], float),
        np.array([LAW[r]["Tn"] for r in RXN_IDS], float),
        np.array([LAW[r]["Ea"] for r in RXN_IDS], float),
        np.ones(len(RXN_IDS)), REVERSIBLE_MODE)

KPS = build_kinetic_set()

# --- Unit correction of the olefin oxidation laws (H-R12, H-R13, H-R15, H-R16) ---------------------------------------
# Source: Westbrook & Dryer (1981), Combust. Sci. Technol. 27, 31-43, DOI 10.1080/00102208108946970, Tables I and IV
#         ("Units are cm-sec-mole-kcal-Kelvins"; Table IV: "Same units as Table I"). Two-step mechanism, fuel -> CO + H2O:
#         C2H4  A = 2.4e12, Ea = 30 kcal/mol, a = 0.1, b = 1.65 ;  C3H6  A = 5.0e11, Ea = 30, a = -0.1, b = 1.85 ;
#         (C2H6 A = 1.3e12, a = 0.1, b = 1.65).  Tokmurzin et al. (2022), Energy Convers. Manag. 267, 115925, Table 3
#         (R18-R22) reproduce these numbers under the heading "mol m-3 s-1", and Sebastiani 2025 SI Table S5 (r12-r16)
#         copies them with concentrations in mol/m3. Their own Fig. 9d/f shows oxidation rates of ~1e8 kmol/(m3 s).
# Equation: r[mol/cm3/s] = A exp(-Ea/RT) [F]^a [O2]^b with [ ] in mol/cm3  ->  with [ ] in mol/m3 and r in mol/m3/s:
#           A_SI = A_cgs * 10^(6 (1 - a - b)).  Here a + b = 1.75 for the four laws -> factor 10^-4.5 = 3.16e-5.
# Units: SI after conversion.  Cross-check inside the same table: r4 (CO oxidation, overall order 1.75) carries
#         A = 1.78e10, i.e. the SI-converted magnitude of the Dryer-Glassman value 10^14.6 quoted in W&D Eq. (10).
# Validity: W&D fitted these laws to laminar flame speeds (flame-zone temperatures 1450-2000 K, 1-25 atm); their use in
#         a 700-1100 C gasifier is an extrapolation already made by the source model. Not changed here.
# Reason for selection: correction of a verified unit error, NOT a fit.
# Not corrected here (flagged): r12 and r13 both carry the two-step C2H4 value (W&D give 4.3e12 for fuel -> CO + H2);
#         r16 (C3H6 -> CO2) carries the C2H6 two-step parameters (1.3e12, 0.1, 1.65) instead of C3H6 ones.
UNIT_FIX = {r: 10.0 ** (6.0 * (1.0 - sum(LAW[r]["fwd"].values()))) for r in ("H-R12", "H-R13", "H-R15", "H-R16")}
_mult_si = KPS.multiplier.copy()
for _r, _f in UNIT_FIX.items():
    _mult_si[RXN_IDS.index(_r)] = _f
KPS_SI = replace(KPS, multiplier=_mult_si)


# -----------------------------------------------------------------------------
# 4-5. EXPERIMENTAL DATA AND FEEDSTOCKS (per dataset, as published)
# -----------------------------------------------------------------------------

SUR_EUCALYPTUS = {"H2": 1.5, "H2O": 10.8, "CO": 10.7, "CO2": 18.5, "CH4": 5.27, "C2H4": 0.64,
                  "C3H6": 0.64, "C6H6": 0.19, "C7H8": 6.51, "C6H6O": 34.5, "C10H8": 2.91, "Tar": 4.8}
SUR_BLU_L = {"H2": 0.5, "H2O": 0.52, "CO": 0.0, "CO2": 0.0, "CH4": 17.4, "C2H4": 32.8, "C3H6": 9.8,
             "C6H6": 13.19, "C7H8": 4.94, "C6H6O": 0.0, "C10H8": 1.31, "Tar": 20.02}
SUR_BLU_C = {"H2": 0.46, "H2O": 0.48, "CO": 0.0, "CO2": 0.0, "CH4": 15.97, "C2H4": 30.10,
             "C3H6": 8.99, "C6H6": 12.11, "C7H8": 4.53, "C6H6O": 0.0, "C10H8": 1.20, "Tar": 23.9}
def mix_surrogates(parts):
    out = {}
    for w, sur in parts:
        for k, v in sur.items():
            out[k] = out.get(k, 0.0) + w * v
    return out
SUR_RDF_TERRADOS = mix_surrogates([(0.35, SUR_EUCALYPTUS), (0.325, SUR_BLU_L), (0.325, SUR_BLU_C)])
# Fuel-specific char-steam kinetics (X-R02). INACTIVE in the reference model; switch: USE_FUEL_CHAR_STEAM (section 10).
# Both parameter sets were checked against their sources and match them.
# Olive pomace char
#   Source: Prestipino, Galvagno, Karlstrom & Brink (2018), Energy 161, 1055-1063, DOI 10.1016/j.energy.2018.07.205,
#           Table 3, column "OP": E = 150 kJ/mol, A = 9.50e4 atm^-n s^-1, n = 1.0 (R2 = 0.93, the poorest fit of the six chars).
#   Equation: k = A exp(-E/RT) p_H2O^n  [1/s], p in atm (Eq. 2 of the source).
#   Validity: thermogravimetric tests at 650-725 C and 0.1-0.5 atm of steam; char made by pyrolysis at 500 C for 1 h in N2.
#           Here it would be EXTRAPOLATED to 700-1100 C. The source fits X(t) = k t or X = 1 - exp(-k t) depending on the
#           ash composition; this model applies k per mole of char in the bed inventory (first order in the remaining char).
# RDF char
#   Source: Le & Kolaczkowski (2015), Chem. Eng. Res. Des. 102, 389-398, DOI 10.1016/j.cherd.2015.07.004.
#           Table 3 (uniform-reaction model, 50 % conversion): E = 105 kJ/mol, A = 3.84e4 bar^-n s^-1.
#           n = 1.56 is the mean of the orders at 50 % conversion in Table 1 (1.579, 1.732, 1.382 at 900, 850, 800 C;
#           shrinking-core fit): a modelling choice, flagged.
#   Equation: (1/(1-X)) dX/dt = A exp(-E/RT) p_H2O^n  [1/s], p in bar (first order in the remaining carbon).
#   Validity: packed bed, 800-900 C, RDF-derived char of 250-500 um, atmospheric pressure.
# Why inactive: with them the fixed carbon is converted far more than the measurements allow, and the validation of the
#   model (Section 3.1 of the article) was made without them. Activating them gives a different model.
CHAR_STEAM_OP  = dict(A=9.50e4, n=1.0,  Ea=150e3, P_ref_Pa=101325.0)
CHAR_STEAM_RDF = dict(A=3.84e4, n=1.56, Ea=105e3, P_ref_Pa=1.0e5)

# --- Terrados-Cristos 2026 ---
OLIVE_POMACE = FeedstockComponent(
    "olive_pomace", dict(C=52.1, H=6.7, N=3.2, S=0.0, O=30.3, Ash=7.7),
    9.4, 66.5, 16.4, 7.7, 20.5, "EXP-TERRADOS Table 3",
    devol_surrogate=SUR_EUCALYPTUS, char_steam=CHAR_STEAM_OP)
RDF = FeedstockComponent(
    "RDF", dict(C=35.7, H=6.4, N=0.2, S=0.0, O=35.3, Ash=26.2),
    0.6, 64.7, 8.5, 26.2, 15.7, "EXP-TERRADOS Table 3",
    devol_surrogate=SUR_RDF_TERRADOS, char_steam=CHAR_STEAM_RDF, f_vol_in_bed=0.65)
BLEND_TERRADOS = FeedstockConfig(
    {"olive_pomace": OLIVE_POMACE, "RDF": RDF},
    {"olive_pomace": 0.70, "RDF": 0.30}, "OP70_RDF30")
REACTOR_TERRADOS = ReactorConfig(
    0.500, 4.550, 50.0, "dolomite", 1.0e-3, 2850.0, 0.42, 36, 7200.0,
    "Terrados_BFB", tar_catalyst=DOLOMITE_ORIO)
TABLE4 = pd.DataFrame({
    "T_C": [700, 800, 900, 1000, 1100],
    "P_gauge_kPa": [-32.1, -30.9, -30.2, -29.1, -28.7],
    "air_Nm3_h": [87.9, 89.2, 89.9, 90.2, 91.4],
    "air_T_C": [21, 22, 20, 21, 22], "feed_kg_h": [50.0]*5,
    "ER": [0.28, 0.28, 0.29, 0.29, 0.29],
    "Q_syngas_Nm3_h": [91.2, 91.55, 92.15, 93.45, 94.65],
    "CO2": [12.9, 11.8, 10.7, 10.2, 10.9], "C2H4": [2.2, 2.1, 2.1, 1.7, 1.3],
    "C2H6": [0.7, 0.5, 0.3, 0.2, 0.2], "C2H2": [0.4, 0.1, 0.1, 0.1, 0.0],
    "N2": [50.7, 50.9, 51.2, 53.0, 55.8], "CH4": [4.1, 3.8, 3.2, 2.8, 2.7],
    "CO": [16.2, 17.8, 17.9, 16.7, 16.2], "H2": [12.6, 12.8, 14.3, 15.2, 12.8],
    "LHV": [6.9, 6.7, 6.5, 6.0, 5.3], "tar_g_kg": [47, 41, 37, 34, 33],
    "char_g_kg": [267, 262, 242, 241, 234], "CGE": [69, 65, 63, 58, 53]}
).set_index("T_C")

def air_n2_consistent(r, fs):
    fe = feed_elements(fs, r.feed_kg_h)
    n2_fuel = fe["el"]["N"] / 2.0 / MOL_PER_NM3 * 3600.0
    return (r.N2 / 100.0 * r.Q_syngas_Nm3_h - n2_fuel) / 0.79

# --- Calado 2023 ---
OLIVE_POMACE_CALADO = FeedstockComponent(
    "olive_pomace_calado", dict(C=53.7, H=7.1, N=2.1, S=0.1, O=28.6, Ash=8.4),
    4.5, 58.4, 28.7, 8.4, 20.1, "EXP-CALADO Table 1",
    devol_surrogate=SUR_EUCALYPTUS, char_steam=CHAR_STEAM_OP)
FEED_CALADO = FeedstockConfig({"op": OLIVE_POMACE_CALADO}, {"op": 1.0}, "OP100_Calado")
REACTOR_CALADO = replace(REACTOR_TERRADOS, H_total_m=4.150, name="Calado_BFB")
TABLE_CALADO = pd.DataFrame({
    "T_C": [700, 750, 800],
    "P_gauge_kPa": [-22.6, -32.1, -30.9],
    "air_Nm3_h": [163.47, 165.87, 167.12], "air_T_C": [20, 20, 20],
    "feed_kg_h": [100.0]*3,
    "Q_syngas_Nm3_h": [248, 250, 252],
    "CO2": [11.74, 12.41, 12.87], "C2H4": [2.73, 1.71, 1.11],
    "C2H6": [0.74, 0.65, 0.62], "C2H2": [0.16, 0.05, 0.04],
    "N2": [53.17, 53.72, 54.02], "CH4": [4.34, 3.89, 2.47],
    "CO": [14.57, 14.78, 15.88], "H2": [12.54, 12.78, 12.97],
    "LHV": [6.93, 6.10, 5.37], "tar_g_Nm3": [5.86, 5.32, 4.80],
    "char_g_kg": [38, 35, 33], "CGE": [89.51, 79.43, 70.48]}
).set_index("T_C")

# --- Pio 2020 ---
PINE_PELLETS = FeedstockComponent(
    "pine_pellets", dict(C=47.5, H=6.2, N=0.1, S=0.0, O=45.9, Ash=0.3),
    4.6, 78.5, 16.6, 0.3, 18.0 + 2.442*9*6.2/100, "EXP-PIO Table 1", devol_surrogate=SUR_EUCALYPTUS)
PINE_CHIPS = FeedstockComponent(
    "pine_chips", dict(C=46.4, H=6.6, N=0.2, S=0.0, O=46.5, Ash=0.3),
    11.0, 77.9, 10.8, 0.3, 18.8 + 2.442*9*6.6/100, "EXP-PIO Table 1", devol_surrogate=SUR_EUCALYPTUS)
RDF_PIO = FeedstockComponent(
    "RDF_pio", dict(C=54.0, H=7.4, N=0.5, S=0.0, O=24.1, Ash=13.4),
    4.3, 75.2, 7.1, 13.4, 24.8 + 2.442*9*7.4/100, "EXP-PIO Table 1",
    devol_surrogate=SUR_RDF_TERRADOS, char_steam=CHAR_STEAM_RDF)
REACTOR_PIO = ReactorConfig(0.25, 2.3, 17.5, "silica_sand", 700e-6, 2600.0, 0.45, 57, 1800.0,
                            "Aveiro_80kW", tar_catalyst=None)
_AIR_PIO = 200.0 * 60 / 1000.0
# Pio et al. (2020), Energy Convers. Manag. 206, 112476. Operating conditions: Table 3 (bed temperature, feed rate, air
# 200 L/min at 0 C and 1.013e5 Pa). Dry gas composition (vol%, N2 included in the basis): Figs. 6 and 7, read from the
# VECTOR drawing of the PDF (marker centres; reading error ~ +/-0.05 vol%; checked against the values quoted in the text:
# CO 18.6 for PC100:ER023, CH4 4.5 / C2H4 1.8 for PP100:ER022, CH4 5.6 / C2H4 5.0 for RDF100:ER027).
# 15 of the 19 experiments are used. NOT used: PC90-RDF10 (ER 0.22 and 0.25) and PC80-RDF20 (ER 0.22 and 0.25), because
# the figure shows two markers at the same abscissa and they cannot be assigned to one ER or the other.
# LHV: only the three values quoted in the text of the article.
def _pio_feed(bio, w_rdf):
    comp = {"pine_chips": PINE_CHIPS} if bio == "PC" else {"pine_pellets": PINE_PELLETS}
    key = next(iter(comp))
    if w_rdf <= 0.0:
        return FeedstockConfig(comp, {key: 1.0}, f"{bio}100")
    if w_rdf >= 1.0:
        return FeedstockConfig({"RDF_pio": RDF_PIO}, {"RDF_pio": 1.0}, "RDF100")
    return FeedstockConfig({**comp, "RDF_pio": RDF_PIO}, {key: 1.0 - w_rdf, "RDF_pio": w_rdf}, f"{bio}{100 - int(100 * w_rdf)}_RDF{int(100 * w_rdf)}")
FEED_PP, FEED_PC, FEED_RDFP = _pio_feed("PP", 0.0), _pio_feed("PC", 0.0), _pio_feed("PP", 1.0)
#             label                 bio  RDF   T   feed   H2     CO     CO2    CH4   C2H4  C2H6  C3H8   LHV
_PIO_ROWS = [("PC100:ER023",        "PC", 0.0, 803, 14, 6.50, 18.60, 15.93, 5.29, 2.17, 0.27, 0.05, np.nan),
             ("PC100:ER031",        "PC", 0.0, 806, 10, 5.70, 13.67, 16.60, 3.93, 1.55, 0.08, 0.02, np.nan),
             ("PC90-RDF10:ER030",   "PC", 0.1, 807, 10, 3.99, 13.25, 16.44, 3.78, 1.77, 0.09, 0.02, np.nan),
             ("PC80-RDF20:ER031",   "PC", 0.2, 811,  9, 4.90, 12.60, 15.76, 3.79, 2.06, 0.09, 0.03, np.nan),
             ("PC50-RDF50:ER032",   "PC", 0.5, 819,  8, 4.83, 13.79, 15.40, 4.00, 2.03, 0.13, 0.05, np.nan),
             ("PP100:ER022",        "PP", 0.0, 791, 13, 7.11, 15.51, 15.36, 4.46, 1.82, 0.16, 0.02, np.nan),
             ("PP100:ER030",        "PP", 0.0, 829, 10, 5.05, 11.36, 15.58, 2.96, 1.17, 0.05, 0.00, 3.7),
             ("PP90-RDF10:ER022",   "PP", 0.1, 797, 13, 6.53, 14.20, 15.71, 4.12, 2.09, 0.25, 0.04, np.nan),
             ("PP90-RDF10:ER031",   "PP", 0.1, 816,  9, 6.73, 13.92, 16.00, 3.57, 1.57, 0.11, 0.02, np.nan),
             ("PP80-RDF20:ER022",   "PP", 0.2, 801, 12, 6.90, 13.91, 15.49, 4.40, 2.34, 0.16, 0.03, np.nan),
             ("PP80-RDF20:ER031",   "PP", 0.2, 806,  9, 5.28, 11.39, 15.76, 3.50, 2.13, 0.13, 0.03, np.nan),
             ("PP50-RDF50:ER021",   "PP", 0.5, 812, 12, 5.38, 11.74, 16.44, 4.69, 3.70, 0.24, 0.05, np.nan),
             ("PP50-RDF50:ER030",   "PP", 0.5, 818,  8, 5.79, 11.57, 15.01, 4.05, 2.85, 0.16, 0.03, np.nan),
             ("RDF100:ER023",       "PP", 1.0, 818,  9, 5.15,  6.88, 15.23, 5.14, 4.29, 0.14, 0.09, 5.8),
             ("RDF100:ER027",       "PP", 1.0, 793,  8, 4.79,  7.26, 14.73, 5.64, 4.96, 0.20, 0.04, 6.4)]
# Measured axial temperature profile of the Aveiro reactor: Pio et al. (2020), Fig. 3 (vector lines, experiment RDF100:ER027),
# thermocouples T-2 ... T-9 at 0.18, 0.30, 0.45, 0.66, 0.86, 1.20, 1.68 and 2.90 m above the distributor: 805, 798, 769, 750,
# 743, 702, 654 and 395 C (time averages). Bed = mean of T-2 and T-3 (801.6 C). Fig. 4 (raster) shows the same shape for the
# other experiments, so the DIFFERENCE to the bed temperature is applied to every experiment (flagged assumption).
# The fuel is fed at 0.30 m; T-9 (2.90 m) is in the exhaust duct, above the 2.3 m of the reactor.
PIO_FB_Z = (0.45, 0.66, 0.86, 1.20, 1.68, 2.90)
PIO_FB_DT = (-32.6, -51.2, -59.0, -100.0, -148.1, -406.3)
def pio_fb_profile(T_bed_C):
    return (PIO_FB_Z, tuple(T_bed_C + d for d in PIO_FB_DT))
PIO_POINTS = [dict(T=T_, P=0.0, air=_AIR_PIO, feed=float(f_), feed_cfg=_pio_feed(b_, w_), label=l_,
                   exp=dict(H2=h2, CO=co, CO2=co2, CH4=ch4, C2H4=c2h4, C2H6=c2h6, C3H8=c3h8, LHV=lhv))
              for l_, b_, w_, T_, f_, h2, co, co2, ch4, c2h4, c2h6, c3h8, lhv in _PIO_ROWS]

# --- FABER (Parrillo 2023) ---
_M_EU = 11.2
EUCALYPTUS_FABER = FeedstockComponent(
    "eucalyptus_chips",
    {k: v / (1 - _M_EU/100) for k, v in dict(C=43.0, H=4.6, N=0.35, S=0.0, O=36.85, Ash=3.55).items()},
    _M_EU, 100 - _M_EU - 3.55 - 19.2, 19.2, 3.55, 14.85, "EXP-PARRILLO2023",
    devol_surrogate=SUR_EUCALYPTUS)
BLUPOLYMER_L = FeedstockComponent(
    "blupolymer_L", dict(C=83.7, H=12.4, N=0.2, S=0.0, O=1.4, Ash=2.4),
    0.01, 100 - 0.01 - 2.4 - 0.34, 0.34, 2.4, 40.5, "EXP-PARRILLO2023",
    devol_surrogate=SUR_BLU_L)
REACTOR_FABER = ReactorConfig(0.489, 5.73, 234.0, "olivine", 316e-6, 2900.0,
                              1 - 1600.0/2900.0, 50, 3600.0, "FABER", tar_catalyst=None)
_PLBM = [(1.0, 0.0), (.8, .2), (.7, .3), (.6, .4), (.6, .4), (.6, .4), (.6, .4), (.6, .4),
         (.3, .7), (.4, .6), (.4, .6), (.4, .6), (.4, .6), (0.0, 1.0)]
TABLE_FABER = pd.DataFrame({
    "PL": [p for p, b in _PLBM], "BM": [b for p, b in _PLBM],
    "feed_kg_h": [42, 60, 53, 72, 67, 63, 60, 51, 85, 89, 71, 71, 65, 80],
    "ER": [.25, .21, .25, .20, .21, .22, .25, .28, .24, .19, .24, .26, .27, .25],
    "AF": [3.5, 2.5, 2.8, 2.1, 2.2, 2.3, 2.5, 2.9, 1.7, 1.7, 2.1, 2.1, 2.3, 1.29],
    "T_bed_C": [890, 865, 868, 849, 864, 888, 872, 920, 823, 797, 861, 889, 908, 806],
    "N2":  [67, 61, 65.3, 60, 61, 61, 63, 66, 53.5, 54, 58, 59, 59, 49],
    "CO2": [11, 13, 12.4, 13, 13, 14, 14, 13, 15.3, 16, 15, 14, 15, 17],
    "CO":  [3.2, 4.1, 4.6, 5.0, 4.8, 5.9, 4.8, 3.7, 9.9, 7.8, 7.3, 8.6, 8.0, 15],
    "H2":  [6.5, 5.7, 5.4, 6.1, 5.9, 6.8, 5.9, 5.6, 10.8, 9.0, 8.0, 8.7, 7.8, 13.7],
    "CH4": [6.2, 7.5, 5.8, 7.3, 6.6, 6.3, 5.0, 5.9, 5.6, 5.8, 5.5, 5.0, 5.3, 4.2],
    "LHV": [7.5, 10.1, 7.8, 10.3, 9.6, 8.4, 8.2, 7.4, 8.0, 9.6, 8.1, 7.1, 7.2, 5.8]})   # MJ/Nm3, Table 3
TABLE_FABER["air_Nm3_h"] = TABLE_FABER.AF * TABLE_FABER.feed_kg_h / 1.293
# Measured freeboard-outlet temperature of each test: thesis of A. Sebastiani, Table 4.2 (tests 2, 13-24 and, for the
# eucalyptus-only test, test 3). Only the outlet value is reported: a linear profile from the bed surface is assumed.
TABLE_FABER["T_fb_out_C"] = [758, 745, 749, 749, 720, 750, 755, 737, 764, 737, 734, 732, 748, 725]
TABLE_FABER["test"] = [f"T{i+1:02d}_{p:.1f}PL_{b:.1f}BM"
                       for i, (p, b) in enumerate(zip(TABLE_FABER.PL, TABLE_FABER.BM))]
P_GAUGE_FABER = 8.2

def faber_feed(pl, bm):
    comps, fr = {}, {}
    if pl > 0: comps["blupolymer_L"]   = BLUPOLYMER_L;   fr["blupolymer_L"]   = pl
    if bm > 0: comps["eucalyptus_chips"] = EUCALYPTUS_FABER; fr["eucalyptus_chips"] = bm
    return FeedstockConfig(comps, fr, f"{pl}PL/{bm}BM")




# -----------------------------------------------------------------------------
# 6. THERMODYNAMICS (NASA-7, WGS equilibrium)
# -----------------------------------------------------------------------------
# NASA-7 coefficients (POLIMI_REDUCED_1410.CKT, CRECK).
# Format: (T_mid, high-T coefficients, low-T coefficients)
_NASA_WGS = {
    "H2":  (1000.0, [2.9914222, 0.00070006441, -5.633828e-08, -9.2315782e-12, 1.582752e-15, -835.033546, -1.35510641],
                    [3.298124, 0.00082494412, -8.1430147e-07, -9.4754343e-11, 4.134872e-13, -1012.521, -3.294094]),
    "H2O": (1000.0, [2.67214569, 0.0030562929, -8.7302607e-07, 1.200996e-10, -6.3916179e-15, -29899.2115, 6.86281125],
                    [3.386842, 0.003474982, -6.3546959e-06, 6.9685804e-09, -2.506588e-12, -30208.11, 2.590232]),
    "CO":  (1000.0, [3.02507617, 0.001442689, -5.6308272e-07, 1.018581e-10, -6.9109511e-15, -14268.3499, 6.10822521],
                    [3.262451, 0.001511941, -3.8817552e-06, 5.5819438e-09, -2.474951e-12, -14310.54, 4.848897]),
    "CO2": (1000.0, [4.45362582, 0.003140168, -1.278411e-06, 2.3939961e-10, -1.669033e-14, -48966.9524, -0.955420007],
                    [2.275724, 0.0099220723, -1.040911e-05, 6.8666859e-09, -2.1172801e-12, -48373.14, 10.18849]),
}

def _g_rt(sp, T):
    tmid, hi, lo = _NASA_WGS[sp]
    a = hi if T > tmid else lo
    h = a[0] + a[1] * T / 2 + a[2] * T ** 2 / 3 + a[3] * T ** 3 / 4 + a[4] * T ** 4 / 5 + a[5] / T
    s_ = a[0] * np.log(T) + a[1] * T + a[2] * T ** 2 / 2 + a[3] * T ** 3 / 3 + a[4] * T ** 4 / 4 + a[6]
    return h - s_

def wgs_Kp(T):
    """Equilibrium constant of CO + H2O <-> CO2 + H2 from NASA-7 thermodynamics (Kp = Kc)."""
    T = float(T)
    return float(np.exp(-(_g_rt("CO2", T) + _g_rt("H2", T) - _g_rt("CO", T) - _g_rt("H2O", T))))

def wgs_equilibrate(F, T):
    a, b, c, d = F[IDX["CO2"]], F[IDX["H2"]], F[IDX["CO"]], F[IDX["H2O"]]
    K = wgs_Kp(T)
    f = lambda x: (a + x) * (b + x) - K * (c - x) * (d - x)
    lo, hi = -min(a, b) + 1e-15, min(c, d) - 1e-15
    if hi <= lo:
        return F
    x = brentq(f, lo, hi) if f(lo) * f(hi) < 0 else 0.0
    G = F.copy()
    G[IDX["CO2"]] += x; G[IDX["H2"]] += x
    G[IDX["CO"]]  -= x; G[IDX["H2O"]] -= x
    return G


# -----------------------------------------------------------------------------
# 7. HYDRODYNAMICS
# -----------------------------------------------------------------------------
# Lumped bed hydrodynamics
def hydrodynamics(rc, T_K, P_abs, Q_gas_m3_s):
    A = np.pi / 4.0 * rc.D_m ** 2
    rho_g = P_abs * 0.0288 / (R_GAS * T_K)
    mu_g = 1.716e-5 * (T_K / 273.15) ** 0.7
    U0 = Q_gas_m3_s / A
    Ar = 9.81 * rc.dp_bed_m ** 3 * rho_g * (rc.rho_p_kg_m3 - rho_g) / mu_g ** 2
    Re_mf = np.sqrt(27.2 ** 2 + 0.0408 * Ar) - 27.2
    Umf = Re_mf * mu_g / (rc.dp_bed_m * rho_g)
    H_mf = rc.bed_mass_kg / (rc.rho_p_kg_m3 * (1.0 - rc.eps_mf) * A)
    if U0 > Umf:
        d_bm = 0.65 * (A * 1e4 * (U0 - Umf) * 100.0) ** 0.4 / 100.0
        d_b0 = 0.347 * (A * 1e4 * (U0 - Umf) * 100.0 / rc.n_orifices) ** 0.4 / 100.0
        z = H_mf / 2.0
        d_b = d_bm - (d_bm - d_b0) * np.exp(-0.3 * z * 100.0 / (rc.D_m * 100.0))
        u_b = U0 - Umf + 0.711 * np.sqrt(9.81 * d_b)
        delta = min((U0 - Umf) / u_b, 0.9)
    else:
        d_b, u_b, delta = 0.0, 0.0, 0.0
    H_f = H_mf / (1.0 - delta)
    V_bed_gas = A * H_f - rc.bed_mass_kg / rc.rho_p_kg_m3
    H_fb = max(rc.H_total_m - H_f, 0.0)
    return dict(A=A, U0=U0, Umf=Umf, U_Umf=U0 / Umf, H_mf=H_mf, H_f=H_f,
                V_bed_gas=V_bed_gas, H_fb=H_fb, V_fb=A * H_fb)

# --- Local two-phase hydrodynamics (used by the two-zone and the 1-D beds) ---------------------------------------
# Source: Kunii & Levenspiel, Fluidization Engineering, 2nd ed., Butterworth-Heinemann (1991), ch. 5-6, 10;
#         same set used by Sebastiani et al. (2025), SI Table S3 (Eqs. S.4-S.6, S.12-S.14).
# Equation: Re_mf = (27.2^2 + 0.0408 Ar)^0.5 - 27.2 ; u_br = 0.711 (g d_b)^0.5 ; u_b = U0 - Umf + u_br ;
#           delta = (U0 - Umf)/u_b ; bubble size: Mori & Wen form, evaluated at the local height z.
# Units: SI.  Validity: Geldart B/D, bubbling regime (no slugging: d_b < ~0.6 D is checked in diagnostics).
# Selected because: same correlations as the lumped bed; only the evaluation is local.
G_ACC = 9.81
def umf_of(rc, T_K, P_abs):
    rho_g = P_abs * 0.0288 / (R_GAS * T_K)
    mu_g = 1.716e-5 * (T_K / 273.15) ** 0.7
    Ar = G_ACC * rc.dp_bed_m ** 3 * rho_g * (rc.rho_p_kg_m3 - rho_g) / mu_g ** 2
    Re_mf = np.sqrt(27.2 ** 2 + 0.0408 * Ar) - 27.2
    return Re_mf * mu_g / (rc.dp_bed_m * rho_g), Ar

def hydro_local(rc, A, Umf, U0, z):
    """Local bubble properties at height z for superficial velocity U0 (two-phase theory)."""
    if U0 <= Umf * (1.0 + 1e-9):
        return 0.0, 0.0, 0.0, 0.0                       # delta, d_b, u_br, u_b : no bubbles
    d_bm = 0.65 * (A * 1e4 * (U0 - Umf) * 100.0) ** 0.4 / 100.0
    d_b0 = 0.347 * (A * 1e4 * (U0 - Umf) * 100.0 / rc.n_orifices) ** 0.4 / 100.0
    d_b = d_bm - (d_bm - d_b0) * np.exp(-0.3 * z / rc.D_m)
    u_br = 0.711 * np.sqrt(G_ACC * d_b)
    u_b = U0 - Umf + u_br
    return min((U0 - Umf) / u_b, 0.9), d_b, u_br, u_b

# Source: Fuller, Schettler & Giddings correlation with the original atomic diffusion volumes, as tabulated in
#         Poling, Prausnitz & O'Connell, The Properties of Gases and Liquids, 5th ed., McGraw-Hill (2001), ch. 11;
#         same equation as Sebastiani et al. (2025), SI Eq. S.15.
# Equation: D_iN2 [cm2/s] = 1e-3 T^1.75 (1/M_i + 1/M_N2)^0.5 / ( P[atm] (v_i^(1/3) + v_N2^(1/3))^2 )
# Units: T in K, P in atm, M in g/mol; converted to m2/s.  Validity: low-pressure binary gas diffusion.
# Selected because: needed for the bubble-emulsion interchange coefficient; N2 is the carrier gas (air gasification).
FULLER_V = {"H2": 7.07, "O2": 16.6, "N2": 17.9, "CO": 18.9, "CO2": 26.9, "H2O": 12.7, "CH4": 24.42, "C2H4": 40.92,
            "C3H6": 61.38, "C6H6": 90.68, "C7H8": 111.14, "C6H6O": 96.16, "C10H8": 140.44, "Tar": 140.44, "C_s": 17.9}
_FV = np.array([FULLER_V[s] for s in SPECIES])
def diffusivity_N2(T_K, P_abs):
    Mi = np.where(IS_SOLID, 28.014, MW)
    return 1e-7 * T_K ** 1.75 * np.sqrt(1.0 / Mi + 1.0 / 28.014) / ((P_abs / P_ATM) * (_FV ** (1 / 3) + 17.9 ** (1 / 3)) ** 2)

# Source: Kunii & Levenspiel (1991), Eqs. for K_bc, K_ce, K_be (Sebastiani 2025 SI Eqs. S.12-S.14).
# Equation: K_bc = 4.5 Umf/d_b + 5.85 D^0.5 g^0.25 / d_b^1.25 ; K_ce = 6.77 (D eps_mf u_br / d_b^3)^0.5 ;
#           1/K_be = 1/K_bc + 1/K_ce   [s^-1, per unit bubble volume]
# Units: SI.  Validity: bubbling beds of Geldart A/B solids (extrapolated to coarse B/D particles: flagged).
# Selected because: it is the interchange model of the reference architecture (Sebastiani).
def k_be(Umf, eps_mf, d_b, u_br, D):
    K_bc = 4.5 * Umf / d_b + 5.85 * np.sqrt(D) * G_ACC ** 0.25 / d_b ** 1.25
    K_ce = 6.77 * np.sqrt(D * eps_mf * u_br / d_b ** 3)
    return 1.0 / (1.0 / K_bc + 1.0 / K_ce)


# -----------------------------------------------------------------------------
# 8. DEVOLATILIZATION
# -----------------------------------------------------------------------------
def feed_elements(fs, feed_kg_h):
    out = {e: 0.0 for e in ["C", "H", "O", "N", "S"]}
    moist = ash = fc = hhv = dry_kg_s = 0.0
    vm_by, fc_by = {}, {}
    feed_kg_s = feed_kg_h / 3600.0
    for key, w in fs.mass_fractions.items():
        c = fs.components[key]
        tot = sum(c.ultimate_dry[k] for k in ["C", "H", "N", "S", "O", "Ash"])
        m_ar = feed_kg_s * w
        m_dry = m_ar * (1.0 - c.moisture_ar / 100.0)
        dry_kg_s += m_dry
        for e in out:
            out[e] += m_dry * (c.ultimate_dry[e] / tot) / AW[e] * 1000.0
        ash += m_dry * c.ultimate_dry["Ash"] / tot
        moist += m_ar * c.moisture_ar / 100.0
        fc_i = m_dry * (c.FC_ar / (1.0 - c.moisture_ar / 100.0)) / 100.0
        fc += fc_i; fc_by[key] = fc_i
        vm_by[key] = m_ar * c.VM_ar / 100.0
        hhv += w * c.HHV_MJ_kg
    return dict(el=out, moist_kg_s=moist, ash_kg_s=ash, fc_kg_s=fc, dry_kg_s=dry_kg_s,
                feed_kg_s=feed_kg_s, HHV_blend=hhv, vm_by=vm_by, fc_by=fc_by,
                H_pct=100*out["H"]*AW["H"]/1000/dry_kg_s,
                O_pct=100*out["O"]*AW["O"]/1000/dry_kg_s,
                M_pct=100*moist/feed_kg_s)

# Measured volatile distribution of a refuse-derived fuel instead of an assumed 35 % eucalyptus + 65 % polyolefin mix.
# Source: Tokmurzin et al. (2022), Energy Convers. Manag. 267, 115925, Table 7: pyrolysis of SRF in a bubbling fluidised bed
#         at 800 C, composition of the volatiles in wt%: H2 1.03, CO 17.26, CH4 11.15, CO2 16.81, C2H4 25.35, C2H2 4.69,
#         C2H6 1.86, C3H6 17.3, tar 4.55 (sum 100). C2H2 and C2H6 are lumped into C2H4 (no such species in the network).
# Units: wt% of the volatiles (used as a shape; C, H, O are closed afterwards as for every proxy).
# Validity: one temperature (800 C); plastic-rich SRF (C 62.5 %, H 8.1 % dry), richer in plastics than the RDF of Terrados
#         (C 35.7 %, O 35.3 %, ash 26.2 %): transfer between RDFs is an assumption, flagged.
# Reason: a measured RDF-class distribution replaces an assumed mixture; literature input, not a fit.
SUR_SRF_TOKMURZIN = {"H2": 1.03, "CO": 17.26, "CH4": 11.15, "CO2": 16.81, "C2H4": 25.35 + 4.69 + 1.86, "C3H6": 17.3, "Tar": 4.55}
SUR_MEASURED = {"RDF": SUR_SRF_TOKMURZIN, "RDF_pio": SUR_SRF_TOKMURZIN}

# Option devol_wood_measured (off): measured fast-pyrolysis yields of wood instead of the eucalyptus column of Sebastiani 2025 Table 2.
# Source: Heidari et al. (2014), J. Ind. Eng. Chem. 20, 2594-2602, DOI 10.1016/j.jiec.2013.10.046 (Eucalyptus grandis,
#         fluidised bed). At 600 C: gas 48.4 wt%, char 14.2 wt%, bio-oil 37.4 wt% (by difference) with 34 wt% water
#         -> water 12.7 wt%, organics 24.7 wt% of the wood. Gas composition at 600 C (Fig. 7, vector reading): CO 44.1,
#         CO2 22.9, CH4 11.3, H2 7.3, C2-C4 3.3.
# Units: the axis of Fig. 7 says "wt%", but as mass fractions the gas alone would carry 108 % of the hydrogen of the wood
#         (H2 = 3.5 wt% of the wood); read as vol% the balance closes and the composition is the usual one of wood
#         pyrolysis gas. They are therefore taken as vol% (normalised to 100) and converted to mass:
#         CO 23.6, CO2 19.3, CH4 3.46, C2H4 1.77, H2 0.28 wt% of the wood. FLAGGED interpretation.
#         The Sebastiani column (CO 10.7, CO2 18.5, phenol 34.5, toluene 6.5, ... sum 111 %) cannot be traced to this source.
# Hypotheses: the organics of the bio-oil are represented by phenol (tar model compound of Ji 2009 / Gerun 2008); C2-C4 as
#         C2H4; valid for woody biomass (eucalyptus, pine); 600 C is the highest temperature measured.
# Reason: measured primary source instead of an elementally inconsistent secondary table; literature input, not a fit.
# Not adopted: it increased the errors for the woody datasets (CH4 doubles: with the organics written as phenol, which is
#         poor in H, the element closure sends the missing hydrogen to CH4). LHV improves, CO does not.
SUR_WOOD_HEIDARI = {"H2": 0.28, "CO": 23.6, "CO2": 19.3, "CH4": 3.46, "C2H4": 1.77, "H2O": 12.7, "C6H6O": 24.7}
SUR_MEASURED_WOOD = {"pine_pellets": SUR_WOOD_HEIDARI, "pine_chips": SUR_WOOD_HEIDARI, "eucalyptus_chips": SUR_WOOD_HEIDARI}

# Option devol_olive_measured (off): measured fast-pyrolysis yields of olive husk for the olive pomace.
# Source: Christoforou et al. (2018), Waste Biomass Valor. 9, 301-313, Tables 6-7 (non-catalytic run on silica sand, 500 C,
#         mass balance closed to 100 %): wt% of dry biomass: CO2 12.80, CO 4.11, H2 0.03, CH4 0.78, C2 0.28, C2= 0.22,
#         C3 0.11, C3= 0.19, C4-C6 0.60, water 21.67, organics 30.99, solid 28.23.
# Hypotheses: C2 species lumped into C2H4 (0.50), C3-C6 into C3H6 (0.90); organics represented by phenol; 500 C is a
#         primary (low-temperature) distribution, secondary conversion is left to the reaction network.
# Reason: fuel-specific measured yields instead of a wood proxy; literature input, not a fit.
# Not adopted: it increased the errors for the olive pomace datasets (a 500 C primary distribution needs a
#         secondary-cracking scheme for the oxygenated organics that the network does not have).
SUR_OLIVE_CHRISTOFOROU = {"H2": 0.03, "CO": 4.11, "CO2": 12.80, "CH4": 0.78, "C2H4": 0.50, "C3H6": 0.90, "H2O": 21.67, "C6H6O": 30.99}
SUR_MEASURED_OLIVE = {"olive_pomace": SUR_OLIVE_CHRISTOFOROU, "olive_pomace_calado": SUR_OLIVE_CHRISTOFOROU}

def blend_proxy(fs, fe, override=None):
    tot = sum(fe["vm_by"].values())
    shape = {}
    for key, vm in fe["vm_by"].items():
        sur = (override or {}).get(fs.components[key].name, fs.components[key].devol_surrogate)
        if sur is None:
            sur = NET["devolatilization"]["final_distribution_wtpct"]
        norm = sum(v for k, v in sur.items() if k != "Char")
        for sp, v in sur.items():
            if sp != "Char":
                shape[sp] = shape.get(sp, 0.0) + vm / tot * v / norm * 100.0
    return shape

def char_steam_set(fs, fe):
    tot = sum(fe["fc_by"].values())
    lst, w_fb = [], 0.0
    for key, fcm in fe["fc_by"].items():
        cs = fs.components[key].char_steam
        w = fcm / tot if tot > 0 else 0.0
        if cs is None:
            w_fb += w
        else:
            lst.append((w, cs["A"], cs["n"], cs["Ea"], cs["P_ref_Pa"]))
    return lst, w_fb

DEVOL_LOG = {}
# Molecular H2 is not allowed as an element-closure species of the devolatilisation.
# Source: measured primary pyrolysis yields of H2 are negligible in the three corpus sources that report them:
#         olive husk 0.03 wt% (Christoforou et al. 2018, Waste Biomass Valor. 9, 301-313, Table 7), hardwood 0.16 wt%
#         (Ji et al. 2009, Table 3, after Nunn et al. 1985), plastic-rich SRF 1.03 wt% (Tokmurzin et al. 2022, Table 7).
#         With H2 allowed in the closure, the closure of H-rich, O-poor fuels fell on (H2, CO2): e.g. 50 g H2 per kg of
#         Calado olive pomace, i.e. more H2 than the gasifier produces in total.
# Equation: same linear closure (shape scale + two closure species closing C, H, O); the pairs containing H2 are skipped,
#         so the next pairs of the list are used: (CH4, CO2), then (H2O, CH4). No new pair is introduced.
# Units: mol/s.  Validity: any fuel.  Reason: removes an unsourced choice that contradicts measurements; not a fit.
# Hydrocarbon closure species of olive residues = CH4 + C2 in the measured molar ratio, instead of CH4 alone.
# Source: Uzun, Putun & Putun (2007), J. Anal. Appl. Pyrolysis 79, 147-153, DOI 10.1016/j.jaap.2006.12.005, Fig. 4:
#         pyrolysis gas of olive-oil residue at 700 C: CH4 ~ 8.0 vol% (trend line), C2H4 3.4 vol%, C2H6 2.6 vol%
#         -> (C2H4 + C2H6)/CH4 = 0.75 mol/mol (digitised, +/- 0.1). C2H6 is lumped into C2H4 (no C2H6 species in the network).
# Equation: closure "species" HC = 1 CH4 + 0.75 C2H4 (fixed ratio); used wherever the closure would use CH4.
# Units: mol/mol.  Validity: olive residues, 550-700 C pyrolysis (ratio 1.5 at 550 C, 0.75 at 700 C).
# Reason: olive pomace keeps residual oil; its excess hydrogen leaves as light hydrocarbons, not only as CH4. Literature
#         input for one fuel class, not a fit (independent of the gasification datasets).
HC_CLOSURE_LIT = {"olive_pomace": {"CH4": 1.0, "C2H4": 0.75}, "olive_pomace_calado": {"CH4": 1.0, "C2H4": 0.75}}
def hc_closure_for(fs, fe):
    tot = sum(fe["vm_by"].values()); out = {}
    for key, vm in fe["vm_by"].items():
        for sp, v in HC_CLOSURE_LIT.get(fs.components[key].name, {"CH4": 1.0}).items():
            out[sp] = out.get(sp, 0.0) + v * vm / tot
    return out

def devolatilization(fe, proxy_wt, allow_h2_closure=True, hc_closure=None):
    hc = hc_closure or {"CH4": 1.0}
    hc_vec = np.zeros(NSP)
    for sp_, v_ in hc.items():
        hc_vec[IDX[sp_]] = v_
    col_of = lambda name: (ATOM[:, :3].T @ hc_vec) if name == "CH4" else ATOM[IDX[name], :3]
    el = fe["el"]
    n_char = fe["fc_kg_s"] * 1000.0 / AW["C"]
    target = np.array([el["C"] - n_char, el["H"], el["O"]])
    shape = np.zeros(NSP)
    for s, wt in proxy_wt.items():
        if s in IDX and wt > 0:
            shape[IDX[s]] = wt / MW[IDX[s]]
    col_shape = ATOM[:, :3].T @ shape
    for a, b in [("H2O", "CO2"), ("H2O", "CO"), ("H2", "CO2"),
                 ("H2", "CO"), ("CH4", "CO2"), ("H2O", "CH4")]:
        if "H2" in (a, b) and not allow_h2_closure:
            continue
        M = np.column_stack([col_shape, col_of(a), col_of(b)])
        try:
            x = np.linalg.solve(M, target)
        except np.linalg.LinAlgError:
            continue
        if np.all(x >= -1e-12):
            F = shape * x[0]
            for name_, x_ in ((a, x[1]), (b, x[2])):       # "CH4" stands for the hydrocarbon closure lump (CH4 alone by default)
                if name_ == "CH4":
                    F += x_ * hc_vec
                else:
                    F[IDX[name_]] += x_
            F[IDX["N2"]] += el["N"] / 2.0
            F[IDX["H2O"]] += fe["moist_kg_s"] * 1000.0 / MW[IDX["H2O"]]
            DEVOL_LOG["pair"] = (a, b)
            return F, n_char
    DEVOL_LOG["pair"] = ("nnls",)
    cl = (["H2"] if allow_h2_closure else []) + ["H2O", "CO", "CO2", "CH4", "C2H4", "C6H6", "C10H8"]
    M = np.column_stack([col_shape] + [ATOM[IDX[c], :3] for c in cl])
    x, _ = nnls(M, target)
    F = shape * x[0]
    for c, v in zip(cl, x[1:]):
        F[IDX[c]] += v
    F[IDX["N2"]] += el["N"] / 2.0
    F[IDX["H2O"]] += fe["moist_kg_s"] * 1000.0 / MW[IDX["H2O"]]
    return F, n_char

# --- 1-D three-phase bed (not used in the adopted model): axial release pattern of volatiles in the bed ------------
# Source: Sebastiani et al. (2025), Chem. Eng. J. 506, 159763, Fig. 1 (conversion profiles derived from X-ray
#         segregation experiments of Iannello et al.); thesis of A. Sebastiani, Fig. 3.5.  Digitised (+/- 0.05).
# Equation: X(xi) = cumulative fraction of volatiles released below xi = z/H_bed (piecewise linear).
# Units: dimensionless.  Validity: on-bed feeding, biomass (hardwood) and polyolefin particles, FI ~ 2.
# Selected because: over-bed/near-surface release of volatiles is the mechanism that separates O2 from volatiles;
#         a perfectly mixed bed premixes all volatiles with all the air.
RELEASE_CURVE = {
    "biomass": (np.array([0.0, 0.85, 0.90, 1.00]), np.array([0.0, 0.0, 0.07, 1.0])),
    "plastic": (np.array([0.0, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
                np.array([0.0, 0.0, 0.006, 0.058, 0.100, 0.134, 0.185, 0.270, 0.430, 0.740, 1.0])),
}
# component -> mixture of curves; RDF follows the weighting already used for its devolatilisation surrogate
# (35 % cellulosic / 65 % polyolefin), i.e. Sebastiani's rule for mixed waste (thesis, section 3.2.1).
RELEASE_OF = {"RDF": {"biomass": 0.35, "plastic": 0.65}, "RDF_pio": {"biomass": 0.35, "plastic": 0.65},
              "blupolymer_L": {"plastic": 1.0}}
def release_pattern(fs, fe):
    """Blend pattern: list of (weight, xi_pts, X_pts); weights = volatile-matter mass share of each curve."""
    tot = sum(fe["vm_by"].values()); w = {}
    for key, vm in fe["vm_by"].items():
        for curve, f in RELEASE_OF.get(fs.components[key].name, {"biomass": 1.0}).items():
            w[curve] = w.get(curve, 0.0) + f * vm / tot
    return [(wt, *RELEASE_CURVE[c]) for c, wt in w.items()]
def release_cum(pat, xi):
    return sum(w * np.interp(xi, xp, Xp) for w, xp, Xp in pat)
def release_density(pat, xi):
    """dX/dxi at xi (piecewise constant)."""
    out = 0.0
    for w, xp, Xp in pat:
        j = min(max(np.searchsorted(xp, xi, side="right") - 1, 0), len(xp) - 2)
        out += w * (Xp[j + 1] - Xp[j]) / (xp[j + 1] - xp[j])
    return out


# -----------------------------------------------------------------------------
# 9. KINETICS
# -----------------------------------------------------------------------------
def _order_table(key):
    rows, cols, ords = [], [], []
    for j, rid in enumerate(RXN_IDS):
        for s_, o in LAW[rid].get(key, {}).items():
            rows.append(j); cols.append(IDX[s_]); ords.append(float(o))
    return np.array(rows, int), np.array(cols, int), np.array(ords)

_FWD = _order_table("fwd")
_REV = _order_table("rev")
_KEQ = [(j, LAW[r]["keq"]) for j, r in enumerate(RXN_IDS) if "keq" in LAW[r]]
_IS_CHAR = np.array([LAW[r].get("basis", "").startswith("char") for r in RXN_IDS])
_IS_SURF = np.array([LAW[r].get("basis") == "char_surface" for r in RXN_IDS])
_IS_DOL  = np.array([LAW[r].get("basis") == "dolomite" for r in RXN_IDS])
_INHIB = [(j, K0, EK, IDX[s_]) for j, r in enumerate(RXN_IDS)
          for K0, EK, s_ in LAW[r].get("inhib", [])]

C_EPS = 1e-6

def _log_power(C, tab):
    rows, cols, ords = tab
    c = C[cols]
    with np.errstate(divide="ignore"):
        lg = np.where(c >= C_EPS, ords * np.log(np.maximum(c, 1e-300)),
                      ords * np.log(C_EPS) + np.log(np.maximum(c, 0.0) / C_EPS))
    out = np.zeros(len(RXN_IDS))
    np.add.at(out, rows, lg)
    return out

P_EPS_BAR = 1.0e-5
def _pow_reg(p_bar, n):
    """p^n with a linear continuation to zero below P_EPS_BAR (same regularisation idea as C_EPS)."""
    return np.maximum(p_bar, P_EPS_BAR) ** n * np.minimum(np.maximum(p_bar, 0.0) / P_EPS_BAR, 1.0)

def rates(T_K, C, kps):
    Cs = np.maximum(C, 0.0)
    k = kps.multiplier * kps.A * T_K ** kps.T_exp * np.exp(-kps.Ea / (R_GAS * T_K))
    f = np.exp(_log_power(Cs, _FWD if kps.fwd_tab is None else kps.fwd_tab))
    if kps.reversible_mode == "source_Keq":
        rev = np.exp(_log_power(Cs, _REV))
        for j, (K0, EK) in _KEQ:
            f[j] -= rev[j] / (K0 * np.exp(EK / (R_GAS * T_K)))
    den = np.ones(len(RXN_IDS))
    for j, K0, EK, i in _INHIB:
        den[j] += K0 * np.exp(-EK / (R_GAS * T_K)) * Cs[i]
    val = k * f / den
    surf = (6.0 / CHAR_PROPS["dp_char_m"]) * (AW["C"] * 1e-3 / CHAR_PROPS["rho_char_kg_m3"])
    val = np.where(_IS_SURF, val * surf, val)
    if kps.char_steam:
        pH2O = Cs[IDX["H2O"]] * R_GAS * T_K
        r_cs = sum(w * A * (pH2O / Pref) ** n * np.exp(-Ea / (R_GAS * T_K))
                   for w, A, n, Ea, Pref in kps.char_steam)
        val[_I_XR02] = kps.char_steam_fallback_w * val[_I_XR02] + r_cs
    if kps.char_lit:                                       # R_C [1/s] = k0 exp(-E/RT) (p/1 bar)^n, per mol of char C
        w_fb = kps.char_lit_fallback_w
        for gas, j_, per_c in (("CO2", _I_XR01, 1.0), ("O2", _I_XR03, 1.0 / char_oxidation_alpha(T_K))):
            p_bar = Cs[IDX[gas]] * R_GAS * T_K / 1.0e5
            r_ = sum(w * A_ * _pow_reg(p_bar, n) * np.exp(-Ea / (R_GAS * T_K)) for w, A_, n, Ea in kps.char_lit.get(gas, []))
            val[j_] = w_fb * val[j_] + r_ * per_c         # X-R03 is written per mol O2: divide the carbon rate by alpha(T)
    val = np.where(_IS_DOL, 0.0, val)
    return np.where(_IS_CHAR, 0.0, val), np.where(_IS_CHAR, val, 0.0)

K_H2O_GUARD = 1e-2
def dolomite_rates(T_K, C, kps):
    Cs = np.maximum(C, 0.0)
    k = kps.A * np.exp(-kps.Ea / (R_GAS * T_K))
    f = np.exp(_log_power(Cs, _FWD if kps.fwd_tab is None else kps.fwd_tab))
    g = Cs[IDX["H2O"]] / (Cs[IDX["H2O"]] + K_H2O_GUARD)
    return np.where(_IS_DOL, k * f * g, 0.0)


# --- Fuel-specific char kinetics from the literature (only where a source for THAT fuel exists) -------------------------
# Source: Senneca (2007), Fuel Process. Technol. 88, 87-97, Table 5 (olive husk char, prepared in a fluidised bed at
#         850 C, < 300 um, thermogravimetric, kinetic regime).
# Equation: R_0.5 = k0 exp(-E/RT) p_g^n  [1/s]  (carbon conversion rate averaged over the first 50 % conversion, Eq. 4)
#         char combustion    : k0 = 1.3e9 1/s, n = 0.9,  E = 35 kcal/mol   (measured 400-480 C, p_O2 0.05-0.21 bar)
#         CO2 gasification   : k0 = 8.0e6 1/s, n = 0.50, E = 48 kcal/mol   (measured 750-910 C, p_CO2 0.05-1 bar)
#         H2O gasification   : measured at a single condition (830 C, 30 % H2O): NO Arrhenius parameters -> not used;
#                              the generic steam kinetics of Table S6 is kept for X-R02.
# Units: p in bar, E converted to J/mol (x 4184), rate per mol of char carbon.
# Validity: combustion is extrapolated far above its measured range and carries no external mass-transfer limit
#         (flagged: at bed temperature the true rate is diffusion controlled). CO2 gasification is inside its range
#         up to 910 C.
# Reason for selection: olive husk = olive pomace of Terrados/Calado; its char is far more reactive (K-rich ash) than
#         the generic char of Table S6 (sewage-sludge kinetics of Petersen & Werther). Literature input, not a fit.
KCAL = 4184.0
CHAR_LIT = {"olive_pomace": {"CO2": (8.0e6, 0.50, 48 * KCAL), "O2": (1.3e9, 0.9, 35 * KCAL)},
            "olive_pomace_calado": {"CO2": (8.0e6, 0.50, 48 * KCAL), "O2": (1.3e9, 0.9, 35 * KCAL)}}
# Char kinetics of wood chips for the chars of woody fuels (pine, eucalyptus), same source and same form as the
# olive husk kinetics above.  Senneca (2007), Table 5, wood chips: combustion k0 = 1.4e5 1/s, n = 0.9, E = 24 kcal/mol
# (measured 400-600 C); CO2 gasification k0 = 1.1e9 1/s, n = 0.64, E = 62 kcal/mol (measured 750-910 C).
# Validity/flags as for the olive husk kinetics (combustion extrapolated, no external mass-transfer limit); the wood of
# Senneca is not the pine or eucalyptus of the validation sets: transfer within the fuel class is an assumption.
CHAR_LIT_WOOD = {k: {"CO2": (1.1e9, 0.64, 62 * KCAL), "O2": (1.4e5, 0.9, 24 * KCAL)} for k in ("pine_pellets", "pine_chips", "eucalyptus_chips")}
def char_lit_set(fs, fe, wood=False):
    """Weights the literature kinetics by the fixed-carbon share of each fuel; the rest keeps the generic kinetics."""
    tot = sum(fe["fc_by"].values()); out, w_fb = {"CO2": [], "O2": []}, 0.0
    for key, fcm in fe["fc_by"].items():
        w = fcm / tot if tot > 0 else 0.0
        lit_ = CHAR_LIT.get(fs.components[key].name) or (CHAR_LIT_WOOD.get(fs.components[key].name) if wood else None)
        if lit_ is None:
            w_fb += w
        else:
            for gas, (A_, n, Ea) in lit_.items():
                out[gas].append((w, A_, n, Ea))
    return (out if w_fb < 1.0 else None), w_fb

# --- Fitting factors of the source kinetic set (Petersen & Werther 2005) ---------------------------------------------
# Source: Petersen & Werther (2005), Chem. Eng. Process. 44, 717-736, DOI 10.1016/j.cep.2004.09.001, Table 5
#         ("Kinetic rate constants valid for sewage sludge gasification", r in mol/m3/s, c in mol/m3). Every law there is
#         r = k(T) * g(c) * f, with f fitted by the authors to axial concentration profiles of a 15 m fluidised-bed riser:
#           (1') char + O2        f = 0.05      (X-R03)      (5') H2 + 1/2 O2   f = 0.001   (H-R03)
#           (2') char + H2O       f = 2, (1 - X) with X = 0.5 -> 1.0 (X-R02)   (6') CO + 1/2 O2   f = 1       (H-R04)
#           (3') char + CO2       f = 1, (1 - X) with X = 0.35 -> 0.65 (X-R01) (7') WGS           f = 0.1     (H-R01)
#           (4') CH4 + 1/2 O2     f = 100       (H-R05)      (8') CH4 + H2O     f = 0.1     (H-R02)
#         Sebastiani 2025 SI Tables S5-S6 list the same k(T) (r1-r5, r21-r23) WITHOUT the factors f and (1 - X).
# Equation: A_used = A_table * f * (1 - X).   Units: unchanged (SI).
# Validity: 1023-1123 K, air/CO2 gasification of sewage sludge in a circulating fluidised bed; applied here outside the
#         fuel it was fitted for, exactly as the unfactored constants were.
# Reason for selection: the factors are part of the published kinetic set; without them H2 oxidation is 1000 times
#         faster and CH4 partial oxidation 100 times slower than in the source. Restores the source; nothing is fitted here.
# Not adopted from the same table (flagged): beta = 1.2 in C + beta H2O (the network keeps C + H2O -> CO + H2); the char
#         term c_S/(M_C rho_char d_p) of the methane-steam law (8'), negligible at the residence times of the bed.
# Check of the unit correction above: P&W use for C2H4 + O2 the SI law 3.71e12 exp(-209205/RT) c^0.9 c_O2^1.18
#         (their reaction 10'), which at 800 C gives a rate of the same order as the converted Westbrook-Dryer law.
PW_FACTORS = {"X-R03": 0.05, "X-R02": 2.0 * (1.0 - 0.5), "X-R01": 1.0 * (1.0 - 0.35), "H-R05": 100.0, "H-R03": 0.001,
              "H-R04": 1.0, "H-R01": 0.1, "H-R02": 0.1}

# --- Reaction order of H-R20 (phenol + 3 H2O -> 4 CO + 2 CH4 + 2 H2) ----------------------------------------------
# Source: Ji, Feng & Chen (2009), Chem. Eng. Sci. 64, 582-592, DOI 10.1016/j.ces.2008.10.015, Table 4:
#         r = 1e7 exp(-1e5/RT) C_C6H6O (mol/m3/s): first order in phenol, NO steam term (attributed there to Morf 2002;
#         phenol as tar model compound after Gerun et al. 2008). Tokmurzin 2022 (Table 5, R29) cites Ji/Morf but prints
#         [C6H6O][H2O]^3, and Sebastiani 2025 SI (r20) copies c_H2O^3.
# Equation: r20 = k20 c_C6H6O
# Units: mol/m3/s with k in 1/s (with c_H2O^3 the pre-exponential 1e7 would not have units of 1/s).
# Validity: air-blown BFB biomass gasification, bed 973-1173 K (range validated by Ji et al.).
# Reason for selection: correction of a transcription error along the source chain, NOT a fit.
LAW_R20_FIX = {"C6H6O": 1}
def _order_table_r20_fixed():
    rows, cols, ords = [], [], []
    for j, rid in enumerate(RXN_IDS):
        for s_, o in (LAW_R20_FIX if rid == "H-R20" else LAW[rid].get("fwd", {})).items():
            rows.append(j); cols.append(IDX[s_]); ords.append(float(o))
    return np.array(rows, int), np.array(cols, int), np.array(ords)
KPS_SI_R20 = replace(KPS_SI, fwd_tab=_order_table_r20_fixed())
_mult_pw = KPS_SI_R20.multiplier.copy()
for _r, _f in PW_FACTORS.items():
    _mult_pw[RXN_IDS.index(_r)] *= _f
KPS_PW = replace(KPS_SI_R20, multiplier=_mult_pw)            # source kinetic set with its fitting factors
def kps_with_pw(*ids, off=()):
    """Kinetic set with the corrected reaction orders and the Petersen & Werther factor applied only to the listed reactions (one-at-a-time ablation);
    reactions in 'off' are switched off."""
    m_ = KPS_SI_R20.multiplier.copy()
    for r_ in ids:
        m_[RXN_IDS.index(r_)] *= PW_FACTORS[r_]
    for r_ in off:
        m_[RXN_IDS.index(r_)] = 0.0
    return replace(KPS_SI_R20, multiplier=m_)

# --- H-R20 (phenol + 3 H2O -> 4 CO + 2 CH4 + 2 H2) switched off -------------------------------------------------------
# Source: Morf, Hasler & Nussbaumer (2002), Fuel 81, 843-853, DOI 10.1016/S0016-2361(01)00216-2.
#   (a) Table 2: the constants A = 1.0e7 1/s, Ea = 100 kJ/mol that Ji 2009 / Tokmurzin 2022 / Sebastiani 2025 attach to BOTH
#       phenol reactions are Morf's k_N2 of the step "intermediate (phenols) -> naphthalene" of the series
#       gravimetric tar -> intermediate -> naphthalene -> soot. Morf gives no steam reaction and no stoichiometry for r20.
#   (b) Fig. 3-4, 8 (homogeneous tar conversion, 500-990 C, digitised): for ~28.5 wt%-of-wood of tar converted the gases grow
#       by CO +6.6, CH4 +1.1, H2 +0.36, CO2 ~+0.5 wt% of wood, the rest being soot/PAH (~20 wt%). Per mole of phenol-equivalent
#       (94 g): CO 0.78, CH4 0.23, H2 0.59 mol.  r19 gives CO 1.0, CH4 0.1, H2 0.75 mol + aromatics (same pattern);
#       r20 gives CO 4, CH4 2, H2 2 mol and no aromatics: ~9 times the measured methane.
# Equation: phenol converts through r19 only (unchanged law r19 = 1e7 exp(-1e5/RT) c_phenol).
# Units: SI.  Validity: wood pyrolysis tar, 500-990 C, < 0.2 s, no external oxidant.
# Reason: the product slate of r20 has no experimental support in its cited source; literature evidence, not a fit.
KPS_R20_OFF = kps_with_pw("H-R01", off=("H-R20",))            # kinetic WGS with f = 0.1, H-R20 off

# -----------------------------------------------------------------------------
# 10. BED
# -----------------------------------------------------------------------------
# Single isothermal CSTR (perfectly mixed bed, formulation A)
def bed_cstr(F_in_gas, n_char_in, T_K, P_abs, V_gas, tau_char, kps, opt, W_dol=0.0):
    n_tot = P_abs * V_gas / (R_GAS * T_K)
    F_tot_in = F_in_gas.sum()
    y0 = F_in_gas / F_tot_in
    x0 = np.concatenate([y0 * n_tot, [n_char_in * tau_char]])
    nu_gas = nu_matrix(T_K)
    def rhs(t, x):
        n = np.maximum(x[:NSP], 0.0); n[I_CS] = 0.0
        N_char = max(x[NSP], 0.0)
        C = n / V_gas
        r_v, r_c = rates(T_K, C, kps)
        gen = nu_gas.T @ (r_v * V_gas + r_c * N_char + dolomite_rates(T_K, C, kps) * W_dol)
        gen_gas = gen.copy(); gen_gas[I_CS] = 0.0
        F_tot_out = F_tot_in + gen_gas.sum()
        F_out = n / max(n.sum(), 1e-30) * F_tot_out
        return np.concatenate([F_in_gas - F_out + gen_gas,
                               [n_char_in - N_char / tau_char + gen[I_CS]]])
    atol = np.concatenate([np.full(NSP, 1e-12 * n_tot), [1e-9 * max(x0[-1], 1.0)]])
    sol = solve_ivp(rhs, [0.0, opt.t_end_factor * tau_char], x0, method="BDF",
                    rtol=opt.rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"bed_cstr: {sol.message}")
    x = sol.y[:, -1]
    resid = np.abs(rhs(0, x))
    n = np.maximum(x[:NSP], 0.0); n[I_CS] = 0.0
    N_char = max(x[NSP], 0.0)
    r_v, r_c = rates(T_K, n / V_gas, kps)
    gen = nu_gas.T @ (r_v * V_gas + r_c * N_char + dolomite_rates(T_K, n / V_gas, kps) * W_dol)
    gen_gas = gen.copy(); gen_gas[I_CS] = 0.0
    F_out = n / n.sum() * (F_tot_in + gen_gas.sum())
    return dict(F_out=F_out, N_char=N_char, F_char_out=N_char / tau_char,
                steady_resid=float(resid.max()), tau_gas=n_tot / F_tot_in)


# -----------------------------------------------------------------------------
# Bed model "two_zone": bubble phase + emulsion phase, both perfectly mixed (formulation B, adopted)
# -----------------------------------------------------------------------------
# Purpose: physical description of how the oxygen is shared in the bed, without any fitted parameter: homogeneous
#   oxidation is confined to the bubble phase, as observed by Hayhurst and co-workers.
# Sources: two-phase theory and interchange coefficient: Kunii & Levenspiel, Fluidization Engineering, 2nd ed. (1991)
#   (functions hydro_local and k_be above). Inhibition of gas-phase oxidation in the particulate phase:
#   Hayhurst & Tucker (1990), Combust. Flame 79, 175-189; Hayhurst (1991), Combust. Flame 85, 155-168.
# Equations (steady state, isothermal; b = bubble phase, e = emulsion phase; c = n/V):
#   V_b = delta A H_f ; V_e = V_gas - V_b
#   air: fraction Umf/U0 to the emulsion, the rest to the bubbles (two-phase theory)
#   X_i = K_be,i V_b (c_b,i - c_e,i)                                  interchange, bubble -> emulsion [mol/s]
#   B   = F_e,in + G_e + sum(X) - c_tot Umf A                         gas in excess of the minimum-fluidisation flow
#                                                                     leaves the emulsion as bubbles (B > 0), or is
#                                                                     drawn from the bubbles (B < 0)
#   dn_e,i/dt = F_e,in,i + G_e,i + X_i - y_e,i F_e,out - B y_src,i
#   dn_b,i/dt = F_b,in,i + G_b,i - X_i - y_b,i F_b,out + B y_src,i
#   G_e = V_e R_hom*(c_e) + N_char R_char(c_e) + W_dol R_dol(c_e) ;  G_b = V_b R_hom(c_b)
#   R_hom* = homogeneous reactions WITHOUT those that consume O2 (inhibited in the particulate phase)
#   char inventory as in bed_cstr; soot formed in either phase joins it.
# Units: SI.  Validity: bubbling regime; Hayhurst's experiments cover sand beds up to about 1000 C (applied here at all T).
# Hypotheses that are NOT fixed by any source: where the volatiles are released. Two limiting cases are available,
#   opt.tz_volatiles_to = "emulsion" (adopted) or "bubble"; both are reported (Supporting Information, Table S10).
# No parameter is fitted.
def bed_two_zone(F_air, F_vol, n_char_in, T_K, P_abs, rc, kps, opt, W_dol, hyd):
    A, Umf, U0, V_gas, tau_char = hyd["A"], hyd["Umf"], hyd["U0"], hyd["V_bed_gas"], rc.tau_char_s
    delta, d_b, u_br, _ = hydro_local(rc, A, Umf, U0, hyd["H_mf"] / 2.0)
    V_b = max(delta * A * hyd["H_f"], 1e-6 * V_gas); V_e = V_gas - V_b
    c_tot = P_abs / (R_GAS * T_K)
    phi_e = min(Umf / U0, 1.0)
    w_e = opt.tz_vol_emulsion_frac if opt.tz_vol_emulsion_frac >= 0.0 else (1.0 if opt.tz_volatiles_to == "emulsion" else 0.0)
    F_e_in = phi_e * F_air + w_e * F_vol
    F_b_in = (1.0 - phi_e) * F_air + (1.0 - w_e) * F_vol
    Fe_in, Fb_in = float(np.sum(F_e_in)), float(np.sum(F_b_in))
    F_mf = c_tot * Umf * A
    Kbe = k_be(Umf, rc.eps_mf, d_b, u_br, diffusivity_N2(T_K, P_abs)) if delta > 0.0 else np.zeros(NSP)
    Kbe = np.where(IS_SOLID, 0.0, Kbe) * opt.tz_kbe_factor
    nu_gas = nu_matrix(T_K)
    ox = nu_gas[:, IDX["O2"]] < 0.0                                 # reactions that consume O2
    y0 = (F_air + F_vol) / (F_air + F_vol).sum()
    x0 = np.concatenate([y0 * c_tot * V_e, y0 * c_tot * V_b, [n_char_in * tau_char]])
    fast_rates = make_rates(T_K, kps)
    def parts(x):
        n_e = np.maximum(x[:NSP], 0.0); n_e[I_CS] = 0.0
        n_b = np.maximum(x[NSP:2 * NSP], 0.0); n_b[I_CS] = 0.0
        N_char = max(x[2 * NSP], 0.0)
        C_e, C_b = n_e / V_e, n_b / V_b
        rv_, rc_, rd_ = fast_rates(np.vstack([C_e, C_b]))            # same rate laws as rates()/dolomite_rates(), vectorised
        rv_e = np.where(ox, 0.0, rv_[0])
        gen_e = nu_gas.T @ (rv_e * V_e + rc_[0] * N_char + rd_[0] * W_dol)
        gen_b = nu_gas.T @ (rv_[1] * V_b)
        soot = gen_e[I_CS] + gen_b[I_CS]
        g_e, g_b = gen_e.copy(), gen_b.copy(); g_e[I_CS] = 0.0; g_b[I_CS] = 0.0
        X = Kbe * V_b * (C_b - C_e)
        y_e, y_b = n_e / max(n_e.sum(), 1e-30), n_b / max(n_b.sum(), 1e-30)
        B = Fe_in + g_e.sum() + X.sum() - F_mf
        if B >= 0.0:
            Tr = B * y_e
        else:
            B = -min(-B, max(Fb_in + g_b.sum() - X.sum(), 0.0)); Tr = B * y_b
        Fe_out = Fe_in + g_e.sum() + X.sum() - B
        Fb_out = Fb_in + g_b.sum() - X.sum() + B
        return n_e, n_b, N_char, g_e, g_b, soot, X, Tr, y_e, y_b, Fe_out, Fb_out
    def rhs(t, x):
        n_e, n_b, N_char, g_e, g_b, soot, X, Tr, y_e, y_b, Fe_out, Fb_out = parts(x)
        return np.concatenate([F_e_in + g_e + X - y_e * Fe_out - Tr, F_b_in + g_b - X - y_b * Fb_out + Tr,
                               [n_char_in - N_char / tau_char + soot]])
    atol = np.concatenate([np.full(NSP, 1e-12 * c_tot * V_e), np.full(NSP, 1e-12 * c_tot * V_b), [1e-9 * max(x0[-1], 1.0)]])
    sol = solve_ivp(rhs, [0.0, opt.t_end_factor * tau_char], x0, method="BDF", rtol=min(opt.rtol, 1e-7), atol=atol)
    if not sol.success:
        raise RuntimeError(f"bed_two_zone: {sol.message}")
    x = sol.y[:, -1]
    n_e, n_b, N_char, g_e, g_b, soot, X, Tr, y_e, y_b, Fe_out, Fb_out = parts(x)
    F_out = y_e * Fe_out + y_b * Fb_out
    return dict(F_out=F_out, N_char=N_char, F_char_out=N_char / tau_char, steady_resid=float(np.abs(rhs(0, x)).max()),
                tau_gas=c_tot * V_gas / (Fe_in + Fb_in), delta=delta, d_b=d_b, phi_e=phi_e, K_be_mean=float(np.mean(Kbe[~IS_SOLID])),
                F_e_out=y_e * Fe_out, F_b_out=y_b * Fb_out)

# --- 1-D three-phase bubbling bed (emulsion + exogenous bubbles + endogenous bubbles); not used in the adopted model --
# Source: architecture of Sebastiani et al. (2025), Table 1 / thesis Tables 3.5 and 3.7 (modified two-phase theory);
#         two-phase theory, interchange and net-flow closure: Kunii & Levenspiel, Fluidization Engineering, 2nd ed. (1991).
# Equations (steady state, isothermal, molar-flow form, per unit bed height z; V'_p = bed volume of phase p per metre):
#   dF_b,i /dz = -K_be,i V'_b  (c_b,i  - c_e,i) + R_hom,i(c_b ) V'_b                    + net flow
#   dF_eb,i/dz = -K_be,i V'_eb (c_eb,i - c_e,i) + R_hom,i(c_eb) V'_eb + F_vol,i dX/dz   - net flow
#   dF_e,i /dz = +interchange + R_hom,i(c_e) A (1-delta) eps_mf + R_char,i N_char/H_f + R_dol,i W_dol/H_f - net flow
#   V'_b + V'_eb = A delta (shared in proportion to the molar flows);  c_p,i = y_p,i P/(R T)
#   net flow: gas in excess of the minimum-fluidisation flow (Umf A c_tot) leaves the emulsion as bubbles; a deficit is
#             taken from the bubbles (two-phase theory), written as a fast relaxation (numerical constant lam)
#   char: perfectly mixed in the emulsion; inventory N_char from  n_in - N_char/tau_char + int(R_Cs dz) = 0
# Units: SI (mol/s, mol/m3, m).
# Hypotheses: isothermal bed at the imposed T; air enters at z = 0 and splits emulsion/bubble by two-phase theory;
#   volatiles + fuel moisture enter the endogenous-bubble phase following the release pattern (section 8); char and
#   dolomite only in the emulsion; homogeneous reactions in every gas phase (opt.bubble_homogeneous). Sebastiani
#   restricts reactions to the emulsion; that restriction is NOT kept (gas-phase reactions do not need solids).
# Selected because: avoids premixing O2 with the volatiles and lets the air meet the char before the volatiles, as
#   imposed by bottom air + on-bed feeding. Not used in the adopted model.
# Fast isothermal evaluator of the SAME rate laws as rates()/dolomite_rates() (section 9); used only by the 1-D bed
# because the ODE system calls it ~1e5 times. Section 19 checks that both evaluators agree.
def make_rates(T_K, kps):
    nR = len(RXN_IDS)
    k = kps.multiplier * kps.A * T_K ** kps.T_exp * np.exp(-kps.Ea / (R_GAS * T_K))
    k_dol = kps.A * np.exp(-kps.Ea / (R_GAS * T_K))
    def dense(tab):
        M = np.zeros((nR, NSP)); M[tab[0], tab[1]] = tab[2]; return M
    OF, OR = dense(_FWD if kps.fwd_tab is None else kps.fwd_tab), dense(_REV)
    MF, MR = (OF != 0).astype(float), (OR != 0).astype(float)
    keq = [(j, K0 * np.exp(EK / (R_GAS * T_K))) for j, (K0, EK) in _KEQ] if kps.reversible_mode == "source_Keq" else []
    INH = np.zeros((nR, NSP))
    for j, K0, EK, i in _INHIB:
        INH[j, i] += K0 * np.exp(-EK / (R_GAS * T_K))
    surf = (6.0 / CHAR_PROPS["dp_char_m"]) * (AW["C"] * 1e-3 / CHAR_PROPS["rho_char_kg_m3"])
    fac = np.where(_IS_SURF, surf, 1.0)
    i_w = IDX["H2O"]; logeps = np.log(C_EPS_BED)
    kq_j = np.array([j for j, _ in keq], int); kq_K = np.array([K for _, K in keq])
    def f(C):
        """C: (n_phases, NSP) concentrations in mol/m3 -> r_v, r_c, r_dol, each (n_phases, n_reactions)."""
        Cs = np.maximum(C, 0.0)
        lc = np.log(np.maximum(Cs, C_EPS_BED))
        corr = np.where(Cs < C_EPS_BED, np.log(np.maximum(Cs, 1e-300)) - logeps, 0.0)
        fw = np.exp(lc @ OF.T + corr @ MF.T)
        fnet = fw.copy()
        if len(kq_j):
            fnet[:, kq_j] -= np.exp(lc @ OR[kq_j].T + corr @ MR[kq_j].T) / kq_K
        val = k * fnet / (1.0 + Cs @ INH.T) * fac
        if kps.char_steam:
            pH2O = Cs[:, i_w] * R_GAS * T_K
            val[:, _I_XR02] = kps.char_steam_fallback_w * val[:, _I_XR02] + sum(
                w * A_ * (pH2O / Pref) ** n * np.exp(-Ea / (R_GAS * T_K)) for w, A_, n, Ea, Pref in kps.char_steam)
        if kps.char_lit:
            for gas, j_, per_c in (("CO2", _I_XR01, 1.0), ("O2", _I_XR03, 1.0 / char_oxidation_alpha(T_K))):
                p_bar = Cs[:, IDX[gas]] * R_GAS * T_K / 1.0e5
                val[:, j_] = kps.char_lit_fallback_w * val[:, j_] + per_c * sum(
                    w * A_ * _pow_reg(p_bar, n) * np.exp(-Ea / (R_GAS * T_K)) for w, A_, n, Ea in kps.char_lit.get(gas, []))
        r_dol = np.where(_IS_DOL, k_dol * fw * (Cs[:, i_w] / (Cs[:, i_w] + K_H2O_GUARD))[:, None], 0.0)
        return np.where(_IS_CHAR | _IS_DOL, 0.0, val), np.where(_IS_CHAR, val, 0.0), r_dol
    return f

C_EPS_BED = 1e-6      # mol/m3: same linear continuation of the power laws as C_EPS in rates()
BED_RTOL = 1e-7       # the near-zero-order oxidation laws need a tight tolerance (looser values break the element balance)
def bed_two_phase(F_air, F_vol, n_char_in, T_K, P_abs, rc, kps, opt, W_dol, pat, profile=False):
    A = np.pi / 4.0 * rc.D_m ** 2
    ctot = P_abs / (R_GAS * T_K)
    Umf, Ar = umf_of(rc, T_K, P_abs)
    Fe_max = Umf * A * ctot
    H_mf = rc.bed_mass_kg / (rc.rho_p_kg_m3 * (1.0 - rc.eps_mf) * A)
    Fair, Fvol = F_air.sum(), F_vol.sum()
    xi_g = np.linspace(0.0, 1.0, 41); Xc = release_cum(pat, xi_g)
    H_f = H_mf
    for _ in range(8):                                   # expanded height from the mean bubble fraction
        dl = [hydro_local(rc, A, Umf, (Fair + Fvol * x) / (ctot * A), s_ * H_f)[0] for s_, x in zip(xi_g, Xc)]
        H_f = H_mf / (1.0 - float(np.mean(dl)))
    NUT = nu_matrix(T_K).T
    rate = make_rates(T_K, kps)
    Dm = diffusivity_N2(T_K, P_abs)
    lam = 200.0 / H_f                                    # relaxation of the net-flow closure (numerical)
    tiny = 1e-12 * max(Fair, 1e-9)
    hom = 1.0 if opt.bubble_homogeneous else 0.0
    def split(y):
        F = np.maximum(y[:3 * NSP], 0.0).reshape(3, NSP); F[:, I_CS] = 0.0
        return F
    def rhs(z, y, N_char):
        F = split(y)
        Ft = F.sum(axis=1)
        Fbub = Ft[1] + Ft[2]
        U0 = (Ft[0] + Fbub) / (ctot * A)
        delta, d_b, u_br, u_b = hydro_local(rc, A, Umf, U0, z)
        yf = F / np.maximum(Ft, tiny)[:, None]
        Cm = yf * ctot
        r_v, r_c, r_d = rate(Cm)
        dF = np.zeros((3, NSP)); dC = 0.0
        if delta > 0.0 and Fbub > tiny:
            K = k_be(Umf, rc.eps_mf, d_b, u_br, Dm)
            for p in (1, 2):
                if Ft[p] <= tiny:
                    continue
                vol = A * delta * Ft[p] / Fbub           # m3 of this bubble phase per m of bed
                J = K * vol * (Cm[p] - Cm[0]); J[I_CS] = 0.0
                g = NUT @ (r_v[p] * (vol * hom)); dC += g[I_CS]; g[I_CS] = 0.0   # soot formed in bubbles joins the bed char
                dF[p] += g - J; dF[0] += J
        g = NUT @ (r_v[0] * (A * (1.0 - delta) * rc.eps_mf) + r_c[0] * (N_char / H_f) + r_d[0] * (W_dol / H_f))
        dC += g[I_CS]; g[I_CS] = 0.0
        dF[0] += g
        dF[2] += F_vol * (release_density(pat, z / H_f) / H_f)
        ex = (Ft[0] - Fe_max) * lam                      # two-phase theory: the emulsion carries the Umf flow only
        if ex > 0.0:
            fl = ex * yf[0]; dF[0] -= fl; dF[1] += fl
        elif Fbub > tiny:
            f_ = min(-ex, lam * Fbub) / Fbub; dF[0] += f_ * (F[1] + F[2]); dF[1] -= f_ * F[1]; dF[2] -= f_ * F[2]
        return np.concatenate([dF.ravel(), [dC]])
    fe0 = min(1.0, Fe_max / Fair)
    y0 = np.concatenate([F_air * fe0, F_air * (1.0 - fe0), np.zeros(NSP), [0.0]])
    atol = np.concatenate([np.full(3 * NSP, 1e-11 * Fair), [1e-11 * max(n_char_in, Fair)]])
    brk = sorted({float(x) * H_f for w, xp, Xp in pat for x in xp if 0.0 < x < 1.0})
    def integrate(N_char, dense=False):
        y, z0, segs = y0.copy(), 0.0, []
        for z1 in brk + [H_f]:                           # restart at the kinks of the release pattern
            sol = solve_ivp(rhs, [z0, z1], y, method="BDF", rtol=BED_RTOL, atol=atol, args=(N_char,), dense_output=dense)
            if not sol.success:
                raise RuntimeError(f"bed_two_phase: {sol.message}")
            y, z0 = sol.y[:, -1], z1; segs.append(sol)
        return y, segs
    resid = lambda N: n_char_in - N / rc.tau_char_s + integrate(N)[0][-1]
    N_hi = n_char_in * rc.tau_char_s
    r_hi = resid(N_hi); k_ = 0
    while r_hi > 0.0 and k_ < 6:                         # soot make-up larger than consumption: widen the bracket
        N_hi *= 2.0; r_hi = resid(N_hi); k_ += 1
    N_char = brentq(resid, 0.0, N_hi, xtol=1e-7 * N_hi, rtol=1e-6) if r_hi < 0.0 else N_hi
    y, segs = integrate(N_char, dense=profile)
    F = split(y); Fe, Fb, Feb = F
    out = dict(F_out=Fe + Fb + Feb, F_e=Fe, F_b=Fb, F_eb=Feb, N_char=N_char, F_char_out=n_char_in + y[-1],
               steady_resid=abs(n_char_in - N_char / rc.tau_char_s + y[-1]),
               tau_gas=(A * H_f - rc.bed_mass_kg / rc.rho_p_kg_m3) * ctot / (Fair + Fvol),
               O2_slip=(Fe + Fb + Feb)[IDX["O2"]] / F_air[IDX["O2"]], char_holdup_kg=N_char * AW["C"] / 1000.0)
    U_air = Fair / (ctot * A); U_exit = out["F_out"].sum() / (ctot * A)
    d_top = hydro_local(rc, A, Umf, U_exit, H_f)
    out["hyd"] = dict(A=A, U0=U_air, U0_exit=U_exit, Umf=Umf, U_Umf=U_air / Umf, U_Umf_exit=U_exit / Umf, Ar=Ar,
                      H_mf=H_mf, H_f=H_f, V_bed_gas=A * H_f - rc.bed_mass_kg / rc.rho_p_kg_m3,
                      H_fb=max(rc.H_total_m - H_f, 0.0), V_fb=A * max(rc.H_total_m - H_f, 0.0),
                      delta_top=d_top[0], db_top=d_top[1], db_over_D=d_top[1] / rc.D_m)
    if profile:
        zz = np.linspace(0.0, H_f, 121); Y = np.zeros((3 * NSP + 1, len(zz)))
        for sol in segs:
            m = (zz >= sol.t[0]) & (zz <= sol.t[-1]); Y[:, m] = sol.sol(zz[m])
        hl = [hydro_local(rc, A, Umf, np.maximum(Y[:3 * NSP, j], 0).sum() / (ctot * A), zz[j]) for j in range(len(zz))]
        out["profile"] = dict(z=zz, Y=np.maximum(Y[:3 * NSP], 0.0).reshape(3, NSP, -1),
                              delta=np.array([h[0] for h in hl]), d_b=np.array([h[1] for h in hl]))
    return out

# -----------------------------------------------------------------------------
# 11. SPLASHING ZONE
# -----------------------------------------------------------------------------
# 1-D three-phase bed only: the three bed streams (emulsion, bubbles, endogenous bubbles) are mixed instantaneously at
# the bed surface. Sebastiani's ghost-bubble zone (progressive mixing over the TDH) is not implemented.
def splash_mix(bed):
    return bed["F_out"].copy()

# -----------------------------------------------------------------------------
# 12. FREEBOARD (homogeneous PFR)
# -----------------------------------------------------------------------------
def freeboard_pfr(F_in, T_K, P_abs, A, L, kps, opt, T_out=None, T_nodes=None):
    """Homogeneous plug flow. Temperature: uniform (default), linear to T_out, or piecewise linear through T_nodes =
    (z above the bed surface in m, T in K) when a measured profile is available (option measured_freeboard)."""
    if L <= 0:
        return F_in.copy()
    T_out = T_K if T_out is None else T_out
    NUT0 = nu_matrix(T_K).T
    def rhs(z, F):
        T = float(np.interp(z, T_nodes[0], T_nodes[1])) if T_nodes is not None else T_K + (T_out - T_K) * z / L
        NUT = NUT0 if (T_out == T_K and T_nodes is None) else nu_matrix(T).T
        Fp = np.maximum(F, 0.0); Fp[I_CS] = 0.0
        Q = Fp.sum() * R_GAS * T / P_abs
        r_v, _ = rates(T, Fp / Q, kps)
        return NUT @ r_v * A
    sol = solve_ivp(rhs, [0.0, L], F_in, method="BDF",
                    rtol=min(opt.rtol, 1e-8), atol=1e-14 * F_in.sum())
    if not sol.success:
        raise RuntimeError(f"freeboard_pfr: {sol.message}")
    return np.maximum(sol.y[:, -1], 0.0)


# -----------------------------------------------------------------------------
# 13. ENERGY
# -----------------------------------------------------------------------------
# The bed temperature is IMPOSED (experimental value). The energy balance of the bed is evaluated afterwards
# (section 14d) and the heat it needs is charged to the process in the assessment.

# -----------------------------------------------------------------------------
# 14. SIMULATE
# -----------------------------------------------------------------------------
def stoich_air_Nm3_per_kg(fs):
    """Stoichiometric dry air (Nm3 per kg of fuel as received) from the ultimate analysis."""
    fe = feed_elements(fs, 1.0)
    o2 = fe["el"]["C"] + fe["el"]["H"] / 4 + fe["el"]["S"] - fe["el"]["O"] / 2
    return o2 / AIR_O2 * 3600 / MOL_PER_NM3

def simulate(fs, rc, op, kps, opt=SimulationOptions(), profile=False):
    T_K = op.T_C + 273.15
    P_abs = P_ATM + op.P_gauge_kPa * 1000.0                 # gauge -> absolute pressure (Pa); never use gauge in PV = nRT
    fe = feed_elements(fs, op.feed_kg_h)
    sur_ov = {**(SUR_MEASURED if opt.devol_rdf_measured else {}), **(SUR_MEASURED_WOOD if opt.devol_wood_measured else {}),
              **(SUR_MEASURED_OLIVE if opt.devol_olive_measured else {})}
    proxy = blend_proxy(fs, fe, sur_ov or None)
    if opt.component_char_kinetics:                          # option: per-component char-steam kinetics (off)
        cs_list, w_fb = char_steam_set(fs, fe)
        kps = replace(kps, char_steam=cs_list, char_steam_fallback_w=w_fb)
    else:                                                    # Sebastiani 2025 SI Table S6
        kps = replace(kps, char_steam=None, char_steam_fallback_w=1.0)
    if opt.fuel_char_kinetics:                               # literature char kinetics per fuel
        lit_, w_fb_ = char_lit_set(fs, fe, wood=opt.wood_char_kinetics)
        kps = replace(kps, char_lit=lit_, char_lit_fallback_w=w_fb_)
    if opt.wgs_equilibrium or opt.o2_priority != 1.0:
        mult = kps.multiplier.copy()
        if opt.wgs_equilibrium:
            mult[RXN_IDS.index("H-R01")] = 0.0
        for rid in ("H-R03", "H-R04"):
            mult[RXN_IDS.index(rid)] *= opt.o2_priority
        kps = replace(kps, multiplier=mult)
    F_devol, n_char_in = devolatilization(fe, proxy, allow_h2_closure=opt.devol_h2_closure,
                                          hc_closure=hc_closure_for(fs, fe) if opt.devol_hc_lump else None)
    devol_pair = DEVOL_LOG.get("pair")
    air = op.air_Nm3_h * MOL_PER_NM3 / 3600.0               # Nm3 at 273.15 K, 101325 Pa -> mol/s
    F_air = np.zeros(NSP); F_air[IDX["O2"]] = AIR_O2 * air; F_air[IDX["N2"]] = (1.0 - AIR_O2) * air
    F_in = F_devol + F_air
    W_dol = rc.bed_mass_kg if rc.tar_catalyst is not None else 0.0
    if opt.bed_model == "two_phase":
        bed = bed_two_phase(F_air, F_devol, n_char_in, T_K, P_abs, rc, kps, opt, W_dol,
                            release_pattern(fs, fe), profile=profile)
        hyd = bed["hyd"]; F = splash_mix(bed)
    elif opt.bed_model == "two_zone":                        # two-zone bed (formulation B, adopted)
        hyd = hydrodynamics(rc, T_K, P_abs, F_in.sum() * R_GAS * T_K / P_abs)
        bed = bed_two_zone(F_air, F_devol, n_char_in, T_K, P_abs, rc, kps, opt, W_dol, hyd)
        F = bed["F_out"]
    else:
        hyd = hydrodynamics(rc, T_K, P_abs, F_in.sum() * R_GAS * T_K / P_abs)
        kps_bed = kps
        if opt.co_ox_bed_factor != 1.0:                      # calibrated level only (section 14b); 1.0 in the predictive model
            m_bed = kps.multiplier.copy(); m_bed[RXN_IDS.index("H-R04")] *= opt.co_ox_bed_factor
            kps_bed = replace(kps, multiplier=m_bed)
        bed = bed_cstr(F_in, n_char_in, T_K, P_abs, hyd["V_bed_gas"], rc.tau_char_s, kps_bed, opt, W_dol=W_dol)
        F = bed["F_out"]
    if opt.wgs_equilibrium:
        F = wgs_equilibrate(F, T_K + opt.wgs_dT_K)
    F_bed_exit = F.copy()
    if opt.level >= 3:
        T_fb = T_K + opt.T_freeboard_offset_K
        T_fb_out = (opt.fb_cool[0] * (T_fb - 273.15) + opt.fb_cool[1] + 273.15) if opt.fb_cool else T_fb
        T_nodes = None
        if opt.measured_freeboard and (op.fb_profile is not None or op.T_fb_out_C is not None):
            # Option measured_freeboard (off): measured freeboard temperature as an INPUT (not a fit). The bed keeps the reported bed temperature.
            if op.fb_profile is not None:                      # thermocouples along the height (Pio 2020, Figs. 3-4)
                z_abs, T_abs = np.asarray(op.fb_profile[0], float), np.asarray(op.fb_profile[1], float) + 273.15
                zr = np.concatenate([[0.0], np.clip(z_abs - hyd["H_f"], 0.0, None)])
                Tr = np.concatenate([[T_fb], T_abs])
                keep = np.concatenate([[True], np.diff(zr) > 1e-9]); T_nodes = (zr[keep], np.minimum(Tr[keep], T_fb))
            else:                                              # only the outlet temperature is reported (FABER): linear profile
                T_nodes = (np.array([0.0, hyd["H_fb"]]), np.array([T_fb, min(op.T_fb_out_C + 273.15, T_fb)]))
            T_fb_out = float(np.interp(hyd["H_fb"], T_nodes[0], T_nodes[1]))
        # Option wgs_in_freeboard = False (off): water-gas shift only where solids are present (bed), not in the freeboard.
        # Source: the WGS law of the network (Biba et al. 1978, via Petersen & Werther 2005 Table 5) was derived for beds of
        #   coal/char, i.e. solids-catalysed. The homogeneous, uncatalysed WGS is orders of magnitude slower: with the gas-phase
        #   laws quoted in Tokmurzin et al. 2022, Table 2 (after Bustamante et al., AIChE J. 2004, 50, 1028 and 2005, 51, 1440;
        #   Ea/R = 36640 K forward, 39260 K reverse) the forward rate at 800-900 C is ~1e-4 - 1e-3 mol/m3/s, i.e. negligible
        #   in the seconds of residence time of a freeboard. Petersen & Werther also report that WGS stays far from equilibrium.
        # Equation: r_WGS = 0 in the freeboard; unchanged in the bed.   Validity: freeboards without significant solids hold-up.
        # Reason: physical (no catalyst in the freeboard), from the literature; not a fit.  Primary source not read directly: FLAGGED.
        kps_fb = kps
        if not opt.wgs_in_freeboard:
            m_fb = kps.multiplier.copy(); m_fb[RXN_IDS.index("H-R01")] = 0.0
            kps_fb = replace(kps, multiplier=m_fb)
        F = freeboard_pfr(F, T_fb, P_abs, hyd["A"], hyd["H_fb"], kps_fb, opt, T_out=T_fb_out, T_nodes=T_nodes)
        if opt.wgs_equilibrium and opt.wgs_final:
            F = wgs_equilibrate(F, T_fb_out + opt.wgs_dT_K)
        tau_fb = hyd["V_fb"] / (F.sum() * R_GAS * T_fb / P_abs)
    else:
        tau_fb = 0.0
    w_m = fe["moist_kg_s"] * 1000 / MW[IDX["H2O"]]
    el_in = np.array([fe["el"]["C"], fe["el"]["H"] + 2 * w_m, fe["el"]["O"] + w_m + 2 * AIR_O2 * air,
                      fe["el"]["N"] + 2 * (1 - AIR_O2) * air])
    F_char_C = bed["F_char_out"]
    el_out = ATOM.T @ F + np.array([F_char_C, 0, 0, 0])
    closure = 100.0 * el_out / el_in
    gas_tot = sum(F[IDX[s]] for s in GAS)
    y_wet = {s: 100.0 * F[IDX[s]] / gas_tot for s in GAS}                       # wet composition (all gas species)
    dry = {s: F[IDX[s]] for s in PERMANENT_GAS}                                   # dry, tar-free gas (as a GC sees it)
    dry_tot = sum(dry.values())
    y = {s: 100.0 * v / dry_tot for s, v in dry.items()}                          # normalised dry composition (incl. N2)
    Q_dry = dry_tot / MOL_PER_NM3 * 3600.0
    LHV = sum(y.get(s, 0.0) / 100.0 * v for s, v in LHV_NM3.items())
    feed_kg_s = op.feed_kg_h / 3600.0
    tar_g_s = sum(F[IDX[s]] * MW[IDX[s]] for s in TAR_SPECIES)
    tar_nb_g_s = tar_g_s - F[IDX["C6H6"]] * MW[IDX["C6H6"]]                      # gravimetric-like tar (benzene excluded)
    char_g_s = (F_char_C + F[I_CS]) * AW["C"] + fe["ash_kg_s"] * 1000.0
    LHV_fuel = fe["HHV_blend"] - 0.212 * fe["H_pct"] - 0.0245 * fe["M_pct"] - LHV_O_COEF * fe["O_pct"]
    CGE = 100.0 * Q_dry * LHV / (op.feed_kg_h * LHV_fuel)
    return dict(op=op, y=y, y_wet=y_wet, Q_dry_Nm3_h=Q_dry, LHV=LHV, C2=y["C2H4"] + y["C3H6"],
                tar_g_kg=tar_g_s / feed_kg_s, tar_g_Nm3=tar_g_s * 3600.0 / Q_dry,
                tar_nb_g_kg=tar_nb_g_s / feed_kg_s, tar_nb_g_Nm3=tar_nb_g_s * 3600.0 / Q_dry,
                char_g_kg=char_g_s / feed_kg_s, CGE=CGE, LHV_fuel=LHV_fuel,
                C_gas_pct=100.0 * sum(F[IDX[s]] * ATOM[IDX[s], 0] for s in PERMANENT_GAS) / el_in[0],
                FC_conv=100.0 * (1.0 - F_char_C / n_char_in) if n_char_in > 0 else np.nan,
                O2_conv=100.0 * (1 - F[IDX["O2"]] / F_in[IDX["O2"]]),
                ER=op.air_Nm3_h / (stoich_air_Nm3_per_kg(fs) * op.feed_kg_h),
                closure=dict(zip(ELEMENTS, closure)), devol_pair=devol_pair, hyd=hyd, bed=bed, tau_fb=tau_fb,
                F_out=F, F_in=F_in, F_bed_exit=F_bed_exit, fe=fe)


# -----------------------------------------------------------------------------
# 10. MODEL CONFIGURATION + shared helpers
# -----------------------------------------------------------------------------
# Elements of the adopted model, each one traced to a source and none fitted to the gasification datasets:
#   1. char-steam kinetics: generic law of Sebastiani 2025 SI Table S6 (fuel-specific laws optional, USE_FUEL_CHAR_STEAM)
#   2. olefin oxidation r12, r13, r15, r16: unit correction x 3.16e-5 -> Westbrook & Dryer 1981 (cm-s-mol units)
#   3. phenol reaction r20: steam order removed and reaction switched off -> Ji et al. 2009 Table 4; Morf et al. 2002
#   4. olive pomace char: CO2-gasification and combustion kinetics    -> Senneca 2007 Table 5
#   5. devolatilisation closure: no molecular H2; CH4:C2 = 1:0.75 for olive residues -> Christoforou 2018; Uzun 2007
#   6. WGS: kinetic law with f = 0.1                                   -> Petersen & Werther 2005 Table 5
#   7. RDF volatiles: measured SRF distribution                        -> Tokmurzin 2022 Table 7
#   8. H2 oxidation factor f = 0.001                                   -> Petersen & Werther 2005 Table 5
#   9. wood chars (pine, eucalyptus): CO2-gasification and combustion kinetics of wood-chip char -> Senneca 2007 Table 5
# Options kept but switched off: measured wood and olive pyrolysis yields (section 8), measured freeboard temperatures
#   (measured_freeboard) and no WGS in the freeboard (wgs_in_freeboard = False).
# LEVEL selects the bed formulation that produces the tables and figures:
#   "two_zone"   (default, adopted) formulation B: two-zone bed (bubble + emulsion), volatiles released in the emulsion,
#                predictive gas kinetics; only the in-bed dolomite activity (eta, section 3) is fitted, on the tar of the
#                reference case. The choice "emulsion" instead of
#                "bubble" was made by comparing both limiting cases with the datasets (Supporting Information, Table S10).
#   "calibrated" formulation A: perfectly mixed bed with the two global factors (comparison of Section 3.1).
#   "predictive" formulation A: perfectly mixed bed, nothing fitted (comparison of Section 3.1).
#   The eucalyptus tests of Parrillo 2021 took part in the choice "emulsion", so they are not a blind test for formulation B.
LEVEL = "two_zone"
# Optional: fuel-specific char-steam kinetics of olive pomace (Prestipino 2018) and RDF (Le & Kolaczkowski 2015), see the
# block of CHAR_STEAM_OP in section 5. False = adopted model (default). True = a DIFFERENT model: the calibration of
# section 14b and the validation were obtained with False and do not apply to it.
USE_FUEL_CHAR_STEAM = False
CAL_GLOBAL = dict(f_H2=0.1, f_CO_bed=0.1)          # global pair (refined grid search, section 14b). FROZEN:
#   the blind validation of section 14c was run once with this pair; changing it afterwards invalidates that test.
OPT_PRED = SimulationOptions(level=3, wgs_equilibrium=False, component_char_kinetics=USE_FUEL_CHAR_STEAM, devol_h2_closure=False,
                             devol_hc_lump=True, fuel_char_kinetics=True, devol_rdf_measured=True,
                             wood_char_kinetics=True)
KPS_PRED = kps_with_pw("H-R01", "H-R03", off=("H-R20",))
def kps_h2(f_H2):
    """Predictive kinetic set (formulation A) with the H2-oxidation factor f_H2 instead of 0.001."""
    k_ = kps_with_pw("H-R01", off=("H-R20",)); m_ = k_.multiplier.copy(); m_[RXN_IDS.index("H-R03")] *= f_H2
    return replace(k_, multiplier=m_)
if LEVEL == "two_zone":
    MODEL_NAME, FILE_TAG = "MODEL_27", "m27"
    MODEL_DESC = ("two-zone bed (bubble + emulsion), volatiles to the emulsion, predictive gas kinetics; "
                  "in-bed dolomite activity calibrated on the reference-case tar")
    OPT_REF, KPS_REF = replace(OPT_PRED, bed_model="two_zone", tz_volatiles_to="emulsion"), KPS_PRED
elif LEVEL == "calibrated":
    MODEL_NAME, FILE_TAG = "formulation Acal", "m25g"
    MODEL_DESC = f"perfectly mixed bed + global calibration (f_H2 = {CAL_GLOBAL['f_H2']:g}, f_CO_bed = {CAL_GLOBAL['f_CO_bed']:g}, same for all reactors)"
    OPT_REF, KPS_REF = replace(OPT_PRED, co_ox_bed_factor=CAL_GLOBAL["f_CO_bed"]), kps_h2(CAL_GLOBAL["f_H2"])
else:
    MODEL_NAME, FILE_TAG = "formulation A0", "m25"
    MODEL_DESC = "perfectly mixed bed, predictive level (nothing fitted to gasification data)"
    OPT_REF, KPS_REF = OPT_PRED, KPS_PRED
KPS_ORIGINAL = KPS                       # kinetic set without the corrections (kept for reference)
KPS, OPT_H1 = KPS_REF, OPT_REF           # names used by the rest of the notebook (and by the TEA/LCA sections)
# Role of each dataset: Terrados = development case; Calado, Pio, FABER = external validation.
# de Deus 2025 is not used: its published N and O balances do not close (116 % and 63-70 %).
DATASET_ROLE = {"Terrados 2026": "development", "Calado 2023": "validation", "Pio 2020": "validation",
                "FABER (Parrillo 2023)": "validation"}

def short_name(ds):
    return {"Terrados 2026": "Terrados", 
            "Calado 2023": "Calado",
            "Pio 2020": "Pio",
            "FABER (Parrillo 2023)": "FABER"}.get(ds, ds)

MAE_DS_ORDER = ["Terrados 2026", "Calado 2023", "Pio 2020", "FABER (Parrillo 2023)"]

print("=" * 78)
print(f"BFB gasification model - {MODEL_NAME}: {MODEL_DESC}")
print("=" * 78)

# -----------------------------------------------------------------------------
# 11. Run all datasets
# -----------------------------------------------------------------------------
MAIN_SP = ["H2", "CO", "CO2", "CH4", "C2H4", "N2"]

def run_points(name, feed_cfg, reactor, points, opt=OPT_H1):
    rows = []
    for p in points:
        fs = p.get("feed_cfg", feed_cfg)
        op = OperatingPoint(p["T"], p["P"], p["feed"], p["air"])
        try:
            s_ = simulate(fs, reactor, op, KPS, opt)
        except Exception as exc:
            print(f"  {name} T={p['T']}: FAILED ({exc})")
            continue
        rec = dict(dataset=name, point=p.get("label", f"T={p['T']:.0f}C"), T=p["T"])
        for k in MAIN_SP:
            rec[f"{k}_mod"] = s_["y"].get(k, 0.0)
            rec[f"{k}_exp"] = p["exp"].get(k, np.nan)
        rec["LHV_mod"] = s_["LHV"]; rec["LHV_exp"] = p["exp"].get("LHV", np.nan)
        rec["closeC"]  = s_["closure"]["C"]
        rec["closeAll"] = max(abs(v - 100.0) for v in s_["closure"].values())
        rows.append(rec)
    return pd.DataFrame(rows)

pts_A = [dict(T=T, P=r.P_gauge_kPa, feed=r.feed_kg_h, air=r.air_Nm3_h,
              exp={k: r[k] for k in MAIN_SP + ["LHV"]})
         for T, r in TABLE4.iterrows()]
pts_B = [dict(T=T, P=r.P_gauge_kPa, feed=r.feed_kg_h,
              air=air_n2_consistent(r, BLEND_TERRADOS),
              exp={k: r[k] for k in MAIN_SP + ["LHV"]})
         for T, r in TABLE4.iterrows()]
pts_C = [dict(T=T, P=r.P_gauge_kPa, feed=r.feed_kg_h, air=r.air_Nm3_h,
              exp={k: r[k] for k in MAIN_SP + ["LHV"]})
         for T, r in TABLE_CALADO.iterrows()]

DATASETS = {}
DATASETS["Terrados 2026"]               = run_points("Terrados 2026", BLEND_TERRADOS, REACTOR_TERRADOS, pts_B)
DATASETS["Calado 2023"]                 = run_points("Calado 2023", FEED_CALADO, REACTOR_CALADO, pts_C)


rows_pio = []
for p in PIO_POINTS:
    s_ = simulate(p["feed_cfg"], REACTOR_PIO,
                  OperatingPoint(p["T"], p["P"], p["feed"], p["air"], fb_profile=pio_fb_profile(p["T"])), KPS, OPT_H1)
    rec = dict(dataset="Pio 2020", point=p["label"], T=p["T"])
    for k in MAIN_SP:
        rec[f"{k}_mod"] = s_["y"].get(k, 0.0)
        rec[f"{k}_exp"] = p["exp"].get(k, np.nan)
    rec["LHV_mod"] = s_["LHV"]; rec["LHV_exp"] = p["exp"].get("LHV", np.nan)
    rec["closeC"]  = s_["closure"]["C"]
    rec["closeAll"] = max(abs(v - 100.0) for v in s_["closure"].values())
    rows_pio.append(rec)
DATASETS["Pio 2020"] = pd.DataFrame(rows_pio)

rows_f = []
for _, r in TABLE_FABER.iterrows():
    s_ = simulate(faber_feed(r.PL, r.BM), REACTOR_FABER,
                  OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h, T_fb_out_C=r.T_fb_out_C), KPS, OPT_H1)
    rec = dict(dataset="FABER (Parrillo 2023)", point=r.test, T=r.T_bed_C)
    for k in MAIN_SP:
        rec[f"{k}_mod"] = s_["y"].get(k, 0.0)
        rec[f"{k}_exp"] = r.get(k, np.nan) if k in r else np.nan
    rec["LHV_mod"] = s_["LHV"]; rec["LHV_exp"] = r.LHV          # Parrillo 2023, Table 3
    rec["closeC"]  = s_["closure"]["C"]
    rec["closeAll"] = max(abs(v - 100.0) for v in s_["closure"].values())
    rows_f.append(rec)
DATASETS["FABER (Parrillo 2023)"] = pd.DataFrame(rows_f)

# --- element balance check (carbon) over every simulated point ---
_close = pd.concat([df[["dataset", "point", "closeC"]] for df in DATASETS.values()], ignore_index=True)
_worst = _close.loc[(_close.closeC - 100).abs().idxmax()] if len(_close) else None
print(f"\n  Carbon balance: max |closure - 100| = {np.abs(_close.closeC - 100).max():.4f} % "
      f"(worst: {_worst.dataset} {_worst.point})")
if np.abs(_close.closeC - 100).max() > 0.5:
    print("  WARNING: carbon balance error above 0.5 % - check solver tolerances")
print(f"  Element balances (C, H, O, N): max |closure - 100| = {max(df.closeAll.max() for df in DATASETS.values()):.4f} %")
print("  LHV MAE (MJ/Nm3): " + ", ".join(
    f"{short_name(ds)} {np.mean(np.abs(df.LHV_mod - df.LHV_exp)[df.LHV_exp.notna()]):.2f}"
    for ds, df in DATASETS.items() if "LHV_exp" in df and df.LHV_exp.notna().any()))
print("  LHV MAPE (%):     " + ", ".join(
    f"{short_name(ds)} {100 * np.mean((np.abs(df.LHV_mod - df.LHV_exp) / df.LHV_exp)[df.LHV_exp.notna()]):.1f}"
    for ds, df in DATASETS.items() if "LHV_exp" in df and df.LHV_exp.notna().any()))

# -----------------------------------------------------------------------------
# 11b. MAE summary (H2, CO, CO2, CH4)
# -----------------------------------------------------------------------------
MAE_SPECIES = ["H2", "CO", "CO2", "CH4"]

def mae_of(df, species=MAE_SPECIES):
    errs = []
    for k in species:
        col_e = f"{k}_exp"
        if col_e in df:
            valid = df[col_e].notna()
            if valid.any():
                errs += list(np.abs(df.loc[valid, f"{k}_mod"] - df.loc[valid, col_e]))
    return float(np.mean(errs)) if errs else np.nan

print("\n  MAE per dataset and species (pp; |model - exp|). Note: in Terrados 2026 the N2 fraction and syngas")
print("  flow were used to build the air input, so N2 is not a validation target (N2 is not in the MAE).")
mae_matrix = pd.DataFrame(index=list(DATASETS.keys()), columns=MAE_SPECIES, dtype=float)
for ds, df in DATASETS.items():
    for k in MAE_SPECIES:
        col_e = f"{k}_exp"
        if col_e in df:
            valid = df[col_e].notna()
            if valid.any():
                mae_matrix.loc[ds, k] = float(np.mean(np.abs(df.loc[valid, f"{k}_mod"] - df.loc[valid, col_e])))
    mae_matrix.loc[ds, "MAE_mean"] = mae_matrix.loc[ds, MAE_SPECIES].mean()
print(mae_matrix.round(2).to_string())

# MAPE = 100/N * sum |model - exp| / exp  (relative error; target metric of the project). All experimental values of the four
# species are > 2 vol%, so the ratio is well defined; no epsilon is added.
def mape_of(df, species=MAE_SPECIES):
    """Macro-MAPE: mean over the species of the per-species MAPE."""
    vals = []
    for k in species:
        col_e = f"{k}_exp"
        if col_e in df and df[col_e].notna().any():
            v = df[col_e].notna()
            vals.append(100.0 * float(np.mean(np.abs(df.loc[v, f"{k}_mod"] - df.loc[v, col_e]) / df.loc[v, col_e])))
    return float(np.mean(vals)) if vals else np.nan

print("\n  MAPE per dataset and species (%; 100 |model - exp| / exp). MAPE_mean = mean of the four species.")
mape_matrix = pd.DataFrame(index=list(DATASETS.keys()), columns=MAE_SPECIES, dtype=float)
for ds, df in DATASETS.items():
    for k in MAE_SPECIES:
        col_e = f"{k}_exp"
        if col_e in df and df[col_e].notna().any():
            v = df[col_e].notna()
            mape_matrix.loc[ds, k] = 100.0 * float(np.mean(np.abs(df.loc[v, f"{k}_mod"] - df.loc[v, col_e]) / df.loc[v, col_e]))
    mape_matrix.loc[ds, "MAPE_mean"] = mape_matrix.loc[ds, MAE_SPECIES].mean()
mape_matrix["role"] = [DATASET_ROLE.get(ds, "") for ds in mape_matrix.index]
print(mape_matrix.round(1).to_string())
print("  The MAPE is reported for completeness; it is not the acceptance criterion (relative errors are inflated when the")
print("  measured value is small: Hyndman & Koehler 2006, Int. J. Forecast. 22, 679-688).")
# Acceptance criterion adopted by the authors: RMSE <= RMSE_LIMIT percentage points for each species.
RMSE_LIMIT = 8.0   # acceptance criterion of the article (pp): RMSE of each species
MAE_LIMIT = 5.0    # acceptance criterion of the article (pp): mean MAE of H2, CO, CO2 and CH4 in each dataset
rmse_matrix = pd.DataFrame(index=list(DATASETS.keys()), columns=MAE_SPECIES, dtype=float)
for ds_, df_ in DATASETS.items():
    for k_ in MAE_SPECIES:
        if f"{k_}_exp" in df_.columns and f"{k_}_mod" in df_.columns:
            v_ = df_[f"{k_}_exp"].notna()
            if v_.any():
                rmse_matrix.loc[ds_, k_] = float(np.sqrt(np.mean((df_.loc[v_, f"{k_}_mod"] - df_.loc[v_, f"{k_}_exp"]) ** 2)))
rmse_matrix["RMSE_max"] = rmse_matrix[MAE_SPECIES].max(axis=1)
rmse_matrix["meets_limit"] = rmse_matrix["RMSE_max"] <= RMSE_LIMIT
print("\n  RMSE per dataset and species (percentage points):")
print(rmse_matrix.round(2).to_string())
_over = [f"{short_name(d_)} {k_} ({rmse_matrix.loc[d_, k_]:.2f})" for d_ in rmse_matrix.index for k_ in MAE_SPECIES if rmse_matrix.loc[d_, k_] > RMSE_LIMIT]
print(f"  Acceptance criterion: RMSE <= {RMSE_LIMIT:g} pp for each species. Within the limit: "
      f"{int((rmse_matrix[MAE_SPECIES] <= RMSE_LIMIT).sum().sum())} of {int(rmse_matrix[MAE_SPECIES].notna().sum().sum())} dataset-species combinations; "
      f"exceeding it: {', '.join(_over) or 'none'}.")
rmse_matrix.to_csv(DATA_OUT / (FILE_TAG + "_rmse_per_dataset.csv"))
REFERENCE_MAPE = {ds: mape_of(df) for ds, df in DATASETS.items()}

REFERENCE_MAE = {ds: mae_of(df) for ds, df in DATASETS.items()}

# -----------------------------------------------------------------------------
# 11c. Summary tables - exp vs model per species (dataset + global)
# -----------------------------------------------------------------------------
SUMMARY_SPECIES = ["H2", "CO", "CO2", "CH4"]

def error_metrics(e, m):
    """Validation metrics for paired experimental (e) and model (m) values.
       MAE, RMSE, bias : absolute errors in pp (bias = mean signed error, shows systematic over/under-prediction;
                         RMSE penalises large deviations, so RMSE >> MAE flags outliers);
       MAPE            : mean absolute percentage error (relative error of each point);
       NRMSE           : 100 RMSE / mean(exp), the coefficient of variation of the RMSE: scale-free like the MAPE but not
                         inflated by individual small experimental values;
       within20        : % of points whose relative error is <= 20 % (the band usually drawn in parity plots)."""
    d = m - e
    return dict(MAE=float(np.mean(np.abs(d))), RMSE=float(np.sqrt(np.mean(d ** 2))), bias=float(np.mean(d)),
                MAPE=float(100 * np.mean(np.abs(d) / e)), NRMSE=float(100 * np.sqrt(np.mean(d ** 2)) / np.mean(e)),
                within20=float(100 * np.mean(np.abs(d) / e <= 0.20)))

def _stats_row(df, ds_name, sp):
    col_e, col_m = f"{sp}_exp", f"{sp}_mod"
    if col_e not in df:
        return None
    v = df[col_e].notna()
    if not v.any():
        return None
    e = df.loc[v, col_e].values
    m = df.loc[v, col_m].values
    d = m - e
    return dict(dataset=short_name(ds_name), species=sp, n=int(v.sum()),
                exp_mean=float(np.mean(e)), mod_mean=float(np.mean(m)), **error_metrics(e, m))

rows1 = []
for ds in MAE_DS_ORDER:
    if ds not in DATASETS: continue
    df = DATASETS[ds]
    for sp in SUMMARY_SPECIES:
        r = _stats_row(df, ds, sp)
        if r is not None:
            rows1.append(r)
TABLE_DETAIL = pd.DataFrame(rows1).set_index(["dataset", "species"])
print("\n" + "=" * 78)
print("SUMMARY TABLE 1 - exp vs model, per dataset and species (MAE, RMSE, bias in pp, dry basis; MAPE, NRMSE, within20 in %)")
print("=" * 78)
print(TABLE_DETAIL.round(2).to_string())

# Global aggregate over the external validation datasets (Terrados is the development dataset and is excluded).
GLOBAL_DS_ORDER = [d for d in MAE_DS_ORDER if DATASET_ROLE.get(d) == "validation"]
rows2 = []
for sp in SUMMARY_SPECIES:
    all_e, all_m = [], []
    for ds in GLOBAL_DS_ORDER:
        if ds not in DATASETS: continue
        df = DATASETS[ds]
        col_e, col_m = f"{sp}_exp", f"{sp}_mod"
        if col_e not in df: continue
        v = df[col_e].notna()
        if not v.any(): continue
        all_e += list(df.loc[v, col_e].values)
        all_m += list(df.loc[v, col_m].values)
    if not all_e: continue
    all_e, all_m = np.array(all_e), np.array(all_m)
    rows2.append(dict(species=sp, n=len(all_e), exp_mean=float(np.mean(all_e)), mod_mean=float(np.mean(all_m)),
                      **error_metrics(all_e, all_m)))
TABLE_GLOBAL = pd.DataFrame(rows2).set_index("species")
print("\n" + "=" * 78)
print("SUMMARY TABLE 2 - exp vs model, pooled over the external validation datasets (Calado, Pio, FABER)")
print("=" * 78)
print(TABLE_GLOBAL.round(2).to_string())

TABLE_DETAIL.to_csv(DATA_OUT / (FILE_TAG + "_summary_by_dataset_species.csv"))
TABLE_GLOBAL.to_csv(DATA_OUT / (FILE_TAG + "_summary_by_species.csv"))
print(f"\n  Saved: {DATA_OUT / (FILE_TAG + '_summary_by_dataset_species.csv')}")
print(f"  Saved: {DATA_OUT / (FILE_TAG + '_summary_by_species.csv')}")

# -----------------------------------------------------------------------------
# 11d. Summary tables - one per author, per temperature and species
# -----------------------------------------------------------------------------
def build_author_table(ds_key):
    df = DATASETS[ds_key]
    rows = []
    for _, row in df.iterrows():
        T = int(round(row["T"]))
        rec = {"T_C": T if str(row["point"]).startswith("T=") else f"{row['point']} ({T} C)"}
        for sp in SUMMARY_SPECIES:
            col_e = f"{sp}_exp"
            if col_e not in df or pd.isna(row.get(col_e, np.nan)):
                rec[(sp, "exp")]   = np.nan
                rec[(sp, "model")] = np.nan
                rec[(sp, "MAE")]   = np.nan
                rec[(sp, "APE%")]  = np.nan
            else:
                rec[(sp, "exp")]   = float(row[col_e])
                rec[(sp, "model")] = float(row[f"{sp}_mod"])
                rec[(sp, "MAE")]   = abs(float(row[f"{sp}_mod"]) - float(row[col_e]))
                rec[(sp, "APE%")]  = 100.0 * abs(float(row[f"{sp}_mod"]) - float(row[col_e])) / float(row[col_e])
        rows.append(rec)
    tbl = pd.DataFrame(rows).set_index("T_C")
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns, names=["species", "stat"])
    tbl = tbl.reindex(columns=pd.MultiIndex.from_product(
        [SUMMARY_SPECIES, ["exp", "model", "MAE", "APE%"]]))
    mean_rec = {}
    for sp in SUMMARY_SPECIES:
        mean_rec[(sp, "exp")]   = tbl[(sp, "exp")].mean()
        mean_rec[(sp, "model")] = tbl[(sp, "model")].mean()
        mean_rec[(sp, "MAE")]   = tbl[(sp, "MAE")].mean()
        mean_rec[(sp, "APE%")]  = tbl[(sp, "APE%")].mean()          # = MAPE of the species in this dataset
    tbl.loc["MEAN", :] = pd.Series(mean_rec)
    return tbl

ALL_DS_ORDER = ["Terrados 2026", "Calado 2023", "Pio 2020", "FABER (Parrillo 2023)"]
TABLES_BY_AUTHOR = {}
for ds_key in ALL_DS_ORDER:
    if ds_key not in DATASETS: continue
    name = short_name(ds_key)
    tbl = build_author_table(ds_key)
    TABLES_BY_AUTHOR[name] = tbl
    print("\n" + "=" * 88)
    print(f"SUMMARY TABLE - {name} ({DATASET_ROLE.get(ds_key, '')}); APE% = absolute percentage error, its MEAN is the MAPE")
    print("=" * 88)
    print(tbl.round(2).to_string())
    fname = FILE_TAG + f"_summary_{name.replace(' ', '_').replace('/', '_')}.csv"
    tbl.to_csv(DATA_OUT / fname)
    print(f"  Saved: {DATA_OUT / fname}")

# -----------------------------------------------------------------------------
# 12. Composition bars per dataset (species colors + exp/model encoding)
# -----------------------------------------------------------------------------
print("\n  Plotting: composition bars per dataset ...")

from matplotlib.patches import Patch

BAR_SPECIES = ["H2", "CO", "CO2", "CH4"]
SPECIES_COLOR = {"H2": "tab:blue", "CO": "tab:orange",
                 "CO2": "tab:green", "CH4": "tab:red"}
SPECIES_HANDLES = [Patch(facecolor=SPECIES_COLOR[sp], edgecolor="black", label=sp)
                   for sp in BAR_SPECIES]
ENCODING_HANDLES = [
    Patch(facecolor="grey", edgecolor="grey", alpha=0.30, hatch="//", label="exp"),
    Patch(facecolor="grey", edgecolor="black", alpha=1.00,             label="Model"),
]
LEGEND_HANDLES = SPECIES_HANDLES + ENCODING_HANDLES

bar_ds = ["Calado 2023", "Terrados 2026", "Pio 2020", "FABER (Parrillo 2023)"]

for ds in bar_ds:
    if ds not in DATASETS: continue
    df = DATASETS[ds].reset_index(drop=True)
    n_pts = len(df)
    ncols = min(n_pts, 5)
    nrows = int(np.ceil(n_pts / ncols))
    panel_w, panel_h, top_strip = 3.4, 3.6, 1.1
    fig_h = top_strip + panel_h * nrows
    fig_w = panel_w * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    top_frac = 1.0 - (top_strip / fig_h)
    fig.subplots_adjust(top=top_frac, bottom=0.10, left=0.06, right=0.98,
                        hspace=0.45, wspace=0.30)
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i // ncols, i % ncols]
        x = np.arange(len(BAR_SPECIES))
        for j, sp in enumerate(BAR_SPECIES):
            c = SPECIES_COLOR[sp]
            exp_v = row[f"{sp}_exp"]; mod_v = row[f"{sp}_mod"]
            if pd.isna(exp_v): exp_v = 0.0
            ax.bar(j - 0.20, exp_v, 0.38, color=c, alpha=0.30,
                   edgecolor=c, linewidth=1.4, hatch="//")
            ax.bar(j + 0.20, mod_v, 0.38, color=c,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(BAR_SPECIES, fontsize=9)
        ax.set_title(f"{row['point']}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    for j in range(n_pts, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"{short_name(ds)} - Syngas Composition: "
                 f"experimental vs Model", fontsize=12, y=0.985)
    fig.legend(handles=LEGEND_HANDLES, loc="upper center",
               bbox_to_anchor=(0.5, 0.945), ncol=len(LEGEND_HANDLES),
               frameon=False, fontsize=10, columnspacing=1.2, handletextpad=0.5)
    plt.savefig(DATA_OUT / (FILE_TAG + f"_bars_{short_name(ds)}.png"), dpi=140, bbox_inches=None)
    plt.show()

# -----------------------------------------------------------------------------
# 13. MAE per dataset (colored bars, author names)
# -----------------------------------------------------------------------------
print("  Plotting: MAE per dataset (colored bars, author names) ...")
DATASET_COLOR = {
    "Calado 2023":           "tab:blue",
    "Terrados 2026": "tab:red",
    "Pio 2020":              "tab:purple",
     "FABER (Parrillo 2023)": "tab:gray",
}
mae_vals = [REFERENCE_MAE[d] for d in MAE_DS_ORDER if d in REFERENCE_MAE]
labels   = [short_name(d) for d in MAE_DS_ORDER if d in REFERENCE_MAE]
colors   = [DATASET_COLOR[d] for d in MAE_DS_ORDER if d in REFERENCE_MAE]

mape_vals = [REFERENCE_MAPE[d] for d in MAE_DS_ORDER if d in REFERENCE_MAPE]
fig, axes13 = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
y = np.arange(len(labels))
for ax, vals, xl, off, fmt in ((axes13[0], mae_vals, "MAE (pp), H2+CO+CO2+CH4", 0.05, "{:.2f}"),
                               (axes13[1], mape_vals, "MAPE (%), mean of H2, CO, CO2, CH4", 0.5, "{:.1f}")):
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v + off, i, fmt.format(v), va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xl); ax.grid(alpha=0.3, axis="x")
axes13[0].invert_yaxis()
fig.suptitle(f"{MODEL_NAME}: MAE and MAPE per dataset")
plt.tight_layout()
plt.savefig(DATA_OUT / (FILE_TAG + "_mae_per_dataset.png"), dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 14. Deviation per species, one panel per author (bias +/- std)
# -----------------------------------------------------------------------------
print("  Plotting: per-author deviation per species (bias +/- std) ...")
DEV_DS = MAE_DS_ORDER
DEV_SPECIES = ["H2", "CO", "CO2", "CH4"]
DEV_COLOR = {"H2": "tab:blue", "CO": "tab:orange",
             "CO2": "tab:green", "CH4": "tab:red"}

def bias_std(df):
    out = {}
    for sp in DEV_SPECIES:
        col_e = f"{sp}_exp"; col_m = f"{sp}_mod"
        if col_e not in df:
            out[sp] = (np.nan, np.nan, 0); continue
        v = df[col_e].notna()
        if not v.any():
            out[sp] = (np.nan, np.nan, 0); continue
        d = (df.loc[v, col_m] - df.loc[v, col_e]).values
        out[sp] = (float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0, len(d))
    return out

n = len(DEV_DS)
fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2), sharey=True)
if n == 1: axes = [axes]
for ax, ds in zip(axes, DEV_DS):
    if ds not in DATASETS: continue
    stats = bias_std(DATASETS[ds])
    xs = np.arange(len(DEV_SPECIES))
    means = [stats[sp][0] for sp in DEV_SPECIES]
    stds  = [stats[sp][1] for sp in DEV_SPECIES]
    colors = [DEV_COLOR[sp] for sp in DEV_SPECIES]
    ax.bar(xs, means, 0.62, color=colors, edgecolor="black", linewidth=0.5,
           yerr=stds, capsize=5, error_kw=dict(elinewidth=1.2, ecolor="black"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(DEV_SPECIES, fontsize=9)
    nmax = max(stats[sp][2] for sp in DEV_SPECIES)
    nmin = min(stats[sp][2] for sp in DEV_SPECIES)
    ax.set_title(f"{short_name(ds)} (n={nmin}-{nmax})", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for x, m, s in zip(xs, means, stds):
        if np.isnan(m): continue
        ax.text(x, m + (0.05 if m >= 0 else -0.05) + (s if s else 0),
                f"{m:+.1f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=8)
axes[0].set_ylabel("bias = model - exp (pp)")
fig.suptitle("Per-author deviation per species- "
             "bars: mean bias; error bars: +/-1 std across the dataset",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(DATA_OUT / (FILE_TAG + "_deviation_per_author.png"), dpi=150, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# 14b. CALIBRATED LEVEL (two parameters per reactor) - reported SEPARATELY from the predictive level above
# -----------------------------------------------------------------------------
# The sections above use the level selected with LEVEL. This section adds, for formulation A, a clearly separated
# calibrated level in which two parameters are fitted (never per operating point, never per dataset of
# the same reactor). Both act on how the oxygen is shared inside the bed:
#   f_H2     : factor of the H2 oxidation law (H-R03). Predictive value 0.001 = the factor that Petersen & Werther (2005,
#              Table 5) fitted to their own riser, i.e. a reactor-specific number already in its source.
#   f_CO_bed : factor of the CO oxidation law (H-R04) INSIDE THE BED only (1 in the freeboard). Physical meaning: homogeneous
#              oxidation does not take place in the particulate phase of a fluidised bed. Sources (both read):
#              Hayhurst & Tucker (1990), Combust. Flame 79, 175-189: in a bed of silica sand at up to 1000 C, CO and O2
#              "do not react in the particulate (emulsion) phase, but do react in the bubble phase as well as in the
#              freeboard"; the inhibition is chemical (radical recombination on the solids), not a heat-transfer effect.
#              Hayhurst (1991), Combust. Flame 85, 155-168: mixtures of C3H8, CH4 or CO with air at up to 1000 C do not
#              burn between the sand particles; combustion occurs in the bubbles or above the bed.
#              The Dryer-Glassman CO law of the network was measured without solids. f_CO_bed < 1 is therefore physical;
#              its VALUE (the share of the bed gas that behaves as bubble phase) is what is calibrated.
# Reason: in the predictive model the bed (one CSTR) burns CO (49-74 % of the O2 with wood fuels) and the gas leaves the
#   bed with ~3 % CO, against 13-19 % measured at the outlet.
# Method: grid of 28 pairs (decade steps over 0.001-1, refined to half-decade steps around the optimum); objective =
#   macro-MAPE of H2, CO, CO2, CH4. The global optimum did not move with the refinement: f_CO_bed = 0.1 is well defined
#   (0.03 and 0.3 are both worse), f_H2 is weakly defined between 0.1 and 1 (mean of the four datasets 20.0-20.1 %).
#   Portalegre: fitted on Terrados (5 points) and TESTED on Calado (3 points, same reactor, not used in the fit).
#   Aveiro (Pio, 15 points) and FABER (14 points): leave-one-out cross-validation (each point is predicted with the
#   parameters fitted to the other points). The in-sample value is optimistic; the cross-validated one is the reference figure.
# Not calibrated: kinetic constants per dataset or per point, fuel properties, devolatilisation, residence times.
CAL_GRID = sorted(set([(a, b) for a in (1e-3, 1e-2, 1e-1, 1.0) for b in (1e-3, 1e-2, 1e-1, 10 ** -0.5, 1.0)] +
                      [(a, b) for a in (10 ** -1.5, 1e-1, 10 ** -0.5, 1.0) for b in (10 ** -1.5, 1e-1, 10 ** -0.5)]))
# MAIN calibrated version: the GLOBAL pair CAL_GLOBAL (section 10), fitted to the four datasets together. Its out-of-sample
# estimate is the leave-one-reactor-out value: the pair is fitted WITHOUT the datasets of one reactor and applied to them.
CAL_LORO = {"Terrados 2026": 9.3, "Calado 2023": 23.3, "Pio 2020": 35.9, "FABER (Parrillo 2023)": 20.1}   # stored leave-one-reactor-out result
# COMPLEMENTARY: one pair per reactor (Portalegre fitted on Terrados and tested on Calado; Pio and FABER leave-one-out).
CAL_PARAMS = {"Portalegre": dict(f_H2=1.0, f_CO_bed=0.1), "Aveiro": dict(f_H2=1.0, f_CO_bed=10 ** -1.5),
              "FABER": dict(f_H2=0.01, f_CO_bed=10 ** -0.5)}
CAL_CV = {"Pio 2020": 28.5, "FABER (Parrillo 2023)": 15.5}             # leave-one-out macro-MAPE, same run
RUN_CALIBRATION = False        # True repeats the grid search and all cross-validations (~17 min) and overwrites the values

def cal_points():
    """dataset -> (reactor label, list of (feedstock, reactor, operating point, experimental dict))."""
    P = {}
    P["Terrados 2026"] = ("Portalegre", [(BLEND_TERRADOS, REACTOR_TERRADOS,
                          OperatingPoint(T, r.P_gauge_kPa, r.feed_kg_h, air_n2_consistent(r, BLEND_TERRADOS)),
                          {k: r[k] for k in MAE_SPECIES}) for T, r in TABLE4.iterrows()])
    P["Calado 2023"] = ("Portalegre", [(FEED_CALADO, REACTOR_CALADO, OperatingPoint(T, r.P_gauge_kPa, r.feed_kg_h, r.air_Nm3_h),
                        {k: r[k] for k in MAE_SPECIES}) for T, r in TABLE_CALADO.iterrows()])
    P["Pio 2020"] = ("Aveiro", [(p["feed_cfg"], REACTOR_PIO, OperatingPoint(p["T"], p["P"], p["feed"], p["air"]),
                     {k: p["exp"][k] for k in MAE_SPECIES}) for p in PIO_POINTS])
    P["FABER (Parrillo 2023)"] = ("FABER", [(faber_feed(r.PL, r.BM), REACTOR_FABER,
                                  OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h),
                                  {k: r[k] for k in MAE_SPECIES}) for _, r in TABLE_FABER.iterrows()])
    return P

def cal_eval(points, f_H2, f_CO_bed):
    """Absolute percentage errors, array (n_points, 4 species), for one pair of reactor parameters."""
    kps_ = kps_with_pw("H-R01", off=("H-R20",)); m_ = kps_.multiplier.copy(); m_[RXN_IDS.index("H-R03")] *= f_H2
    kps_ = replace(kps_, multiplier=m_); opt_ = replace(OPT_PRED, co_ox_bed_factor=f_CO_bed)
    out = []
    for fs_, rc_, op_, e_ in points:
        y_ = simulate(fs_, rc_, op_, kps_, opt_)["y"]
        out.append([100.0 * abs(y_[k] - e_[k]) / e_[k] for k in MAE_SPECIES])
    return np.array(out)

CAL_PTS = cal_points()
if RUN_CALIBRATION:
    _grid = list(CAL_GRID)
    _ape = {ds: {g: cal_eval(pts, *g) for g in _grid} for ds, (_, pts) in CAL_PTS.items()}
    _best = lambda ds, idx: min(_grid, key=lambda g: _ape[ds][g][idx].mean(axis=0).mean())
    _all = lambda ds: list(range(len(CAL_PTS[ds][1])))
    g_ = _best("Terrados 2026", _all("Terrados 2026")); CAL_PARAMS["Portalegre"] = dict(f_H2=g_[0], f_CO_bed=g_[1])
    _mac = lambda d, g: _ape[d][g].mean(axis=0).mean()
    g_ = min(_grid, key=lambda g: np.mean([_mac(d, g) for d in CAL_PTS]))
    print(f"  grid search: best global pair f_H2 = {g_[0]:g}, f_CO_bed = {g_[1]:g} (in use: {CAL_GLOBAL}); edit CAL_GLOBAL in section 10 to change it")
    for lab_ in ("Portalegre", "Aveiro", "FABER"):                         # leave one reactor out
        out_ = [d for d in CAL_PTS if CAL_PTS[d][0] == lab_]; in_ = [d for d in CAL_PTS if d not in out_]
        gl_ = min(_grid, key=lambda g: np.mean([_mac(d, g) for d in in_]))
        for d in out_:
            CAL_LORO[d] = float(_mac(d, gl_))
    for ds, lab in (("Pio 2020", "Aveiro"), ("FABER (Parrillo 2023)", "FABER")):
        g_ = _best(ds, _all(ds)); CAL_PARAMS[lab] = dict(f_H2=g_[0], f_CO_bed=g_[1])
        held = np.array([_ape[ds][_best(ds, [j for j in _all(ds) if j != i])][i] for i in _all(ds)])
        CAL_CV[ds] = float(held.mean(axis=0).mean())

rows_cal = []
for ds, (lab, pts) in CAL_PTS.items():
    pred_ = REFERENCE_MAPE[ds] if LEVEL == "predictive" else float(cal_eval(pts, 1e-3, 1.0).mean(axis=0).mean())
    glob_ = REFERENCE_MAPE[ds] if LEVEL == "calibrated" else float(cal_eval(pts, **CAL_GLOBAL).mean(axis=0).mean())
    reac_ = float(cal_eval(pts, **CAL_PARAMS[lab]).mean(axis=0).mean())
    rows_cal.append(dict(dataset=short_name(ds), predictive=pred_, global_pair=glob_, global_leave_reactor_out=CAL_LORO[ds],
                         per_reactor=reac_, per_reactor_cross_validated=CAL_CV.get(ds, np.nan),
                         reactor=lab, f_H2_reactor=f"{CAL_PARAMS[lab]['f_H2']:g}", f_CO_bed_reactor=f"{CAL_PARAMS[lab]['f_CO_bed']:.2g}"))
TABLE_CAL = pd.DataFrame(rows_cal).set_index("dataset")
print("\n" + "=" * 78)
print("TWO-LEVEL SUMMARY - macro-MAPE (%) of H2, CO, CO2, CH4")
print("=" * 78)
print(TABLE_CAL.round(1).to_string())
print("  predictive               : formulation A, nothing fitted (f_H2 = 0.001, f_CO_bed = 1).")
print(f"  global_pair              : MAIN calibrated version. One pair for all reactors: f_H2 = {CAL_GLOBAL['f_H2']:g}, f_CO_bed = {CAL_GLOBAL['f_CO_bed']:g}.")
print("  global_leave_reactor_out : the pair is fitted without the datasets of that reactor and then applied to them (out-of-sample estimate).")
print("  per_reactor              : complementary. One pair per reactor (last two columns); Calado = test of the pair fitted on Terrados.")
print("  per_reactor_cross_validated: leave-one-out inside the reactor.")
TABLE_CAL.to_csv(DATA_OUT / (FILE_TAG + "_two_level_summary.csv"))


# -----------------------------------------------------------------------------
# 14c. BLIND VALIDATION - Parrillo et al. (2021), both levels frozen
# -----------------------------------------------------------------------------
# Source: Parrillo, Ardolino, Cali, Marotto, Pettinau, Arena (2021), Energy 219, 119604, Table 4: air gasification of
#   eucalyptus chips in the FABER pilot bubbling fluidised bed (olivine bed), ten tests at ER 0.21-0.35.
# These data were NOT used to build, select or calibrate formulation A. They were run ONCE with the predictive and the
# calibrated levels of formulation A, both frozen beforehand. Nothing was changed afterwards.
# Exception: the test at ER = 0.25 is the eucalyptus-only row already present in the FABER 2023 dataset (same feed rate,
#   air/fuel ratio, bed temperature and composition), so it is NOT blind and is excluded from the blind metrics.
# Inputs: fuel = the eucalyptus chips already defined for FABER; air = (air/fuel ratio) x feed / 1.293 kg/Nm3, as in the
#   FABER 2023 dataset; bed temperature = reported value; pressure = P_GAUGE_FABER.
P21 = pd.DataFrame({
    "ER":        [0.21, 0.24, 0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.34, 0.35],
    "feed_kg_h": [94.0, 95.5, 80.0, 84.4, 81.8, 69.9, 75.0, 64.5, 65.5, 57.8],
    "AF":        [1.10, 1.13, 1.29, 1.31, 1.35, 1.47, 1.44, 1.60, 1.70, 1.78],        # kg air / kg fuel
    "T_bed_C":   [764, 795, 806, 809, 830, 840, 853, 846, 887, 887],
    "Q_m3N_kg":  [1.51, 1.50, 1.79, 1.69, 1.76, 1.93, 1.81, 2.01, 2.10, 2.14],        # syngas, Nm3 per kg of fuel
    "LHV":       [5.6, 5.9, 5.8, 5.7, 5.5, 5.3, 5.3, 5.2, 5.0, 4.7],                  # MJ/Nm3
    "N2":  [49.2, 50.8, 48.8, 52.4, 51.9, 51.9, 54.4, 53.7, 54.3, 56.7],
    "CO2": [18.3, 16.8, 17.0, 17.3, 16.3, 15.9, 15.8, 15.5, 16.1, 15.1],
    "CO":  [12.5, 14.3, 15.0, 13.1, 14.7, 14.2, 13.9, 15.2, 14.1, 14.1],
    "H2":  [14.7, 12.3, 13.7, 11.4, 12.2, 13.3, 10.9, 10.9, 11.1, 10.1],
    "CH4": [3.90, 4.03, 4.20, 3.84, 3.54, 3.40, 3.54, 3.60, 3.16, 3.00],
    "C2":  [1.36, 1.71, 1.22, 1.70, 1.26, 1.15, 1.43, 0.94, 1.07, 0.92]})             # CnHm, n = 2-3
P21["air_Nm3_h"] = P21.AF * P21.feed_kg_h / 1.293
P21["Q"] = P21.Q_m3N_kg * P21.feed_kg_h
P21["blind"] = P21.ER != 0.25
P21_BLIND = P21.copy()   # kept for the article output (section 26): the name P21 is reused in section 21
_fs21 = faber_feed(0.0, 1.0)
_lev21 = {"predictive (formulation A0)": (KPS_PRED, OPT_PRED),
          "calibrated (global pair)": (kps_h2(CAL_GLOBAL["f_H2"]), replace(OPT_PRED, co_ox_bed_factor=CAL_GLOBAL["f_CO_bed"]))}
rows21, sum21 = [], []
for lev_, (kp_, op_) in _lev21.items():
    for _, r in P21.iterrows():
        s_ = simulate(_fs21, REACTOR_FABER, OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), kp_, op_)
        rec = dict(level=lev_, ER=r.ER, blind=bool(r.blind))
        for k in MAE_SPECIES + ["N2"]:
            rec[f"{k}_exp"], rec[f"{k}_mod"] = r[k], s_["y"][k]
        rec.update(C2_exp=r.C2, C2_mod=s_["C2"], LHV_exp=r.LHV, LHV_mod=s_["LHV"], Q_exp=r.Q, Q_mod=s_["Q_dry_Nm3_h"])
        rows21.append(rec)
TABLE_BLIND_POINTS = pd.DataFrame(rows21)
for lev_ in _lev21:
    d_ = TABLE_BLIND_POINTS[(TABLE_BLIND_POINTS.level == lev_) & TABLE_BLIND_POINTS.blind]
    m_ = {k: error_metrics(d_[f"{k}_exp"].values, d_[f"{k}_mod"].values) for k in MAE_SPECIES + ["N2", "LHV", "Q", "C2"]}
    rec = dict(level=lev_, n=len(d_), macro_MAPE=float(np.mean([m_[k]["MAPE"] for k in MAE_SPECIES])))
    rec.update({f"{k}_MAPE": m_[k]["MAPE"] for k in MAE_SPECIES})
    rec.update({f"{k}_bias": m_[k]["bias"] for k in MAE_SPECIES})
    rec.update(LHV_MAPE=m_["LHV"]["MAPE"], N2_MAPE=m_["N2"]["MAPE"], Q_MAPE=m_["Q"]["MAPE"], C2_MAPE=m_["C2"]["MAPE"],
               within20=float(np.mean([m_[k]["within20"] for k in MAE_SPECIES])))
    sum21.append(rec)
TABLE_BLIND = pd.DataFrame(sum21).set_index("level")
print("\n" + "=" * 78)
print("BLIND VALIDATION - Parrillo et al. (2021), eucalyptus chips, FABER reactor, 9 blind tests (ER = 0.25 excluded)")
print("=" * 78)
print(TABLE_BLIND.round(1).to_string())
print("  MAPE and within20 in %; bias = mean (model - exp) in pp. within20 = mean over H2, CO, CO2, CH4 of the share of tests within +/-20 %.")
TABLE_BLIND.to_csv(DATA_OUT / (FILE_TAG + "_blind_parrillo2021_summary.csv"))
TABLE_BLIND_POINTS.to_csv(DATA_OUT / (FILE_TAG + "_blind_parrillo2021_points.csv"), index=False)

# -----------------------------------------------------------------------------
# 14e. COMPARISON OF THE BED FORMULATIONS: two-zone bed (bubble + emulsion) without fitted parameters
# -----------------------------------------------------------------------------
# Question: does a physical description of the oxygen split in the bed (function bed_two_zone, section 9) do what the
#   calibrated factor f_CO_bed does, without fitting anything?
# Rules: (1) the result is reported whatever it is; (2) no gas-phase parameter is fitted: predictive kinetics (KPS_PRED;
#   dolomite activity eta of section 3 for the dolomite beds),
#   interchange from Kunii & Levenspiel, and the two limiting cases for the release of the volatiles; (3) formulation A
#   is not changed; (4) for formulation B the Parrillo 2021 data are not a blind test.
# TZ_STORED holds the stored run (mean MAE of H2, CO, CO2, CH4; RMSE and bias per species, in percentage points), used for
# Tables 6 and 7 of the article. RUN_TWO_ZONE = True repeats it (about 8 min).
RUN_TWO_ZONE = False
TZ_CASES = {"reference (mixed bed)": OPT_PRED,
            "two-zone, volatiles to emulsion": replace(OPT_PRED, bed_model="two_zone", tz_volatiles_to="emulsion"),
            "two-zone, volatiles to bubbles": replace(OPT_PRED, bed_model="two_zone", tz_volatiles_to="bubble")}
TZ_STORED = {
 "reference (mixed bed)": {
  "Terrados 2026": {
   "MAE": 1.2052442395512473,
   "RMSE": [
    2.4916805830620214,
    1.2836924098262976,
    1.1700388581766148,
    0.7041480686668159
   ],
   "bias": [
    1.5381683745465868,
    -0.5452711257883589,
    1.1291449951588046,
    -0.4470128780416126
   ]
  },
  "Calado 2023": {
   "MAE": 1.8749936938090785,
   "RMSE": [
    4.763338187910531,
    1.1499633407922516,
    0.7536309335464968,
    1.2167618851367445
   ],
   "bias": [
    -4.7590724041028745,
    -1.1425722862172112,
    -0.03182116019405493,
    0.9631882661661608
   ]
  },
  "Pio 2020": {
   "MAE": 4.054592394519232,
   "RMSE": [
    2.7818580381956206,
    7.829247245485713,
    5.882212355197279,
    0.9604566764118726
   ],
   "bias": [
    2.5243306682799367,
    -7.218132616506311,
    5.600025950986725,
    -0.875880342303956
   ]
  },
  "FABER (Parrillo 2023)": {
   "MAE": 1.406081206988034,
   "RMSE": [
    1.5946922027498946,
    2.7572193521086596,
    2.283123920379102,
    0.7606490086494377
   ],
   "bias": [
    -0.8656096659211927,
    -0.8312298710425939,
    1.84640779414449,
    -0.4566406665549573
   ]
  },
  "Parrillo 2021 (not blind here)": {
   "MAE": 4.316903812663298,
   "RMSE": [
    3.3972969866929024,
    6.79694856774885,
    6.136237245450134,
    1.3157981013527025
   ],
   "bias": [
    -3.126271194056329,
    -6.722947444574104,
    6.118115571562311,
    -1.3002810404604463
   ]
  }
 },
 "two-zone, volatiles to emulsion": {
  "Terrados 2026": {
   "MAE": 1.1219517253342055,
   "RMSE": [
    3.37657493860778,
    1.058234365703072,
    0.5264846546573334,
    0.7710046060178456
   ],
   "bias": [
    2.5054256922720124,
    0.6539400615068075,
    0.3071318936568478,
    -0.5507560278193573
   ]
  },
  "Calado 2023": {
   "MAE": 1.0394360428817029,
   "RMSE": [
    0.22522064808750944,
    2.9941709743059537,
    0.3695247083144217,
    0.9841052989190369
   ],
   "bias": [
    0.16744295760336966,
    -2.944588167327558,
    0.3078173030454767,
    0.7253400693190931
   ]
  },
  "Pio 2020": {
   "MAE": 3.617980088287731,
   "RMSE": [
    4.027703854382562,
    6.143068523726197,
    4.617620944479284,
    1.0774967899012473
   ],
   "bias": [
    3.7081878659029486,
    -5.350525355158252,
    4.355379977474714,
    -0.9828079493191378
   ]
  },
  "FABER (Parrillo 2023)": {
   "MAE": 1.2456563960784535,
   "RMSE": [
    1.5169115330126366,
    2.5221388161876974,
    1.3951378575003774,
    0.9315842085794003
   ],
   "bias": [
    0.9409260916664319,
    0.7679357564445163,
    0.6108042913055437,
    -0.7416742826737269
   ]
  },
  "Parrillo 2021 (not blind here)": {
   "MAE": 3.2118427027995358,
   "RMSE": [
    1.7243506060191298,
    5.092210953999801,
    4.825434326709274,
    1.5069926398216276
   ],
   "bias": [
    -1.1542693550399579,
    -5.0492204930409,
    4.812185074409514,
    -1.4930409742385646
   ]
  }
 },
 "two-zone, volatiles to bubbles": {
  "Terrados 2026": {
   "MAE": 2.107248054154714,
   "RMSE": [
    2.5286174265317336,
    3.4757074940728954,
    2.566045248792965,
    0.5559505997227678
   ],
   "bias": [
    -1.223544946795684,
    -3.256296420081051,
    2.513923700194378,
    -0.31198169950786375
   ]
  },
  "Calado 2023": {
   "MAE": 3.304212333235761,
   "RMSE": [
    4.185146687869248,
    6.468969978007641,
    1.5174445751460437,
    1.3633909625123193
   ],
   "bias": [
    -4.180117820120581,
    -6.458967088324037,
    1.385224572782817,
    1.1925398517156098
   ]
  },
  "Pio 2020": {
   "MAE": 3.7137387175600476,
   "RMSE": [
    3.645532064128597,
    6.598171894092342,
    4.898367565725842,
    1.0332216729130765
   ],
   "bias": [
    3.341857622035564,
    -5.893570521946716,
    4.674348724412684,
    -0.9451780018452229
   ]
  },
  "FABER (Parrillo 2023)": {
   "MAE": 1.4242038848519152,
   "RMSE": [
    1.8440044243460032,
    3.207305381414413,
    1.7495389022695105,
    0.9663012140045291
   ],
   "bias": [
    0.4759212259303025,
    0.13562106066011217,
    0.7159728298870068,
    -0.7732885966725752
   ]
  },
  "Parrillo 2021 (not blind here)": {
   "MAE": 4.83726982034132,
   "RMSE": [
    3.7987375409791126,
    8.022424074931536,
    6.060173899788042,
    1.6485492360397425
   ],
   "bias": [
    -3.667296115299492,
    -7.998012280425611,
    6.045005075791485,
    -1.638765809848693
   ]
  }
 }
}

def two_zone_run():
    sets_ = {ds: pts for ds, (_, pts) in CAL_PTS.items()}
    sets_["Parrillo 2021 (not blind here)"] = [(_fs21, REACTOR_FABER, OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h),
                                                {k: r[k] for k in MAE_SPECIES}) for _, r in P21[P21.blind].iterrows()]
    out_ = {}
    for case_, o_ in TZ_CASES.items():
        out_[case_] = {}
        for ds_, pts_ in sets_.items():
            D_ = np.array([[simulate(fs, rc, op, KPS_PRED, o_)["y"][k] - ex[k] for k in MAE_SPECIES] for (fs, rc, op, ex) in pts_])
            out_[case_][ds_] = dict(MAE=float(np.abs(D_).mean()), RMSE=np.sqrt((D_ ** 2).mean(axis=0)).tolist(), bias=D_.mean(axis=0).tolist())
    return out_
TZ = two_zone_run() if RUN_TWO_ZONE else TZ_STORED
_iCO = MAE_SPECIES.index("CO")
TABLE_TWO_ZONE = pd.DataFrame({(c_[:31], "MAE"): {d_: v_["MAE"] for d_, v_ in r_.items()} for c_, r_ in TZ.items()})
for c_, r_ in TZ.items():
    TABLE_TWO_ZONE[(c_[:31], "CO bias")] = pd.Series({d_: v_["bias"][_iCO] for d_, v_ in r_.items()})
    TABLE_TWO_ZONE[(c_[:31], "max RMSE")] = pd.Series({d_: max(v_["RMSE"]) for d_, v_ in r_.items()})
TABLE_TWO_ZONE = TABLE_TWO_ZONE[sorted(TABLE_TWO_ZONE.columns, key=lambda c: (list(TZ).index(next(k for k in TZ if k[:31] == c[0])), ["MAE", "CO bias", "max RMSE"].index(c[1])))]
print("\n" + "=" * 78)
print("COMPARISON OF BED FORMULATIONS - two-zone bed, no fitted parameter (percentage points; predictive kinetics)" + ("" if RUN_TWO_ZONE else "  [stored run]"))
print("=" * 78)
print(TABLE_TWO_ZONE.round(2).to_string())
print("  MAE = mean over H2, CO, CO2, CH4. Calibrated level for comparison (mean MAE): Terrados 1.02, Calado 2.10, Pio 3.17, FABER 1.28, Parrillo 2021 3.24.")
print("  Reading: with the volatiles released in the emulsion the two-zone bed is better than the mixed bed in every dataset and close to")
print("  the calibrated level; with the volatiles released in the bubbles it is worse. The CO deficit with woody fuels remains (about -5 pp).")
TABLE_TWO_ZONE.to_csv(DATA_OUT / (FILE_TAG + "_two_zone_exploratory.csv"))

# -----------------------------------------------------------------------------
# 14f. BLIND TEST OF THE TWO-ZONE BED (formulation B) ON TWO FURTHER PILOT PLANTS
# -----------------------------------------------------------------------------
# Sources: Campoy, Gomez-Barea, Ollero, Nilsson & Fuentes-Cano (2011), Proc. 19th European Biomass Conference, 1370-1375,
#            Tables I-II: exhausted olive pomace (orujillo) and wood pellets, 100 kWth BFB, ofite bed, air.
#          Arena & Di Gregorio (2014), Fuel 117, 528-536, Tables 1-3: solid recovered fuel (SRF), 0.381 m BFB, olivine bed, air.
# These data played no part in the development of the model. They were run ONCE with the two-zone bed frozen;
# the protocol and the assumptions were written before the run and nothing was changed afterwards.
# ASSUMPTIONS, flagged (the articles do not report these inputs):
#   - orujillo treated as the olive pomace of the model; SRF as its RDF; wood pellets as its pine pellets;
#   - HHV obtained by inverting the fuel-LHV formula of the model so that the LHV equals the reported one;
#   - SRF of Arena: volatile matter and fixed carbon in the proportion of the Terrados RDF (no proximate analysis given);
#   - Campoy rig: bed inventory 12 kg (not reported); equivalent height 1.40 + 2.15 (0.25/0.15)^2 m, which keeps the
#     freeboard volume (the model has a single diameter);
#   - olivine of Arena 300 um (midpoint of 200-400 um); eps_mf = 0.45; orifices 20 and 50; atmospheric pressure;
#   - char residence time 3600 s; no catalytic tar conversion (ofite and olivine treated as inert);
#   - air of Arena = (air/fuel ratio) x feed / 1.293 kg/Nm3; CO of test 2 of Arena read as 10.58 % (printed "1058").
def _hhv_from_lhv(lhv, H, M, O):
    return lhv + 0.212 * H + 0.0245 * M + LHV_O_COEF * O
def _comp_b2(name, C, H, N, S, O, M, VM, FC, ash_ar, lhv, sur):
    return FeedstockComponent(name, {"C": C, "H": H, "N": N, "S": S, "O": O, "Ash": ash_ar / (1 - M / 100.0)}, M, VM, FC, ash_ar,
                              _hhv_from_lhv(lhv, H, M, O), "blind test 2", sur)
_ORUJILLO = _comp_b2("olive_pomace", 46.4, 5.3, 1.8, 0.2, 32.1, 8.7, 59.4, 19.0, 13.0, 15.8, SUR_EUCALYPTUS)
_WOODPEL = _comp_b2("pine_pellets", 49.5, 5.8, 2.0, 0.1, 41.9, 6.3, 75.9, 17.2, 0.7, 17.1, SUR_EUCALYPTUS)
_m_srf = 6.4; _d_srf = 1 - _m_srf / 100
_SRF_ARENA = _comp_b2("RDF", 43.3 / _d_srf, 6.25 / _d_srf, 0.68 / _d_srf, 0.2 / _d_srf, 23.55 / _d_srf, _m_srf, 65.54, 8.61, 19.45, 19.95, SUR_RDF_TERRADOS)
REACTOR_CAMPOY = ReactorConfig(0.150, 1.40 + 2.15 * (0.25 / 0.15) ** 2, 12.0, "ofite", 750e-6, 2620.0, 0.45, 20, 3600.0, "Campoy_US", tar_catalyst=None)
REACTOR_ARENA = ReactorConfig(0.381, 4.64, 146.0, "olivine", 300e-6, 2900.0, 0.45, 50, 3600.0, "Arena_BFB", tar_catalyst=None)
_fs1 = lambda c_, k_: FeedstockConfig({k_: c_}, {k_: 1.0}, k_)
BLIND2 = {
    "Campoy 2011, orujillo (blind)": (_fs1(_ORUJILLO, "olive_pomace"), REACTOR_CAMPOY, [
        dict(feed=15.0, air=14.0, T=770, CO=10.8, H2=12.7, CO2=15.2, CH4=6.7, LHV=5.1), dict(feed=12.9, air=17.0, T=830, CO=9.7, H2=11.0, CO2=16.8, CH4=5.4, LHV=4.4),
        dict(feed=10.2, air=14.0, T=835, CO=9.3, H2=10.3, CO2=17.5, CH4=4.9, LHV=4.0), dict(feed=8.0, air=14.0, T=870, CO=7.5, H2=8.7, CO2=17.9, CH4=4.0, LHV=3.3)]),
    "Arena 2014, SRF (blind)": (_fs1(_SRF_ARENA, "RDF"), REACTOR_ARENA, [
        dict(feed=f_, air=af_ * f_ / 1.293, T=T_, CO2=co2_, CO=co_, H2=h2_, CH4=ch4_, LHV=lhv_ / 1000.0)
        for f_, af_, T_, co2_, co_, h2_, ch4_, lhv_ in [(61.4, 1.47, 849, 13.36, 9.74, 8.48, 4.18, 5390), (58.6, 1.54, 852, 15.69, 8.98, 7.11, 4.20, 5652),
            (57.65, 1.56, 869, 12.28, 10.47, 8.44, 4.57, 5550), (51.9, 1.73, 879, 12.83, 10.40, 8.24, 4.35, 5160),
            (49.4, 1.82, 898, 14.04, 12.73, 7.08, 3.33, 4910), (47.4, 1.90, 932, 11.66, 10.58, 7.68, 3.78, 4690)]]),
    "Campoy 2011, wood pellets (not part of the blind test)": (_fs1(_WOODPEL, "pine_pellets"), REACTOR_CAMPOY, [
        dict(feed=17.9, air=18.0, T=785, CO=17.8, H2=13.0, CO2=14.4, CH4=5.9, LHV=5.8), dict(feed=14.1, air=18.0, T=808, CO=17.5, H2=10.9, CO2=14.8, CH4=5.1, LHV=5.2),
        dict(feed=11.9, air=18.0, T=813, CO=15.9, H2=8.5, CO2=15.0, CH4=4.9, LHV=4.7)]),
}
_lev_b2 = {"B, two-zone": (KPS_PRED, replace(OPT_PRED, bed_model="two_zone", tz_volatiles_to="emulsion")),
           "A, calibrated": (kps_h2(CAL_GLOBAL["f_H2"]), replace(OPT_PRED, co_ox_bed_factor=CAL_GLOBAL["f_CO_bed"]))}
rows_b2, sum_b2 = [], []
for lev_, (kp_, op_) in _lev_b2.items():
    for ds_, (fs_, rc_, pts_) in BLIND2.items():
        for p_ in pts_:
            s_ = simulate(fs_, rc_, OperatingPoint(float(p_["T"]), 0.0, p_["feed"], p_["air"]), kp_, op_)
            rec = dict(formulation=lev_, dataset=ds_, T_C=p_["T"], ER=s_["ER"], LHV_exp=p_["LHV"], LHV_mod=s_["LHV"])
            for k in MAE_SPECIES:
                rec[f"{k}_exp"], rec[f"{k}_mod"] = p_[k], s_["y"][k]
            rows_b2.append(rec)
TABLE_BLIND2_POINTS = pd.DataFrame(rows_b2)
for (lev_, ds_), d_ in TABLE_BLIND2_POINTS.groupby(["formulation", "dataset"], sort=False):
    m_ = {k: error_metrics(d_[f"{k}_exp"].values, d_[f"{k}_mod"].values) for k in MAE_SPECIES}
    rec = dict(formulation=lev_, dataset=ds_, n=len(d_), MAE_mean=float(np.mean([m_[k]["MAE"] for k in MAE_SPECIES])),
               RMSE_max=float(max(m_[k]["RMSE"] for k in MAE_SPECIES)), MAPE_mean=float(np.mean([m_[k]["MAPE"] for k in MAE_SPECIES])))
    for k in MAE_SPECIES:
        rec.update({f"{k}_MAE": m_[k]["MAE"], f"{k}_RMSE": m_[k]["RMSE"], f"{k}_bias": m_[k]["bias"], f"{k}_MAPE": m_[k]["MAPE"]})
    sum_b2.append(rec)
TABLE_BLIND2 = pd.DataFrame(sum_b2)
print("\n" + "=" * 78)
print("BLIND TEST OF THE TWO-ZONE BED - orujillo (Campoy et al. 2011) and SRF (Arena & Di Gregorio 2014)")
print("=" * 78)
print(TABLE_BLIND2[["formulation", "dataset", "n", "MAE_mean", "RMSE_max", "MAPE_mean"] + [f"{k}_RMSE" for k in MAE_SPECIES] + [f"{k}_bias" for k in MAE_SPECIES]].round(2).to_string(index=False))
print("  MAE, RMSE and bias in percentage points; MAPE in %. Run once with the model frozen; the wood-pellet tests are not part of the blind test.")
TABLE_BLIND2.to_csv(DATA_OUT / (FILE_TAG + "_blind2_summary.csv"), index=False)
TABLE_BLIND2_POINTS.to_csv(DATA_OUT / (FILE_TAG + "_blind2_points.csv"), index=False)

# -----------------------------------------------------------------------------
# 14d. LINK TO THE TECHNO-ECONOMIC, LCA AND OPTIMISATION SECTIONS (15-24)
# -----------------------------------------------------------------------------
# Sections 15-24 use KPS and OPT_H1, which point to the level selected with LEVEL (section 10).
#   - Monte Carlo of the model uncertainty: with the two-zone model (LEVEL = "two_zone") it varies the bubble-emulsion
#     interchange coefficient (x 0.5-2, log-uniform) and the share of the volatiles released in the emulsion (0.5-1);
#     with the calibrated mixed-bed model, the two calibrated factors (f_H2 in [0.1, 1], f_CO_bed in [0.03, 0.3],
#     log-uniform). The char residence time (x 0.7-1.3) and the dolomite activity (x 0.5-2) are varied in all cases.
#     See mc_factors below.
#   - Freeboard profile (23c): starts from the bed outlet and uses the kinetic WGS.
# VALIDITY: the model is validated for olive pomace/RDF in the Portalegre reactor at ER 0.26-0.29, 30 % RDF and
#   700-1100 C, and for plastic-rich blends in FABER. The maps and the optimum explore ER 0.20-0.40 and 10-50 % RDF: outside
#   the validated window they are EXTRAPOLATIONS. Temperature and ER are treated as independent inputs.
_MC_LAST = {}
NASA_EB = {"N2": [1000.0, [2.92663788, 0.001487977, -5.685e-07, 1e-10, -0.0, -922.795384, 5.98054018], [3.298677, 0.0014082399, -3.9632e-06, 5.6e-09, -0.0, -1020.9, 3.950372]], "O2": [1000.0, [3.69757685, 0.0006135197, -1.259e-07, 0.0, -0.0, -1233.92966, 3.18917125], [3.212936, 0.0011274861, -5.756e-07, 1.3e-09, -0.0, -1005.249, 6.034739]], "H2": [1000.0, [2.9914222, 0.0007000644, -5.63e-08, -0.0, 0.0, -835.033546, -1.35510641], [3.298124, 0.0008249441, -8.143e-07, -1e-10, 0.0, -1012.521, -3.294094]], "H2O": [1000.0, [2.67214569, 0.0030562929, -8.73e-07, 1e-10, -0.0, -29899.2115, 6.86281125], [3.386842, 0.003474982, -6.3547e-06, 7e-09, -0.0, -30208.11, 2.590232]], "CO": [1000.0, [3.02507617, 0.001442689, -5.631e-07, 1e-10, -0.0, -14268.3499, 6.10822521], [3.262451, 0.001511941, -3.8818e-06, 5.6e-09, -0.0, -14310.54, 4.848897]], "CO2": [1000.0, [4.45362582, 0.003140168, -1.2784e-06, 2e-10, -0.0, -48966.9524, -0.955420007], [2.275724, 0.0099220723, -1.04091e-05, 6.9e-09, -0.0, -48373.14, 10.18849]], "CH4": [1000.0, [1.68346564, 0.01023724, -3.8751e-06, 7e-10, -0.0, -10080.7773, 9.62347575], [0.7787417, 0.01747668, -2.78341e-05, 3.05e-08, -0.0, -9825.228, 13.72219]], "C2H4": [1000.0, [3.52841648, 0.01148518, -4.4184e-06, 8e-10, -0.0, 4428.2903, 2.23039249], [-0.8614879, 0.027961619, -3.38868e-05, 2.79e-08, -0.0, 5573.047, 24.21148]], "C3H6": [1000.0, [6.73231663, 0.01490834, -4.9499e-06, 7e-10, -0.0, -923.623057, -13.3137684], [1.493307, 0.02092517, 4.4868e-06, -1.67e-08, 0.0, 1074.826, 16.14534]], "C6H6": [1000.0, [12.9109067, 0.01723296, -5.0242e-06, 6e-10, -0.0, 3664.37279, -50.0281184], [-3.138012, 0.04723103, -2.9622e-06, -3.26e-08, 0.0, 8890.03, 36.57573]], "C6H5OH": [1000.0, [14.1552427, 0.019935034, -7.1822e-06, 1.2e-09, -0.0, -18128.7441, -51.7984911], [-0.290978575, 0.0408562397, 2.42829e-05, -7.14e-08, 0.0, -13412.978, 26.8745637]], "C7H8": [1000.0, [9.889287, 0.03123517, -1.10466e-05, 1.3e-09, 0.0, 590.756417, -29.4263546], [-4.695182, 0.069510803, -4.38445e-05, 1.04e-08, 0.0, 4693.355, 46.40732]], "C10H8": [1401.0, [23.4023214, 0.0242434427, -8.3628e-06, 1.3e-09, -0.0, 6519.41329, -107.433208], [-8.83645988, 0.109300567, -9.55201e-05, 4.22e-08, -0.0, 16653.3366, 62.1064766]]}
NASA_EB["Tar"] = NASA_EB["C10H8"]; NASA_EB["C6H6O"] = NASA_EB["C6H5OH"]
def h_molar(sp, T):
    """Molar enthalpy (J/mol) incl. formation, NASA-7."""
    tmid, hi, lo = NASA_EB[sp]
    a = hi if T > tmid else lo
    return R_GAS * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)
CP_CHAR, CP_ASH, T_REF = 1400.0, 1000.0, 298.15

# Energy-consistent plant sizing.
#   Equation: first law for the bed at the imposed temperature (Eqs. S20-S21):
#       H_fuel + H_air = H_gas(T) + H_solid(T) + Q.  If Q < 0 the bed needs heat to stay at T.
#   That heat is taken from the product gas, so the power sent to the boiler is P_gas - Q_ext and more fuel is needed
#   to deliver the fixed heat demand.
#   Units: W (bed_surplus_W), kW (bed_heat_deficit_kW), both at the feed rate of the simulation.
#   ASSUMPTIONS, flagged: no wall losses (UA = 0), air at 25 C, heat transferred to the bed without loss, no recovery of
#   the sensible heat of the gas. Validity: steady state. Thermodynamic data: NASA-7 polynomials.
#   Reason: temperature and ER are independent inputs of the model; without this term a hot bed at low ER is credited
#   with an efficiency that the energy balance does not allow.
ENERGY_CONSISTENT = True
def bed_surplus_W(sim, T_air_C=25.0):
    """(H_in - H_out) of the bed in W at the feed rate of the simulation; negative = heat must be supplied."""
    op = sim["op"]; fe = sim["fe"]; T = op.T_C + 273.15; m_fuel = op.feed_kg_h / 3600.0
    LHV_ar = (fe["HHV_blend"] - 0.212 * fe["H_pct"] - 0.0245 * fe["M_pct"] - LHV_O_COEF * fe["O_pct"]) * 1e6   # J/kg
    nC, nH, nN = fe["el"]["C"], fe["el"]["H"] + 2 * fe["moist_kg_s"] * 1000 / 18.015, fe["el"]["N"]
    H_fuel = nC * h_molar("CO2", T_REF) + nH / 2 * h_molar("H2O", T_REF) + nN / 2 * h_molar("N2", T_REF) + LHV_ar * m_fuel
    air = op.air_Nm3_h * MOL_PER_NM3 / 3600.0; Ta = T_air_C + 273.15
    H_air = air * (AIR_O2 * h_molar("O2", Ta) + (1 - AIR_O2) * h_molar("N2", Ta))
    Fb = sim["bed"]["F_out"]
    H_gas = sum(Fb[IDX[sp_]] * h_molar(sp_, T) for sp_ in SPECIES if sp_ in NASA_EB and Fb[IDX[sp_]] > 0)
    char_kg_s = (sim["bed"]["F_char_out"] + Fb[I_CS]) * AW["C"] / 1000.0
    return (H_fuel + H_air) - (H_gas + (char_kg_s * CP_CHAR + fe["ash_kg_s"] * CP_ASH) * (T - T_REF))
def bed_heat_deficit_kW(sim):
    """Heat (kW) that the bed needs to stay at the imposed temperature; 0 if the balance has a surplus."""
    return max(0.0, -bed_surplus_W(sim)) / 1e3 if ENERGY_CONSISTENT else 0.0

def mc_factors(rng_):
    """Draw of the uncertain model quantities for the model-uncertainty Monte Carlo (two numbers, see mc_opt / mc_kps).
       two_zone  : multiplier of the bubble-emulsion interchange coefficient, log-uniform in [0.5, 2] (uncertainty of the
                   Kunii-Levenspiel correlation, assumed), and share of the volatiles released in the emulsion, uniform in
                   [0.5, 1] (assumed: the validation favours the emulsion, the bubble limit is rejected by it).
       calibrated: the two calibrated factors over the range in which the calibration cannot tell them apart.
       predictive: nothing to draw."""
    if LEVEL == "two_zone":
        a_, b_ = 10 ** rng_.uniform(np.log10(0.5), np.log10(2.0)), rng_.uniform(0.5, 1.0)
    elif LEVEL == "calibrated":
        a_, b_ = 10 ** rng_.uniform(-1.0, 0.0), 10 ** rng_.uniform(-1.5, -0.5)
    else:
        a_, b_ = 1.0e-3, 1.0
    _MC_LAST["a"], _MC_LAST["b"] = a_, b_
    return a_, b_
def mc_kps(f_H2, fdol):
    k_ = KPS_PRED if LEVEL == "two_zone" else kps_h2(f_H2)
    return replace(k_, A=np.where(_IS_DOL, k_.A * fdol, k_.A))
def mc_opt(f_CO_bed):
    if LEVEL == "two_zone":
        return replace(OPT_REF, tz_kbe_factor=_MC_LAST["a"], tz_vol_emulsion_frac=_MC_LAST["b"])
    return replace(OPT_REF, co_ox_bed_factor=f_CO_bed)

# =============================================================================
# 15. TECHNO-ECONOMIC (NPV, IRR, discounted payback) AND LIFE-CYCLE (GWP100) MODULE
#   Driven by the kinetic model outputs (syngas flow, LHV, carbon distribution).
#   Case study and inventory follow Terrados-Cristos et al. (2026), Tables 5-6:
#   thermal-energy supply to a glass plant, 60 kg/h OP70/RDF30, 7200 h/y, FU = 1 MJ useful heat.
#   Every parameter carries its source; values not reported by Terrados are ASSUMED and
#   are the first candidates for literature sourcing and sensitivity analysis.
# =============================================================================
from scipy.optimize import brentq as _brentq
TEA = {  # value, unit, source
    "P_design_kWth":    (209.8,  "kWth",        "Terrados 2026, Table 5 (fixed thermal demand; feed rate adjusts)"),
    "hours":            (7200.0, "h/y",         "Terrados 2026, Table 5"),
    "capex_spec":       (1087.0, "EUR/kWth",    "Terrados 2026, Table 5 (190 kEUR)"),
    "om_frac":          (0.05,   "-/y of CAPEX","Porcu et al. 2019, Energies 12:494 (Terrados: 0.20)"),
    "pomace_cost":      (30.0,   "EUR/t",       "Terrados 2026, Table 5 [Polonio]"),
    "rdf_avoided_cost": (35.0,   "EUR/t",       "Portuguese landfill tax TGR 2025 (Despacho 15554-A/2025); lower bound"),
    "ng_price":         (48.1,   "EUR/MWh",     "Eurostat, PT non-household natural gas, H2 2025"),
    "ets_price":        (75.0,   "EUR/t CO2",   "EU ETS 2025 (Enerdata; range 60-80)"),
    "eta_NG":           (0.90,   "-",           "ASSUMED (natural-gas boiler)"),
    "EF_NG_direct":     (0.202,  "t CO2/MWh",   "IPCC default 56.1 kg CO2/GJ natural gas"),
    "boiler_eff":       (0.80,   "-",           "Terrados 2026, Table 6 (thermal efficiency)"),
    "elec_kWh_per_MJ":  (0.007,  "kWh/MJ",      "Terrados 2026, Table 6 (pretreatment)"),
    "elec_price":       (133.0,  "EUR/MWh",     "Eurostat nrg_pc_205, PT non-household, H1 2024"),
    "discount_rate":    (0.08,   "-",           "Porcu et al. 2019"),
    "lifetime":         (15,     "y",           "Porcu et al. 2019"),
}
LCA = {
    "rdf_fossil_C_frac": (0.78,   "-",              "Derived from Terrados Table 2 (range 0.45-0.85; Mohn et al. 2012)"),
    "grid_EF":           (0.166,  "kgCO2eq/kWh",    "PT power sector 2023, life-cycle GWP100 (Ember/Statista)"),
    "CH4_slip":          (1.0e-4, "kg CH4/MJ",      "Terrados 2026, Table 6"),
    "GWP_CH4":           (29.8,   "kgCO2eq/kg",     "IPCC AR6 (fossil CH4, GWP100)"),
    "GWP_CH4_biogenic":  (27.0,   "kgCO2eq/kg",     "IPCC AR6 (non-fossil CH4, GWP100; landfill gas of the biogenic DOC)"),
    "NG_reference":      (0.070,  "kgCO2eq/MJ",     "Terrados 2026, Table 6 (natural-gas boiler)"),
}
P = {k: v[0] for k, v in {**TEA, **LCA}.items()}

def npv_irr_pb(capex, cf, r, n):
    flows = [-capex] + [cf] * int(n)
    npv = sum(f / (1 + r) ** t for t, f in enumerate(flows))
    try:
        irr = _brentq(lambda x: sum(f / (1 + x) ** t for t, f in enumerate(flows)), -0.99, 1.0)
    except ValueError:
        irr = np.nan
    cum, pb = -capex, np.nan
    for t in range(1, int(n) + 1):
        cum += cf / (1 + r) ** t
        if cum >= 0:
            pb = t; break
    return npv, irr, pb

def tea_lca(sim, fs, p=P):
    """sim: output of simulate() at the model feed rate; results scaled to p['feed_kg_h']."""
    feed_model = sim["fe"]["feed_kg_s"] * 3600.0
    P_th_model = sim["Q_dry_Nm3_h"] * sim["LHV"] / 3.6               # kWth in syngas (Terrados Eq. 6)
    P_th = p["P_design_kWth"]                                        # fixed thermal demand
    q_ext_model = bed_heat_deficit_kW(sim)                           # kW that the bed needs at the imposed temperature
    scale = P_th / max(P_th_model - q_ext_model, 1e-9)               # the gas left after heating the bed meets the demand
    feed_needed = feed_model * scale
    Q = sim["Q_dry_Nm3_h"] * scale
    heat_MWh = P_th * p["boiler_eff"] * p["hours"] / 1000.0          # useful heat
    # six-tenths scaling from the reference plant (Towler & Sinnott); exponent 1.0 = linear
    capex = p["capex_spec"] * p.get("P_ref_kWth", P_th) * (P_th / p.get("P_ref_kWth", P_th)) ** p.get("capex_exp", 1.0)
    w = fs.mass_fractions
    pomace_t = feed_needed * w.get("olive_pomace", 0.0) * p["hours"] / 1000.0
    rdf_t = feed_needed * w.get("RDF", 0.0) * p["hours"] / 1000.0
    MJ_useful = heat_MWh * 3600.0
    elec_MWh = p["elec_kWh_per_MJ"] * MJ_useful / 1000.0
    # value of the substituted heat = natural-gas cost + ETS cost of its direct CO2 (per MWh of heat)
    heat_price = p["ng_price"] / p["eta_NG"] + p["ets_price"] * p["EF_NG_direct"] / p["eta_NG"]
    # GWP100 per MJ useful heat: fossil CO2 from burning the syngas carbon + CH4 slip + electricity
    F = sim["F_out"]
    C_gas = sum(F[IDX[s]] * ATOM[IDX[s], 0] for s in PERMANENT_GAS) * scale      # mol C/s in dry gas
    comp = fs.components
    c_tot = c_foss = 0.0
    for k, wk in w.items():
        c = comp[k]
        tot = sum(c.ultimate_dry[e] for e in ["C", "H", "N", "S", "O", "Ash"])
        cm = wk * (1 - c.moisture_ar / 100) * c.ultimate_dry["C"] / tot
        c_tot += cm
        c_foss += cm * (p["rdf_fossil_C_frac"] if k == "RDF" else 0.0)
    f_fossil = c_foss / c_tot                                                     # fossil share of feed carbon
    co2_fossil = C_gas * f_fossil * 44.01e-3 * 3600 * p["hours"]                 # kg/y (syngas carbon -> CO2)
    gwp = (co2_fossil / MJ_useful + p["CH4_slip"] * p["GWP_CH4"]
           + p["elec_kWh_per_MJ"] * p["grid_EF"])
    revenue = heat_MWh * heat_price + rdf_t * p["rdf_avoided_cost"]
    opex = (p["om_frac"] * capex + pomace_t * p["pomace_cost"] + elec_MWh * p["elec_price"]
            + co2_fossil / 1000.0 * p["ets_price"])                      # ETS on the fossil CO2 of the syngas
    npv, irr, pb = npv_irr_pb(capex, revenue - opex, p["discount_rate"], p["lifetime"])
    return dict(heat_price=heat_price, feed_kg_h=feed_needed, Q_ext_kW=q_ext_model * scale,
                eff_net_pct=sim["CGE"] * (P_th_model - q_ext_model) / max(P_th_model, 1e-9), P_th_kW=P_th, heat_MWh_y=heat_MWh, CAPEX_kEUR=capex / 1e3, rev_kEUR_y=revenue / 1e3,
                opex_kEUR_y=opex / 1e3, NPV_kEUR=npv / 1e3, IRR_pct=100 * irr if irr == irr else np.nan,
                PB_y=pb, GWP_kg_MJ=gwp, GWP_vs_NG_pct=100 * (gwp / p["NG_reference"] - 1),
                fossil_C_share=f_fossil)

print("\n" + "=" * 78)
print("15. TECHNO-ECONOMIC (NPV/IRR/PB) AND LIFE-CYCLE (GWP100) MODULE")
print("=" * 78)
print("  Parameters (value, unit, source):")
for k, (v, u, src) in {**TEA, **LCA}.items():
    print(f"    {k:18s} {v:>10} {u:14s} {src}")

# ---- 15a. consistency check of the published Terrados economic case (Table 5) ----
rev_T, om_T, pom_T = 49.9e3, 38.0e3, 60 * 0.7 * 7200 / 1000 * 30.0
print("\n  15a. Published Terrados case (Table 5): revenue 49.9 kEUR/y, O&M 38 kEUR/y, pomace "
      f"{pom_T/1e3:.1f} kEUR/y -> net {(rev_T - om_T - pom_T)/1e3:.1f} kEUR/y against CAPEX 190 kEUR.")
print(f"      With any discount rate >= 0 and 20-25 y, NPV <= {(rev_T - om_T - pom_T) * 25 / 1e3 - 190:.0f} kEUR: "
      "the published NPV (407 kEUR), IRR (9 %) and payback (14.4 y) are NOT reproducible from Table 5 "
      "alone (financing, inflation escalation and avoided-disposal credits are not fully reported).")

# ---- 15b. base case: model at 700 C (Terrados design choice), air from the nitrogen balance ----
r700 = TABLE4.loc[700]
sim_base = simulate(BLEND_TERRADOS, REACTOR_TERRADOS,
                    OperatingPoint(700, r700.P_gauge_kPa, r700.feed_kg_h, air_n2_consistent(r700, BLEND_TERRADOS)),
                    KPS, OPT_H1)
base = tea_lca(sim_base, BLEND_TERRADOS)
print(f"\n  15b. Base case (model {MODEL_NAME}, 700 C, Terrados 2026):")
for k, v in base.items():
    print(f"      {k:14s} = {v:,.3f}")

# ---- 15c. parametric T x ER map (technical + economic + environmental) ----
def stoich_air_Nm3_per_kg(fs):
    fe = feed_elements(fs, 1.0)
    o2 = fe["el"]["C"] + fe["el"]["H"] / 4 + fe["el"]["S"] - fe["el"]["O"] / 2          # mol/s per kg/h
    return o2 / AIR_O2 * 3600 / MOL_PER_NM3
AIR_ST = stoich_air_Nm3_per_kg(BLEND_TERRADOS)
T_GRID = [700, 800, 900, 1000, 1100]
ER_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
rows15 = []
for T in T_GRID:
    for er in ER_GRID:
        op = OperatingPoint(T, -30.0, 50.0, er * AIR_ST * 50.0)
        sim = simulate(BLEND_TERRADOS, REACTOR_TERRADOS, op, KPS, OPT_H1)
        rows15.append(dict(T=T, ER=er, H2=sim["y"]["H2"], CO=sim["y"]["CO"], CH4=sim["y"]["CH4"],
                           CO2=sim["y"]["CO2"], N2=sim["y"]["N2"], gas_Nm3_kg=sim["Q_dry_Nm3_h"] / 50.0,   # N2 and gas yield (Fig. 6)
                           LHV=sim["LHV"], CGE=sim["CGE"], tar_g_kg=sim["tar_g_kg"],
                           **tea_lca(sim, BLEND_TERRADOS)))
MAP15 = pd.DataFrame(rows15)
print(f"\n  15c. T x ER map (stoichiometric air {AIR_ST:.2f} Nm3/kg; ER on the elemental basis):")
for col in ["H2", "CO", "LHV", "CGE", "tar_g_kg", "feed_kg_h", "NPV_kEUR", "IRR_pct", "GWP_kg_MJ"]:
    print(f"\n    {col}:")
    print(MAP15.pivot(index="T", columns="ER", values=col).round(2).to_string())
MAP15.to_csv(DATA_OUT / (FILE_TAG + "_T_ER_tea_lca_map.csv"), index=False)

# ---- 15d. one-at-a-time sensitivity of NPV and GWP (base case, +/-30 %) ----
sens = []
for k in ["capex_spec", "om_frac", "ng_price", "ets_price", "discount_rate", "lifetime", "pomace_cost",
          "boiler_eff", "elec_price", "rdf_fossil_C_frac", "grid_EF"]:
    lo, hi = dict(P), dict(P)
    lo[k] = P[k] * 0.7; hi[k] = P[k] * 1.3
    if k == "lifetime": lo[k], hi[k] = 10, 20
    a, b = tea_lca(sim_base, BLEND_TERRADOS, lo), tea_lca(sim_base, BLEND_TERRADOS, hi)
    sens.append(dict(param=k, NPV_lo=a["NPV_kEUR"], NPV_hi=b["NPV_kEUR"], GWP_lo=a["GWP_kg_MJ"], GWP_hi=b["GWP_kg_MJ"]))
SENS15 = pd.DataFrame(sens).set_index("param")
SENS15["NPV_span"] = (SENS15.NPV_hi - SENS15.NPV_lo).abs()
SENS15 = SENS15.sort_values("NPV_span")
print("\n  15d. Sensitivity (+/-30 %, lifetime 15-25 y):")
print(SENS15.round(3).to_string())

fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
for ax, col, ttl, cmap in zip(axes, ["CGE", "LHV", "NPV_kEUR", "GWP_kg_MJ"],
                              ["CGE (%)", "LHV (MJ/Nm3)", "NPV (kEUR, 15 y, 8 %)", "GWP100 (kg CO2-eq/MJ heat)"],
                              ["viridis", "viridis", "RdYlGn", "RdYlGn_r"]):
    m = MAP15.pivot(index="T", columns="ER", values=col)
    im = ax.imshow(m.values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels([f"{c:.2f}" for c in m.columns])
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}" if col == "GWP_kg_MJ" else f"{m.values[i, j]:.0f}" if col != "LHV" else f"{m.values[i, j]:.1f}",
                    ha="center", va="center", fontsize=7)
    ax.set_xlabel("ER"); ax.set_ylabel("T (C)"); ax.set_title(ttl, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(DATA_OUT / (FILE_TAG + "_T_ER_maps.png"), dpi=150, bbox_inches="tight"); plt.show()

# Same four indicators as curves, to show how they evolve: against temperature (one line per ER, top row) and against ER
# (one line per temperature, bottom row). Data: MAP15 (the 25 points of the map above; 70/30 blend, 209.8 kWth).
_cols15 = [("CGE", "CGE (%)"), ("LHV", "LHV (MJ/Nm3)"), ("NPV_kEUR", "NPV (kEUR, 15 y, 8 %)"), ("GWP_kg_MJ", "GWP100 (kg CO2-eq/MJ heat)")]
fig, axes = plt.subplots(2, 4, figsize=(21, 9.2))
_er_vals, _t_vals = sorted(MAP15.ER.unique()), sorted(MAP15["T"].unique())
_c_er = plt.cm.viridis(np.linspace(0.05, 0.9, len(_er_vals))); _c_t = plt.cm.plasma(np.linspace(0.05, 0.85, len(_t_vals)))
_mk = ["o", "s", "^", "D", "v"]
for j, (col, ttl) in enumerate(_cols15):
    m = MAP15.pivot(index="T", columns="ER", values=col)
    ax = axes[0, j]
    for i, er_ in enumerate(_er_vals):
        ax.plot(m.index, m[er_].values, marker=_mk[i % 5], ms=6, lw=1.8, color=_c_er[i], label=f"ER {er_:.2f}")
    ax.set_xlabel("T (C)"); ax.set_ylabel(ttl); ax.set_title(ttl + " vs temperature", fontsize=10); ax.set_xticks(list(m.index)); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=8, frameon=False, title="Equivalence ratio", title_fontsize=8)
    ax = axes[1, j]
    for i, t_ in enumerate(_t_vals):
        ax.plot(m.columns, m.loc[t_].values, marker=_mk[i % 5], ms=6, lw=1.8, color=_c_t[i], label=f"{t_:.0f} C")
    ax.set_xlabel("ER"); ax.set_ylabel(ttl); ax.set_title(ttl + " vs equivalence ratio", fontsize=10); ax.set_xticks(list(m.columns)); ax.grid(alpha=0.3)
    if j == 0: ax.legend(fontsize=8, frameon=False, title="Bed temperature", title_fontsize=8)
plt.tight_layout(); plt.savefig(DATA_OUT / (FILE_TAG + "_T_ER_curves.png"), dpi=200, bbox_inches="tight"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (lo_c, hi_c, basev, ttl) in zip(axes, [("NPV_lo", "NPV_hi", base["NPV_kEUR"], "NPV (kEUR)"),
                                               ("GWP_lo", "GWP_hi", base["GWP_kg_MJ"], "GWP100 (kg CO2-eq/MJ)")]):
    y = np.arange(len(SENS15))
    ax.barh(y, SENS15[lo_c] - basev, left=basev, color="tab:blue", label="-30 %")
    ax.barh(y, SENS15[hi_c] - basev, left=basev, color="tab:orange", label="+30 %")
    ax.axvline(basev, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(SENS15.index, fontsize=8)
    ax.set_title(f"Tornado - {ttl}"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(DATA_OUT / (FILE_TAG + "_tornado.png"), dpi=150, bbox_inches="tight"); plt.show()

# =============================================================================
# 16. PROCESS OPTIMIZATION AND UNCERTAINTY
#   16a. Multi-objective operating map: T x ER x RDF fraction (5 x 5 x 5 = 125 simulations)
#        objectives: max NPV (fixed 209.8 kWth demand), min tar (g/kg), min GWP100 (kg CO2-eq/MJ)
#        -> non-dominated (Pareto) set
#   16b. Scale optimization: six-tenths CAPEX scaling (Towler & Sinnott), break-even capacity
#   16c. Monte Carlo (60 model runs x 33 TEA/LCA draws = 1980 samples; uniform distributions,
#        lifetime discrete uniform, dolomite activity log-uniform; seed 2026) at the base
#        point and at the best-NPV Pareto point: kinetic (dT_WGS, tau_char, dolomite k0)
#        and TEA/LCA parameter uncertainty.
# =============================================================================
from dataclasses import replace as _dc_replace
rng = np.random.default_rng(2026)

def blend_rdf(x):
    return FeedstockConfig({"olive_pomace": OLIVE_POMACE, "RDF": RDF},
                           {"olive_pomace": 1.0 - x, "RDF": x}, f"OP{100*(1-x):.0f}_RDF{100*x:.0f}")

def run_point(T, ER, x_rdf, opt=OPT_H1, rc=REACTOR_TERRADOS, kps=KPS, feed=50.0):
    fs = blend_rdf(x_rdf)
    air = ER * stoich_air_Nm3_per_kg(fs) * feed
    sim = simulate(fs, rc, OperatingPoint(T, -30.0, feed, air), kps, opt)
    return sim, fs

print("\n" + "=" * 78)
print("16. PROCESS OPTIMIZATION AND UNCERTAINTY")
print("=" * 78, flush=True)
X_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
rows16 = []
for T in T_GRID:
    for er in ER_GRID:
        for x in X_GRID:
            sim, fs = run_point(T, er, x)
            tl = tea_lca(sim, fs)
            rows16.append(dict(T=T, ER=er, x_RDF=x, H2=sim["y"]["H2"], CO=sim["y"]["CO"], CH4=sim["y"]["CH4"],
                               LHV=sim["LHV"], CGE=sim["CGE"], tar_g_kg=sim["tar_g_kg"], **tl))
    print(f"  T = {T} C done", flush=True)
OPT16 = pd.DataFrame(rows16)

# Objectives of the screening: max NPV, min tar and min GWP100. PARETO_WITH_TAR = False drops tar (two objectives).
# The tar yield is calibrated on the reference case (eta, Eq. D1), so tar is kept as an objective.
PARETO_WITH_TAR = True
def pareto_mask(df, cols_max=("NPV_kEUR",), cols_min=(("tar_g_kg", "GWP_kg_MJ") if PARETO_WITH_TAR else ("GWP_kg_MJ",))):
    v = np.column_stack([-df[c].values for c in cols_max] + [df[c].values for c in cols_min])
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        if not keep[i]:
            continue
        dom = np.all(v <= v[i], axis=1) & np.any(v < v[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep
OPT16["pareto"] = pareto_mask(OPT16)
PAR = OPT16[OPT16.pareto].sort_values("NPV_kEUR", ascending=False)
# Constrained design point: max NPV in the Pareto set with CGE >= 70 %, GWP100 <= 0.5 GWP_NG and x_RDF >= 0.30. Used from here on.
_cons16 = PAR[(PAR.CGE >= 70) & (PAR.GWP_kg_MJ <= 0.5 * P["NG_reference"]) & (PAR.x_RDF >= 0.30 - 1e-9)]
T_COMP, ER_COMP, X_COMP = (float(_cons16.iloc[0]["T"]), float(_cons16.iloc[0].ER), float(_cons16.iloc[0].x_RDF)) if len(_cons16) else (1100.0, 0.20, 0.30)
print(f"    constrained design point: T {T_COMP:.0f} C, ER {ER_COMP:.2f}, RDF {100 * X_COMP:.0f} %")
print(f"\n  16a. Pareto set: {len(PAR)} of {len(OPT16)} operating points")
print(PAR[["T", "ER", "x_RDF", "CGE", "LHV", "tar_g_kg", "feed_kg_h", "NPV_kEUR", "IRR_pct", "GWP_kg_MJ"]].round(3).to_string(index=False))
best_npv = PAR.iloc[0]; best_gwp = PAR.sort_values("GWP_kg_MJ").iloc[0]; best_tar = PAR.sort_values("tar_g_kg").iloc[0]
for lab, r in [("max NPV", best_npv), ("min GWP", best_gwp), ("min tar", best_tar)]:
    print(f"    {lab:8s}: T {r['T']:.0f} C, ER {r.ER:.2f}, RDF {100*r.x_RDF:.0f} % -> NPV {r.NPV_kEUR:.1f} kEUR, "
          f"tar {r.tar_g_kg:.1f} g/kg, GWP {r.GWP_kg_MJ:.4f}, CGE {r.CGE:.1f} %")
er_b = air_n2_consistent(TABLE4.loc[700], BLEND_TERRADOS) / (stoich_air_Nm3_per_kg(BLEND_TERRADOS) * TABLE4.loc[700].feed_kg_h)
print(f"    base case (700 C, case B air) corresponds to ER = {er_b:.3f}")
OPT16.to_csv(DATA_OUT / "opt16_T_ER_xRDF.csv", index=False)

# ---- 16b. scale optimization ----
P_SCALE = dict(P, P_ref_kWth=209.8, capex_exp=0.6)
def npv_at_scale(sim, fs, Pk, p=P_SCALE):
    return tea_lca(sim, fs, dict(p, P_design_kWth=Pk))
sim_base, fs_base = run_point(700, er_b, 0.30)
sim_opt, fs_opt = run_point(best_npv["T"], best_npv.ER, best_npv.x_RDF)
SC = []
for Pk in [100, 209.8, 400, 800, 1500, 3000, 5000, 10000]:
    for lab, (s_, f_) in [("base", (sim_base, fs_base)), ("opt", (sim_opt, fs_opt))]:
        r = npv_at_scale(s_, f_, Pk)
        SC.append(dict(case=lab, P_kWth=Pk, NPV_kEUR=r["NPV_kEUR"], IRR_pct=r["IRR_pct"], PB_y=r["PB_y"],
                       CAPEX_kEUR=r["CAPEX_kEUR"], feed_kg_h=r["feed_kg_h"]))
SCALE16 = pd.DataFrame(SC)
print("\n  16b. NPV vs capacity (six-tenths CAPEX scaling, other parameters of Table 3):")
print(SCALE16.pivot(index="P_kWth", columns="case", values=["NPV_kEUR", "IRR_pct"]).round(1).to_string())
BE = {}
for lab, (s_, f_) in [("base", (sim_base, fs_base)), ("opt", (sim_opt, fs_opt))]:
    g = lambda Pk: npv_at_scale(s_, f_, Pk)["NPV_kEUR"]
    try:
        BE[lab] = _brentq(g, 100.0, 20000.0)
    except ValueError:
        BE[lab] = np.nan
    print(f"    break-even capacity ({lab}): {BE[lab]:.0f} kWth")
SCALE16.to_csv(DATA_OUT / "opt16_scale.csv", index=False)

# ---- 16c. Monte Carlo ----
N_MODEL, N_TEA = 60, 2000
def draw_tea(n):
    u = lambda a, b: rng.uniform(a, b, n)
    return pd.DataFrame(dict(capex_spec=u(0.7 * 1087, 1.3 * 1087), om_frac=u(0.03, 0.08), ng_price=u(35, 60), ets_price=u(60, 90),
                             discount_rate=u(0.06, 0.10), lifetime=rng.integers(10, 21, n), pomace_cost=u(20, 40),
                             rdf_avoided_cost=u(35, 60), elec_price=u(110, 160), boiler_eff=u(0.75, 0.85),
                             rdf_fossil_C_frac=u(0.45, 0.85), grid_EF=u(0.12, 0.21)))
MC = []
for lab, (T0, er0, x0) in [("base", (700, er_b, 0.30)), ("opt", (best_npv["T"], best_npv.ER, best_npv.x_RDF))]:
    for i in range(N_MODEL):
        fh2_, fco_ = mc_factors(rng); tau = rng.uniform(0.7, 1.3) * 7200.0; fdol = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        kps_i = mc_kps(fh2_, fdol)
        sim_i, fs_i = run_point(T0, er0, x0, opt=mc_opt(fco_),
                                rc=_dc_replace(REACTOR_TERRADOS, tau_char_s=tau), kps=kps_i)
        d = draw_tea(N_TEA // N_MODEL)
        for _, pr in d.iterrows():
            pp = dict(P, **pr.to_dict()); pp["lifetime"] = int(pp["lifetime"])
            r = tea_lca(sim_i, fs_i, pp)
            r1 = tea_lca(sim_i, fs_i, dict(pp, P_ref_kWth=209.8, capex_exp=0.6, P_design_kWth=1000.0))
            MC.append(dict(case=lab, CGE=sim_i["CGE"], LHV=sim_i["LHV"], tar=sim_i["tar_g_kg"],
                           NPV=r["NPV_kEUR"], GWP=r["GWP_kg_MJ"], NPV_1MW=r1["NPV_kEUR"]))
    print(f"  16c. Monte Carlo {lab} done", flush=True)
MC16 = pd.DataFrame(MC)
print("\n  16c. Monte Carlo summary (P5 / P50 / P95; probability NPV > 0):")
for lab, g in MC16.groupby("case", sort=False):
    q = g[["CGE", "LHV", "tar", "NPV", "NPV_1MW", "GWP"]].quantile([0.05, 0.5, 0.95]).T
    print(f"   [{lab}]\n" + q.round(3).to_string())
    print(f"     P(NPV > 0) at 209.8 kWth = {100*(g.NPV > 0).mean():.1f} % | at 1 MWth = {100*(g.NPV_1MW > 0).mean():.1f} % | "
          f"P(GWP < NG) = {100*(g.GWP < P['NG_reference']).mean():.1f} %")
MC16.to_csv(DATA_OUT / "opt16_montecarlo.csv", index=False)

# =============================================================================
# 17. PUBLICATION FIGURES (Biomass and Bioenergy style: 600 dpi PNG + vector PDF)
# =============================================================================
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
FIG_DIR = AUX_OUT / "figures_paper"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "savefig.dpi": 600, "figure.dpi": 110})
W1, W2 = 3.54, 7.48          # single / double column width (inches)
SPC = {"H2": "H$_2$", "CO": "CO", "CO2": "CO$_2$", "CH4": "CH$_4$"}
DSC = {"Calado 2023": ("#1f77b4", "o"), "Terrados 2026": ("#d62728", "s"),
       "de Deus 2025": ("#2ca02c", "^"),
       "Pio 2020": ("#9467bd", "v"),"FABER (Parrillo 2023)": ("#7f7f7f", "X")}
def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight"); fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

# ---- Auxiliary figure: modelling framework ----
fig, ax = plt.subplots(figsize=(W2, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="0.25", lw=0.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=5.8, linespacing=1.25)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2"))
box(0.5, 31, 16, 11, "Feedstock\nolive pomace + RDF\nproximate & ultimate\nanalyses", "#fdf0d5")
box(20.5, 31, 16, 11, "Devolatilization\ncomponent\nsurrogates\nC/H/O closure", "#e8f1fa")
box(40.5, 31, 18, 11, ("Bubbling bed (two zones)\nbubble + emulsion\nchar inventory\ndolomite tar kinetics" if LEVEL == "two_zone" else "Bubbling bed (CSTR)\n22 reactions\nchar inventory\ndolomite tar kinetics"), "#e8f1fa")
box(62.5, 31, 15, 11, "Freeboard (PFR)\nhomogeneous\nkinetics", "#e8f1fa")
box(81.5, 31, 18, 11, "Kinetic WGS\n(bed and\nfreeboard)", "#e3f4e3")
box(40.5, 16, 18, 10, "Syngas\ncomposition, LHV,\nCGE, tar, char\n(C/H/O/N closed)", "#f2f2f2")
box(62.5, 16, 15, 10, "Heat supply\n209.8 kWth\n(glass plant)", "#f2f2f2")
box(81.5, 16, 18, 10, "TEA: NPV, IRR\nLCA: GWP100\n(ISO 14040/44)", "#fbe3e3")
box(0.5, 16, 36, 10, ("Validation\n5 datasets, 3 reactors\nMAE, RMSE, bias" if LEVEL == "two_zone" else "Validation\n4 datasets, 3 reactors, blind test\nMAPE, RMSE, cross-validation"), "#fff7e6")
box(0.5, 1.5, 36, 10, "Data-consistency audit\nN$_2$ tie element\nimplied vs. declared ER", "#fff7e6")
box(40.5, 1.5, 59, 10, "Optimization and uncertainty\nT \u00d7 ER \u00d7 RDF fraction (Pareto: NPV, tar, GWP)\nsix-tenths scale analysis and Monte Carlo", "#efe6f7")
for a in [(17.3, 36.5, 20, 36.5), (37.3, 36.5, 40, 36.5), (59.3, 36.5, 62, 36.5), (78.3, 36.5, 81, 36.5),
          (90.5, 30.5, 55, 26.6), (59.3, 21, 62, 21), (78.3, 21, 81, 21), (37.3, 21, 40, 21), (37.3, 6.5, 40, 6.5)]:
    arr(*a)
save(fig, "Fig1_framework")

# ---- Auxiliary figure: parity plots ----
fig, axs = plt.subplots(2, 2, figsize=(W2, 6.2))
for ax, sp in zip(axs.flat, ["H2", "CO", "CO2", "CH4"]):
    lim = 0
    for ds, df in DATASETS.items():
        if f"{sp}_exp" not in df: continue
        d = df[[f"{sp}_exp", f"{sp}_mod"]].dropna()
        if d.empty: continue
        c, m = DSC.get(ds, ("k", "o"))
        hollow = ds == "Terrados 2026"
        ax.scatter(d[f"{sp}_exp"], d[f"{sp}_mod"], s=18, marker=m, facecolors="none" if hollow else c,
                   edgecolors=c, lw=0.7, label=short_name(ds), zorder=3)
        lim = max(lim, d.values.max())
    lim = np.ceil(lim * 1.1)
    xs = np.array([0, lim]); ax.plot(xs, xs, "k-", lw=0.7); ax.plot(xs, xs + 2, "k--", lw=0.5); ax.plot(xs, xs - 2, "k--", lw=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"Experimental {SPC[sp]} (vol%, dry)"); ax.set_ylabel(f"Model {SPC[sp]} (vol%, dry)")
    ax.set_title(f"({'abcd'[['H2','CO','CO2','CH4'].index(sp)]}) {SPC[sp]}", loc="left")
axs[0, 0].legend(loc="upper left", frameon=False, ncol=1)
fig.tight_layout(); save(fig, "Fig3_parity")

# ---- Auxiliary figure: Terrados A / B vs temperature ----
A_ = DATASETS["Terrados 2026"].set_index("T"); B_ = DATASETS["Terrados 2026"].set_index("T")
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.2))
for ax, sp in zip(axs, ["H2", "CO", "CO2", "CH4"]):
    ax.plot(B_.index, B_[f"{sp}_exp"], "ko", ms=4, label="Experimental")
    ax.plot(B_.index, B_[f"{sp}_mod"], "-s", color="#d62728", ms=3, label="Model, case B")
    ax.plot(A_.index, A_[f"{sp}_mod"], "--D", color="#ff7f0e", ms=3, mfc="none", label="Model, case A")
    ax.set_xlabel("T (\u00b0C)"); ax.set_title(SPC[sp] + " (vol%)"); ax.set_xticks([700, 900, 1100])
axs[0].legend(frameon=False, loc="best")
fig.tight_layout(); save(fig, "Fig4_Terrados_T")

# ---- Auxiliary figure: data consistency audit ----
r4 = TABLE4
n2rec = 100 * (r4.N2 / 100 * r4.Q_syngas_Nm3_h) / (0.79 * r4.air_Nm3_h)
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
axs[0].bar(r4.index.astype(str), n2rec, color="#d62728", width=0.6)
axs[0].axhline(100, color="k", lw=0.7, ls="--"); axs[0].set_ylim(0, 120)
axs[0].set_xlabel("T (\u00b0C)"); axs[0].set_ylabel("N$_2$ recovered (% of air N$_2$)"); axs[0].set_title("(a) Terrados: N$_2$ tie element", loc="left")
aud = pd.Series({"Pio": 1.00, "Calado": 1.31, "Terrados": 1.34})   # data-consistency check (section 3.1)
axs[1].barh(aud.index, aud.values, color=["#9467bd", "#1f77b4", "#d62728"], height=0.6)
axs[1].axvline(1, color="k", lw=0.7, ls="--"); axs[1].set_xlabel("ER implied by air flow / ER declared"); axs[1].set_title("(b) ER consistency", loc="left")
fig.tight_layout(); save(fig, "Fig2_data_audit")

# ---- Auxiliary figure: T x ER contour maps at 30 % RDF ----
m30 = OPT16[np.isclose(OPT16.x_RDF, 0.30)]
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3))
for ax, (col, ttl, cmap) in zip(axs, [("CGE", "(a) CGE (%)", "viridis"), ("LHV", "(b) LHV (MJ Nm$^{-3}$)", "viridis"),
                                      ("tar_g_kg", "(c) Tar (g kg$^{-1}$)", "magma_r"), ("GWP_kg_MJ", "(d) GWP$_{100}$ (kg MJ$^{-1}$)", "RdYlGn_r")]):
    pv = m30.pivot(index="T", columns="ER", values=col)
    cs = ax.contourf(pv.columns, pv.index, pv.values, levels=12, cmap=cmap)
    ax.contour(pv.columns, pv.index, pv.values, levels=6, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03); ax.set_title(ttl, loc="left")
    ax.set_xlabel("ER"); ax.set_ylabel("T (\u00b0C)" if col == "CGE" else "")
fig.tight_layout(); save(fig, "Fig5_T_ER_maps")

# ---- Auxiliary figure: Pareto front ----
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.scatter(OPT16.GWP_kg_MJ, OPT16.NPV_kEUR, s=8, c="0.8", label="All operating points")
sc = ax.scatter(PAR.GWP_kg_MJ, PAR.NPV_kEUR, s=22, c=PAR.tar_g_kg, cmap="magma_r", edgecolors="k", lw=0.4, label="Pareto set")
fig.colorbar(sc, ax=ax, label="Tar (g kg$^{-1}$)")
for lab, r in [("max NPV", best_npv), ("min GWP", best_gwp)]:
    ax.annotate(f"{lab}\n{r['T']:.0f} \u00b0C, ER {r.ER:.2f}, RDF {100*r.x_RDF:.0f}%", (r.GWP_kg_MJ, r.NPV_kEUR),
                xytext=(-150, -35) if lab == "max NPV" else (25, 30), textcoords="offset points", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5))
cons = PAR[(PAR.CGE >= 70) & (PAR.GWP_kg_MJ <= 0.5 * P["NG_reference"]) & (PAR.x_RDF >= 0.30 - 1e-9)].sort_values("NPV_kEUR", ascending=False)
if len(cons):
    rc_ = cons.iloc[0]
    ax.scatter([rc_.GWP_kg_MJ], [rc_.NPV_kEUR], s=70, facecolors="none", edgecolors="#2ca02c", lw=1.2, label="Constrained optimum")
    ax.annotate(f"CGE \u2265 70 %, GWP \u2264 0.5 NG, RDF \u2265 30 %\n{rc_['T']:.0f} \u00b0C, ER {rc_.ER:.2f}, RDF {100*rc_.x_RDF:.0f}%", (rc_.GWP_kg_MJ, rc_.NPV_kEUR),
                xytext=(18, -4), textcoords="offset points", fontsize=6.5, color="#2ca02c")
ax.axvline(P["NG_reference"], color="k", ls=":", lw=0.6)
ax.set_xlabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.set_ylabel("NPV (k\u20ac, 209.8 kWth)"); ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); save(fig, "Fig6_Pareto")

# ---- Auxiliary figure: scale and Monte Carlo ----
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
for lab, st in [("base", "-o"), ("opt", "-s")]:
    g = SCALE16[SCALE16.case == lab]; axs[0].plot(g.P_kWth, g.NPV_kEUR, st, ms=3, label=("Base" if lab == "base" else "Max-NPV point"))
axs[0].axhline(0, color="k", lw=0.6); axs[0].set_xscale("log"); axs[0].set_xlabel("Capacity (kWth)"); axs[0].set_ylabel("NPV (k\u20ac)")
axs[0].set_title("(a) Scale", loc="left"); axs[0].legend(frameon=False)
for lab, c in [("base", "#1f77b4"), ("opt", "#d62728")]:
    g = MC16[MC16.case == lab]
    axs[1].hist(g.NPV_1MW, bins=40, alpha=0.55, color=c, label=("Base" if lab == "base" else "Max-NPV point"))
    axs[2].hist(g.GWP, bins=40, alpha=0.55, color=c)
axs[1].axvline(0, color="k", lw=0.6); axs[1].set_xlabel("NPV at 1 MWth (k\u20ac)"); axs[1].set_ylabel("Frequency"); axs[1].set_title("(b) Monte Carlo", loc="left")
axs[2].axvline(P["NG_reference"], color="k", ls="--", lw=0.7); axs[2].text(P["NG_reference"], axs[2].get_ylim()[1] * 0.9, " NG", fontsize=6)
axs[2].set_xlabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); axs[2].set_title("(c) Monte Carlo", loc="left")
fig.tight_layout(); save(fig, "Fig7_scale_MC")

# ---- Auxiliary figure: GWP breakdown ----
def gwp_parts(sim, fs, p=P):
    tot = tea_lca(sim, fs, p)["GWP_kg_MJ"]
    ch4 = p["CH4_slip"] * p["GWP_CH4"]; el = p["elec_kWh_per_MJ"] * p["grid_EF"]
    return dict(fossil_CO2=tot - ch4 - el, CH4_slip=ch4, electricity=el)
GB = pd.DataFrame({"Base": gwp_parts(sim_base, fs_base), "Max NPV": gwp_parts(sim_opt, fs_opt),
                   "Min GWP": gwp_parts(*run_point(best_gwp["T"], best_gwp.ER, best_gwp.x_RDF))}).T
fig, ax = plt.subplots(figsize=(W1, 2.4))
bottom = np.zeros(len(GB))
for col, c in zip(GB.columns, ["#6c757d", "#e07a5f", "#3d405b"]):
    ax.bar(GB.index, GB[col], bottom=bottom, color=c, width=0.55, label=col.replace("_", " ")); bottom += GB[col].values
ax.bar(["Natural gas"], [P["NG_reference"]], color="#b5b5b5", width=0.55)
ax.set_ylabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "Fig9_GWP_breakdown")

# ---- Auxiliary figure: tornado ----
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
lbl = {"capex_spec": "Specific CAPEX", "om_frac": "O&M fraction", "ng_price": "Natural-gas price", "ets_price": "EU ETS price", "discount_rate": "Discount rate",
       "lifetime": "Lifetime", "pomace_cost": "Pomace cost", "boiler_eff": "Boiler efficiency", "elec_price": "Electricity price",
       "rdf_fossil_C_frac": "RDF fossil C fraction", "grid_EF": "Grid emission factor"}
for ax, (lo, hi, key, ttl) in zip(axs, [("NPV_lo", "NPV_hi", "NPV_kEUR", "(a) NPV (k\u20ac)"), ("GWP_lo", "GWP_hi", "GWP_kg_MJ", "(b) GWP$_{100}$ (kg MJ$^{-1}$)")]):
    bv = base[key]; S = SENS15.copy(); S["span"] = (S[hi] - S[lo]).abs(); S = S.sort_values("span")
    y = np.arange(len(S))
    ax.barh(y, S[lo] - bv, left=bv, color="#1f77b4", height=0.6, label="\u221230 %")
    ax.barh(y, S[hi] - bv, left=bv, color="#ff7f0e", height=0.6, label="+30 %")
    ax.axvline(bv, color="k", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([lbl.get(i, i) for i in S.index]); ax.set_title(ttl, loc="left")
axs[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "Fig8_tornado")
print(f"\n  17. Figures written to {FIG_DIR}: " + ", ".join(sorted(p.name for p in FIG_DIR.glob('*.png'))))

# =============================================================================
# 18. ENERGY-PERFORMANCE FIGURES (LHV, HHV, CGE, tar) AND FINAL FIGURE SET
#   HHV of the syngas from species higher heating values (MJ/Nm3, 0 C, 1 atm).
#   Final numbering: 1 framework | 2 parity | 3 composition vs T | 4 energy performance vs T
#   5 T x ER maps | 6 RDF fraction | 7 Pareto | 8 scale & MC | 9 tornado | 10 GWP breakdown
# =============================================================================
HHV_NM3 = {"H2": 12.75, "CO": 12.63, "CH4": 39.82, "C2H4": 63.40, "C2H6": 70.30, "C2H2": 58.06,
           "C3H6": 91.80, "H2S": 25.35}
def hhv_mix(y):
    return sum(HHV_NM3[k] * y.get(k, 0.0) for k in HHV_NM3) / 100.0

# --- Terrados (case B) model vs experiment ---
rowsT = []
for T, r in TABLE4.iterrows():
    s_ = simulate(BLEND_TERRADOS, REACTOR_TERRADOS,
                  OperatingPoint(T, r.P_gauge_kPa, r.feed_kg_h, air_n2_consistent(r, BLEND_TERRADOS)), KPS, OPT_H1)
    rowsT.append(dict(T=T, LHV_mod=s_["LHV"], HHV_mod=hhv_mix(s_["y"]), CGE_mod=s_["CGE"], tar_mod=s_["tar_g_kg"],
                      LHV_exp=r.LHV, HHV_exp=hhv_mix({k: r[k] for k in HHV_NM3 if k in r}), CGE_exp=r.CGE,
                      tar_exp=r.tar_g_kg, **{f"{k}_mod": s_["y"][k] for k in ["H2", "CO", "CO2", "CH4"]},
                      **{f"{k}_exp": r[k] for k in ["H2", "CO", "CO2", "CH4"]}))
EP = pd.DataFrame(rowsT).set_index("T")
print("\n" + "=" * 78)
print("18. ENERGY PERFORMANCE (Terrados, case B): model vs experiment")
print("=" * 78)
print(EP[["LHV_mod", "LHV_exp", "HHV_mod", "HHV_exp", "CGE_mod", "CGE_exp", "tar_mod", "tar_exp"]].round(2).to_string())
for v in ["LHV", "HHV", "CGE", "tar"]:
    e = EP[f"{v}_mod"] - EP[f"{v}_exp"]
    print(f"  {v}: MAE {np.mean(np.abs(e)):.2f} | RMSE {np.sqrt(np.mean(e**2)):.2f}")
    #print(f"  {v}: MAE {np.mean(np.abs(e)):.2f} | RMSE {np.sqrt(np.mean(e**2)):.2f} | MAPE {100*np.mean(np.abs(e)/EP[f'{v}_exp']):.1f} %")

# --- T x ER map with HHV (30 % RDF) ---
rowsM = []
for T in T_GRID:
    for er in ER_GRID:
        s_, f_ = run_point(T, er, 0.30)
        rowsM.append(dict(T=T, ER=er, CGE=s_["CGE"], LHV=s_["LHV"], HHV=hhv_mix(s_["y"]), tar=s_["tar_g_kg"]))
MAP18 = pd.DataFrame(rowsM)
# --- RDF-fraction effect at the base and at the constrained optimum ---
rowsX = []
for lab, (T0, er0) in [("Base (700 \u00b0C, ER 0.26)", (700, er_b)), (f"Optimum ({T_COMP:.0f} \u00b0C, ER {ER_COMP:.2f})", (T_COMP, ER_COMP))]:
    for x in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        if x == 0.0:
            fs0 = FeedstockConfig({"olive_pomace": OLIVE_POMACE}, {"olive_pomace": 1.0}, "OP100")
            s_ = simulate(fs0, REACTOR_TERRADOS, OperatingPoint(T0, -30.0, 50.0, er0 * stoich_air_Nm3_per_kg(fs0) * 50.0), KPS, OPT_H1)
        else:
            s_, _ = run_point(T0, er0, x)
        rowsX.append(dict(case=lab, x=x, CGE=s_["CGE"], LHV=s_["LHV"], HHV=hhv_mix(s_["y"]), tar=s_["tar_g_kg"]))
RDF18 = pd.DataFrame(rowsX)
print("\n  RDF fraction effect:")
print(RDF18.round(2).to_string(index=False))
EP.to_csv(DATA_OUT / "energy_performance_terrados.csv"); MAP18.to_csv(DATA_OUT / "map_T_ER_HHV.csv", index=False)
RDF18.to_csv(DATA_OUT / "rdf_fraction_effect.csv", index=False)

# ---- Fig. 1 of the article: structure of the model and of the assessment ----
fig, ax = plt.subplots(figsize=(W2, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="0.25", lw=0.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=5.8, linespacing=1.25)
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2"))
box(0.5, 31, 16, 11, "Feedstock\nolive pomace + RDF\nproximate & ultimate\nanalyses", "#fdf0d5")
box(20.5, 31, 16, 11, "Devolatilization\ncomponent\nsurrogates\nC/H/O closure", "#e8f1fa")
box(40.5, 31, 18, 11, ("Bubbling bed (two zones)\nbubble + emulsion\nchar inventory\ndolomite tar kinetics" if LEVEL == "two_zone" else "Bubbling bed (CSTR)\n22 reactions\nchar inventory\ndolomite tar kinetics"), "#e8f1fa")
box(62.5, 31, 15, 11, "Freeboard (PFR)\nhomogeneous\nkinetics", "#e8f1fa")
box(81.5, 31, 18, 11, "Kinetic WGS\n(bed and\nfreeboard)", "#e3f4e3")
box(40.5, 16, 18, 10, "Syngas\ncomposition, LHV, HHV,\nCGE, tar, char\n(C/H/O/N closed)", "#f2f2f2")
box(62.5, 16, 15, 10, "Heat supply\n209.8 kWth\n(glass plant)", "#f2f2f2")
box(81.5, 16, 18, 10, "TEA: NPV, IRR\nLCA: GWP100\n(ISO 14040/44)", "#fbe3e3")
box(0.5, 16, 36, 10, ("Validation\n5 datasets, 3 reactors\nMAE, RMSE, bias" if LEVEL == "two_zone" else "Validation\n4 datasets, 3 reactors, blind test\nMAPE, RMSE"), "#fff7e6")
box(0.5, 1.5, 36, 10, "Energy performance\nLHV, HHV, CGE and tar vs. T, ER\nand RDF fraction", "#fff7e6")
box(40.5, 1.5, 59, 10, "Optimization and uncertainty\nT \u00d7 ER \u00d7 RDF fraction (Pareto: NPV, tar, GWP)\nsix-tenths scale analysis and Monte Carlo", "#efe6f7")
for a in [(17.3, 36.5, 20, 36.5), (37.3, 36.5, 40, 36.5), (59.3, 36.5, 62, 36.5), (78.3, 36.5, 81, 36.5),
          (90.5, 30.5, 55, 26.6), (59.3, 21, 62, 21), (78.3, 21, 81, 21), (37.3, 21, 40, 21), (37.3, 6.5, 40, 6.5)]:
    arr(*a)
fig.savefig(FIG_ART / "Fig01_framework.png", bbox_inches="tight"); fig.savefig(FIG_ART / "Fig01_framework.pdf", bbox_inches="tight")
save(fig, "Fig1_framework")

# ---- Auxiliary figure: parity without the published-air case ----
fig, axs = plt.subplots(2, 2, figsize=(W2, 6.2))
for ax, sp in zip(axs.flat, ["H2", "CO", "CO2", "CH4"]):
    lim = 0
    for ds, df in DATASETS.items():
        if ds == "Terrados 2026" or f"{sp}_exp" not in df: continue
        d = df[[f"{sp}_exp", f"{sp}_mod"]].dropna()
        if d.empty: continue
        c, m = DSC.get(ds, ("k", "o"))
        lab = "Terrados" if ds.startswith("Terrados 2026") else short_name(ds)
        ax.scatter(d[f"{sp}_exp"], d[f"{sp}_mod"], s=18, marker=m, color=c, lw=0.7, label=lab, zorder=3)
        lim = max(lim, d.values.max())
    lim = np.ceil(lim * 1.1); xs = np.array([0, lim])
    ax.plot(xs, xs, "k-", lw=0.7); ax.plot(xs, xs + 2, "k--", lw=0.5); ax.plot(xs, xs - 2, "k--", lw=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"Experimental {SPC[sp]} (vol%, dry)"); ax.set_ylabel(f"Model {SPC[sp]} (vol%, dry)")
    ax.set_title(f"({'abcd'[['H2','CO','CO2','CH4'].index(sp)]}) {SPC[sp]}", loc="left")
axs[0, 0].legend(loc="upper left", frameon=False)
fig.tight_layout(); save(fig, "Fig2_parity")

# ---- Auxiliary figure: Terrados composition vs T, model vs experiment ----
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.2))
for ax, sp in zip(axs, ["H2", "CO", "CO2", "CH4"]):
    ax.plot(EP.index, EP[f"{sp}_exp"], "ko", ms=4, label="Experimental")
    ax.plot(EP.index, EP[f"{sp}_mod"], "-s", color="#d62728", ms=3, label="Model")
    ax.set_xlabel("T (\u00b0C)"); ax.set_title(SPC[sp] + " (vol%)"); ax.set_xticks([700, 900, 1100])
axs[0].legend(frameon=False)
fig.tight_layout(); save(fig, "Fig3_composition_T")

# ---- Energy performance vs T (auxiliary figure) ----
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3))
for ax, (v, ttl) in zip(axs, [("LHV", "(a) LHV (MJ Nm$^{-3}$)"), ("HHV", "(b) HHV (MJ Nm$^{-3}$)"),
                              ("CGE", "(c) CGE (%)"), ("tar", "(d) Tar (g kg$^{-1}$)")]):
    ax.plot(EP.index, EP[f"{v}_exp"], "ko", ms=4, label="Experimental")
    ax.plot(EP.index, EP[f"{v}_mod"], "-s", color="#d62728", ms=3, label="Model")
    ax.set_xlabel("T (\u00b0C)"); ax.set_title(ttl, loc="left"); ax.set_xticks([700, 900, 1100])
axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "Fig4_energy_performance")

# ---- Auxiliary figure: T x ER maps of CGE, LHV, HHV and tar ----
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3))
for ax, (col, ttl, cmap) in zip(axs, [("CGE", "(a) CGE (%)", "viridis"), ("LHV", "(b) LHV (MJ Nm$^{-3}$)", "viridis"),
                                      ("HHV", "(c) HHV (MJ Nm$^{-3}$)", "viridis"), ("tar", "(d) Tar (g kg$^{-1}$)", "magma_r")]):
    pv = MAP18.pivot(index="T", columns="ER", values=col)
    cs = ax.contourf(pv.columns, pv.index, pv.values, levels=12, cmap=cmap)
    ax.contour(pv.columns, pv.index, pv.values, levels=6, colors="k", linewidths=0.3)
    fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03); ax.set_title(ttl, loc="left")
    ax.set_xlabel("ER"); ax.set_ylabel("T (\u00b0C)" if col == "CGE" else "")
fig.tight_layout(); save(fig, "Fig5_T_ER_maps")

# ---- Effect of the RDF fraction (auxiliary figure) ----
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3))
for (lab, g), st in zip(RDF18.groupby("case", sort=False), ["-o", "-s"]):
    axs[0].plot(100 * g.x, g.CGE, st, ms=3, label=lab)
    axs[1].plot(100 * g.x, g.LHV, st, ms=3, label=f"LHV, {lab.split(' ')[0].lower()}")
    axs[1].plot(100 * g.x, g.HHV, st.replace("-", "--"), ms=3, mfc="none", label=f"HHV, {lab.split(' ')[0].lower()}")
    axs[2].plot(100 * g.x, g.tar, st, ms=3)
axs[0].set_title("(a) CGE (%)", loc="left"); axs[1].set_title("(b) Heating value (MJ Nm$^{-3}$)", loc="left")
axs[2].set_title("(c) Tar (g kg$^{-1}$)", loc="left")
for ax in axs: ax.set_xlabel("RDF in blend (wt%)")
axs[0].legend(frameon=False, fontsize=6); axs[1].legend(frameon=False, fontsize=5.5, ncol=1)
fig.tight_layout(); save(fig, "Fig6_RDF_fraction")

# ---- rename remaining figures to the final numbering ----
import shutil
for old, new in [("Fig6_Pareto", "Fig7_Pareto"), ("Fig7_scale_MC", "Fig8_scale_MC"),
                 ("Fig8_tornado", "Fig9_tornado")]:
    for ext in ("png", "pdf"):
        src = FIG_DIR / f"{old}.{ext}"
        if src.exists(): shutil.copy(src, FIG_DIR / f"{new}.{ext}")
for stale in ["Fig2_data_audit", "Fig3_parity", "Fig4_Terrados_T", "Fig6_Pareto", "Fig7_scale_MC", "Fig8_tornado", "Fig9_GWP_breakdown", "Fig10_GWP_breakdown"]:
    for ext in ("png", "pdf"):
        p_ = FIG_DIR / f"{stale}.{ext}"
        if p_.exists(): p_.unlink()
print("  18. Final figure set: " + ", ".join(sorted(p.name for p in FIG_DIR.glob('*.png'))))

# =============================================================================
# 19. ENERGY-ENVIRONMENTAL OPTIMIZATION OF THE OPERATING CONDITIONS
#   Decision variables: T (700-1100 C), ER (0.20-0.40), RDF fraction (0.10-0.50).
#   Objectives: max LHV (MJ/Nm3); min GWP100 (kg CO2-eq/MJ useful heat, fossil basis);
#               min direct CO2 emission (kg CO2/MJ useful heat, fossil + biogenic,
#               from the complete combustion of the syngas carbon).
#   Method: overall desirability D = (d_LHV d_GWP d_CO2)^(1/3) (Derringer, linear
#   one-sided functions on fixed bounds) maximized with differential evolution;
#   every evaluation is stored and the non-dominated set is extracted from them.
#   Robustness: Monte Carlo (kinetic parameters) at the optimum.
# =============================================================================
from scipy.optimize import differential_evolution
BOUNDS19 = [(700.0, 1100.0), (0.20, 0.40), (0.10, 0.50)]
DLIM = {"LHV": (3.0, 8.5), "GWP": (0.080, 0.005), "CO2": (0.30, 0.10)}   # (worst, best)

def env_eval(T, ER, x, opt=OPT_H1, rc=REACTOR_TERRADOS, kps=KPS):
    sim, fs = run_point(T, ER, x, opt=opt, rc=rc, kps=kps)
    tl = tea_lca(sim, fs)
    F = sim["F_out"]
    C_gas = sum(F[IDX[s]] * ATOM[IDX[s], 0] for s in PERMANENT_GAS)              # mol C/s
    heat_MJ_s = (sim["Q_dry_Nm3_h"] * sim["LHV"] / 3.6 - bed_heat_deficit_kW(sim)) / 1000.0 * P["boiler_eff"]   # useful heat (net)
    co2 = C_gas * 44.01e-3 / heat_MJ_s                                             # kg CO2/MJ
    return dict(T=T, ER=ER, x_RDF=x, LHV=sim["LHV"], CGE=sim["CGE"], tar=sim["tar_g_kg"],
                GWP=tl["GWP_kg_MJ"], CO2=co2, NPV=tl["NPV_kEUR"], H2=sim["y"]["H2"], CO=sim["y"]["CO"])

def desir(r):
    d = []
    for k in ["LHV", "GWP", "CO2"]:
        w, b = DLIM[k]
        d.append(np.clip((r[k] - w) / (b - w), 0.0, 1.0))
    return float(np.prod(d) ** (1 / 3)), d

EVALS19 = []
def neg_desir(z):
    r = env_eval(*z)
    D_, d = desir(r)
    r.update(D=D_, d_LHV=d[0], d_GWP=d[1], d_CO2=d[2]); EVALS19.append(r)
    return -D_

print("\n" + "=" * 78)
print("19. ENERGY-ENVIRONMENTAL OPTIMIZATION (max LHV, min GWP100, min CO2)")
print("=" * 78, flush=True)
de = differential_evolution(neg_desir, BOUNDS19, popsize=8, maxiter=12, tol=1e-4, seed=2026, polish=False)
OPT19 = pd.DataFrame(EVALS19)
v = np.column_stack([-OPT19.LHV, OPT19.GWP, OPT19.CO2]); keep = np.ones(len(v), bool)
for i in range(len(v)):
    if (np.all(v <= v[i], 1) & np.any(v < v[i], 1)).any(): keep[i] = False
OPT19["pareto"] = keep
best19 = OPT19.loc[OPT19.D.idxmax()]
print(f"  Evaluations: {len(OPT19)} | non-dominated: {int(keep.sum())}")
print(f"  Optimum (max desirability D = {best19.D:.3f}): T = {best19['T']:.0f} C, ER = {best19.ER:.3f}, RDF = {100*best19.x_RDF:.1f} %")
print(f"    LHV {best19.LHV:.2f} MJ/Nm3 | CGE {best19.CGE:.1f} % | GWP100 {best19.GWP:.4f} kg/MJ | CO2 {best19.CO2:.3f} kg/MJ | "
      f"tar {best19.tar:.1f} g/kg | NPV {best19.NPV:.1f} kEUR (209.8 kWth)")
for lab, r in [("max LHV", OPT19.loc[OPT19.LHV.idxmax()]), ("min GWP", OPT19.loc[OPT19.GWP.idxmin()]), ("min CO2", OPT19.loc[OPT19.CO2.idxmin()])]:
    print(f"    {lab:8s}: T {r['T']:.0f}, ER {r.ER:.3f}, RDF {100*r.x_RDF:.0f} % -> LHV {r.LHV:.2f}, GWP {r.GWP:.4f}, CO2 {r.CO2:.3f}, D {r.D:.3f}")
# Monte Carlo robustness at the optimum (kinetic uncertainty)
mc19 = []
for i in range(40):
    fh2_, fco_ = mc_factors(rng); tau = rng.uniform(0.7, 1.3) * 7200.0; fdol = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    kps_i = mc_kps(fh2_, fdol)
    r = env_eval(best19["T"], best19.ER, best19.x_RDF, opt=mc_opt(fco_),
                 rc=_dc_replace(REACTOR_TERRADOS, tau_char_s=tau), kps=kps_i)
    r["D"] = desir(r)[0]; mc19.append(r)
MC19 = pd.DataFrame(mc19)
print("  Monte Carlo at the optimum (P5 / P50 / P95):")
print(MC19[["LHV", "CGE", "GWP", "CO2", "D"]].quantile([0.05, 0.5, 0.95]).T.round(4).to_string())
OPT19.to_csv(DATA_OUT / "opt19_energy_environment.csv", index=False)

# ---- Auxiliary figure: energy-environmental optimization ----
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
sc = axs[0].scatter(OPT19.GWP, OPT19.LHV, c=OPT19.CO2, s=10, cmap="viridis_r", alpha=0.6)
axs[0].scatter(OPT19[OPT19.pareto].GWP, OPT19[OPT19.pareto].LHV, s=14, facecolors="none", edgecolors="k", lw=0.4, label="Non-dominated")
axs[0].scatter([best19.GWP], [best19.LHV], s=80, facecolors="none", edgecolors="#d62728", lw=1.3, label="Max desirability")
fig.colorbar(sc, ax=axs[0], label="CO$_2$ (kg MJ$^{-1}$)")
axs[0].set_xlabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); axs[0].set_ylabel("LHV (MJ Nm$^{-3}$)")
axs[0].set_title("(a) Evaluated operating points", loc="left"); axs[0].legend(frameon=False, fontsize=6, loc="lower right")

axs[1].set_title(f"(b) Desirability, RDF {100*best19.x_RDF:.0f} %", loc="left")
rows_d = []
for T in T_GRID:
    for er in ER_GRID:
        r = env_eval(T, er, best19.x_RDF); rows_d.append(dict(T=T, ER=er, D=desir(r)[0]))
DM = pd.DataFrame(rows_d).pivot(index="T", columns="ER", values="D")
cs = axs[1].contourf(DM.columns, DM.index, DM.values, levels=12, cmap="RdYlGn")
axs[1].contour(DM.columns, DM.index, DM.values, levels=6, colors="k", linewidths=0.3)
axs[1].plot([best19.ER], [best19["T"]], "k*", ms=8)
fig.colorbar(cs, ax=axs[1], label="Desirability D"); axs[1].set_xlabel("ER"); axs[1].set_ylabel("T (\u00b0C)")
fig.tight_layout(); save(fig, "Fig11_energy_environment_opt")


# =============================================================================
# 20. NON-ISOTHERMAL MODEL: BED ENERGY BALANCE (mode "T given -> air computed")
#   Control volume: bubbling bed. H_in (fuel as received + air at T_air) =
#   H_out (bed-outlet gas + char + ash at T_bed) + Q_loss,  Q_loss = UA (T_bed - 298 K).
#   Fuel enthalpy from its LHV (as received, Terrados Eq. 2): H_fuel = H(CO2, H2O(g), N2) + LHV*m.
#   Gas enthalpies: NASA-7 polynomials (POLIMI/CRECK thermodynamics, incl. formation).
#   Char = graphite (cp 1.4 kJ/kg K) and ash (cp 1.0 kJ/kg K): ASSUMED.
#   UA calibrated on Calado (same reactor, measured air), then Terrados air is predicted.
# =============================================================================
# (NASA_EB, h_molar and the heat capacities are defined in section 14d, where the bed heat demand is computed)

def bed_energy(fs, rc, op, opt=OPT_H1, T_air_C=25.0):
    """Returns (H_in - H_out) in W for the bed at the given operating point, and the simulation."""
    sim = simulate(fs, rc, op, KPS, opt)
    fe = sim["fe"]; T = op.T_C + 273.15
    m_fuel = op.feed_kg_h / 3600.0
    LHV_ar = (fe["HHV_blend"] - 0.212 * fe["H_pct"] - 0.0245 * fe["M_pct"] - LHV_O_COEF * fe["O_pct"]) * 1e6   # J/kg
    nC, nH, nN = fe["el"]["C"], fe["el"]["H"] + 2 * fe["moist_kg_s"] * 1000 / 18.015, fe["el"]["N"]
    H_fuel = nC * h_molar("CO2", T_REF) + nH / 2 * h_molar("H2O", T_REF) + nN / 2 * h_molar("N2", T_REF) + LHV_ar * m_fuel
    air = op.air_Nm3_h * MOL_PER_NM3 / 3600.0; Ta = T_air_C + 273.15
    H_air = air * (AIR_O2 * h_molar("O2", Ta) + (1 - AIR_O2) * h_molar("N2", Ta))
    Fb = sim["bed"]["F_out"]
    H_gas = sum(Fb[IDX[s]] * h_molar(s, T) for s in SPECIES if s in NASA_EB and Fb[IDX[s]] > 0)
    char_kg_s = (sim["bed"]["F_char_out"] + Fb[I_CS]) * AW["C"] / 1000.0
    H_solid = (char_kg_s * CP_CHAR + fe["ash_kg_s"] * CP_ASH) * (T - T_REF)
    return (H_fuel + H_air) - (H_gas + H_solid), sim

print("\n" + "=" * 78)
print("20. NON-ISOTHERMAL MODEL: bed energy balance (T given -> air computed)")
print("=" * 78, flush=True)
UA_i = []
for T, r in TABLE_CALADO.iterrows():
    dH, _ = bed_energy(FEED_CALADO, REACTOR_CALADO, OperatingPoint(T, r.P_gauge_kPa, r.feed_kg_h, r.air_Nm3_h))
    UA_i.append(dH / (T + 273.15 - T_REF))
    print(f"  Calado {T} C: surplus {dH/1e3:.1f} kW -> UA = {UA_i[-1]:.1f} W/K")
UA = float(np.mean(UA_i))
print(f"  UA (mean, Calado) = {UA:.1f} W/K  (range {min(UA_i):.1f}-{max(UA_i):.1f})")

print("  -> a negative UA means the bed balance only closes with an external heat input (not reported);")
print("     the non-isothermal model is therefore applied to the autothermal FABER plant (section 21).")

# =============================================================================
# 21. NON-ISOTHERMAL MODEL ON FABER (autothermal pilot plant): ER given -> bed T computed
#   Parrillo et al. (2023): quasi-autothermal operation, only the air is preheated.
#   Fuel energy from the as-received LHV of each component (Table 2 of Parrillo 2023).
#   UA (bed heat losses) calibrated leave-one-test-out; bed T predicted for the held-out test.
# =============================================================================
T_AIR_FABER = [99, 96, 96, 101, 97, 99, 104, 97, 100, 97, 99, 100, 97, 202]      # C, Table 3
LHV_AR_FABER = {"eucalyptus_chips": 14.85, "blupolymer_L": 40.5}                 # MJ/kg, Table 2

def bed_energy_lhv(fs, rc, op, LHV_MJ_kg, T_air_C, opt=OPT_H1):
    sim = simulate(fs, rc, op, KPS, opt)
    fe = sim["fe"]; T = op.T_C + 273.15; m_fuel = op.feed_kg_h / 3600.0
    nC, nH, nN = fe["el"]["C"], fe["el"]["H"] + 2 * fe["moist_kg_s"] * 1000 / 18.015, fe["el"]["N"]
    H_fuel = nC * h_molar("CO2", T_REF) + nH / 2 * h_molar("H2O", T_REF) + nN / 2 * h_molar("N2", T_REF) + LHV_MJ_kg * 1e6 * m_fuel
    air = op.air_Nm3_h * MOL_PER_NM3 / 3600.0; Ta = T_air_C + 273.15
    H_air = air * (AIR_O2 * h_molar("O2", Ta) + (1 - AIR_O2) * h_molar("N2", Ta))
    Fb = sim["bed"]["F_out"]
    H_gas = sum(Fb[IDX[s]] * h_molar(s, T) for s in SPECIES if s in NASA_EB and Fb[IDX[s]] > 0)
    char_kg_s = (sim["bed"]["F_char_out"] + Fb[I_CS]) * AW["C"] / 1000.0
    H_solid = (char_kg_s * CP_CHAR + fe["ash_kg_s"] * CP_ASH) * (T - T_REF)
    return (H_fuel + H_air) - (H_gas + H_solid), sim

print("\n" + "=" * 78)
print("21. NON-ISOTHERMAL MODEL ON FABER (ER given -> bed temperature computed)")
print("=" * 78, flush=True)
rows21 = []
for i, r in TABLE_FABER.iterrows():
    fs = faber_feed(r.PL, r.BM); lhv = sum(fs.mass_fractions[k] * LHV_AR_FABER[k] for k in fs.mass_fractions)
    dH, s_ = bed_energy_lhv(fs, REACTOR_FABER, OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), lhv, T_AIR_FABER[i])
    rows21.append(dict(test=r.test, T_meas=r.T_bed_C, dH_kW=dH / 1e3, UA=dH / (r.T_bed_C + 273.15 - T_REF), lhv=lhv))
F21 = pd.DataFrame(rows21)
print(F21[["test", "T_meas", "dH_kW", "UA"]].round(2).to_string(index=False))
print(f"  UA at measured T: mean {F21.UA.mean():.1f} W/K, range {F21.UA.min():.1f} to {F21.UA.max():.1f}", flush=True)

pred = []
for i, r in TABLE_FABER.iterrows():
    UA_i = F21.drop(index=i).UA.mean()
    fs = faber_feed(r.PL, r.BM); lhv = F21.loc[i, "lhv"]
    f = lambda Tc: bed_energy_lhv(fs, REACTOR_FABER, OperatingPoint(Tc, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), lhv, T_AIR_FABER[i])[0] - UA_i * (Tc + 273.15 - T_REF)
    try:
        Tp = _brentq(f, 600.0, 1200.0, xtol=1.0)
    except ValueError:
        Tp = np.nan
    s_iso = simulate(fs, REACTOR_FABER, OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), KPS, OPT_H1)
    s_ni = simulate(fs, REACTOR_FABER, OperatingPoint(Tp, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), KPS, OPT_H1) if Tp == Tp else s_iso
    pred.append(dict(test=r.test, T_meas=r.T_bed_C, T_pred=Tp, UA=UA_i,
                     **{f"{k}_iso": s_iso["y"][k] for k in ["H2", "CO", "CO2", "CH4"]}, LHV_iso=s_iso["LHV"],
                     **{f"{k}_ni": s_ni["y"][k] for k in ["H2", "CO", "CO2", "CH4"]}, LHV_ni=s_ni["LHV"],
                     **{f"{k}_exp": r[k] for k in ["H2", "CO", "CO2", "CH4", "LHV"]}))
    print(f"  {r.test}: T measured {r.T_bed_C} C | predicted {Tp:.0f} C", flush=True)
P21 = pd.DataFrame(pred)
e = P21.T_pred - P21.T_meas
print(f"\n  Bed temperature (leave-one-test-out): MAE {np.mean(np.abs(e)):.1f} K | RMSE {np.sqrt(np.mean(e**2)):.1f} K | "
      f"bias {e.mean():+.1f} K | r = {np.corrcoef(P21.T_pred, P21.T_meas)[0,1]:.2f}")
for v in ["H2", "CO", "CO2", "CH4", "LHV"]:
    a = np.mean(np.abs(P21[f"{v}_iso"] - P21[f"{v}_exp"])); b = np.mean(np.abs(P21[f"{v}_ni"] - P21[f"{v}_exp"]))
    print(f"  {v:4s}: MAE isothermal (measured T) {a:.2f} | non-isothermal (predicted T) {b:.2f}")
P21.to_csv(DATA_OUT / "faber_nonisothermal.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8))
axs[0].scatter(P21.T_meas, P21.T_pred, s=20, c="#7f7f7f", edgecolors="k", lw=0.4)
lo, hi = 680, 980; axs[0].plot([lo, hi], [lo, hi], "k-", lw=0.7)
axs[0].plot([lo, hi], [lo + 25, hi + 25], "k--", lw=0.5); axs[0].plot([lo, hi], [lo - 25, hi - 25], "k--", lw=0.5)
axs[0].set_xlim(lo, hi); axs[0].set_ylim(lo, hi); axs[0].set_aspect("equal")
axs[0].set_xlabel("Measured bed T (\u00b0C)"); axs[0].set_ylabel("Predicted bed T (\u00b0C)"); axs[0].set_title("(a) Bed temperature", loc="left")
sp_ = ["H2", "CO", "CO2", "CH4"]; x = np.arange(len(sp_)); w = 0.35
m_iso = [np.mean(np.abs(P21[f"{k}_iso"] - P21[f"{k}_exp"])) for k in sp_]
m_ni = [np.mean(np.abs(P21[f"{k}_ni"] - P21[f"{k}_exp"])) for k in sp_]
axs[1].bar(x - w / 2, m_iso, w, color="#1f77b4", label="Isothermal (measured T)")
axs[1].bar(x + w / 2, m_ni, w, color="#d62728", label="Non-isothermal (predicted T)")
axs[1].set_xticks(x); axs[1].set_xticklabels([SPC[k] for k in sp_])
axs[1].set_ylabel("MAE (percentage points)"); axs[1].set_title("(b) Syngas composition error, 14 tests", loc="left")
axs[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "Fig12_FABER_nonisothermal")

# =============================================================================
# 22. INTEGRATED TEA-LCA: LEVELISED COST OF HEAT, CO2 ABATEMENT COST, SYSTEM EXPANSION
#     AND PROSPECTIVE SCENARIOS (2025 / 2030)
#   LCOH (EUR/MWh useful heat) = [I0 CRF + O&M + pomace + electricity + ETS on fossil CO2
#                                  - avoided landfill fee of RDF] / heat
#   NG reference heat cost      = NG price / eta_NG + ETS * EF_NG,direct / eta_NG
#   System expansion: avoided landfill CH4 of the RDF, IPCC methane generation potential L0
#   (mass-balance method = cumulative FOD emission without time truncation). The degradable
#   organic carbon is the biogenic carbon of the RDF: DOC = (1 - f_fossil) * C_RDF (wet basis),
#   consistent with the fossil fraction used in the attributional GWP.
#   Abatement cost (EUR/t CO2-eq) = (LCOH - cost_NG) / (GWP_NG - GWP_net)   [GWP in t/MWh]
#   Two functional units: 1 MJ useful heat and 1 t of blend treated.
# =============================================================================
SCEN = {
    "2025": dict(ng_price=48.1, ets=75.0, rdf_avoided_cost=35.0, grid_EF=0.166),   # Eurostat H2 2025; ETS 60-90; TGR 2025; grid 2023
    "2030": dict(ng_price=48.1, ets=126.0, rdf_avoided_cost=60.0, grid_EF=0.030),  # ETS consensus 2030; TGR 2030; PNEC 93 % RES-E
}
ETA_NG, EF_NG_DIRECT = 0.90, 0.2020          # NG boiler efficiency (assumed); 56.1 kg CO2/GJ fuel = 0.202 t/MWh_fuel (IPCC)
# Landfill of RDF (IPCC 2019 Refinement, L0): DOC from biogenic C, DOCf 0.5, MCF 1, F 0.5, OX 0.1, gas recovery R 0.5
C_RDF_WET = RDF.ultimate_dry["C"] / sum(RDF.ultimate_dry.values()) * (1 - RDF.moisture_ar / 100.0)
LF = dict(DOC=(1 - P["rdf_fossil_C_frac"]) * C_RDF_WET, DOCf=0.5, MCF=1.0, F=0.5, OX=0.1, R=0.5)
def landfill_EF(lf=LF):
    ch4_gen = 1000.0 * lf["DOC"] * lf["DOCf"] * lf["MCF"] * lf["F"] * 16.0 / 12.0       # kg CH4/t
    return ch4_gen * (1 - lf["R"]) * (1 - lf["OX"]) * P["GWP_CH4_biogenic"]                    # kg CO2-eq/t RDF

def tea_lca2(sim, fs, sc, Pd=209.8, p=P):
    pp = dict(p, P_design_kWth=Pd, P_ref_kWth=209.8, capex_exp=0.6 if Pd != 209.8 else 1.0,
              rdf_avoided_cost=sc["rdf_avoided_cost"], grid_EF=sc["grid_EF"], ng_price=sc["ng_price"], ets_price=sc["ets"])
    t = tea_lca(sim, fs, pp)
    heat = t["heat_MWh_y"]; MJ = heat * 3600.0
    i, N = pp["discount_rate"], pp["lifetime"]; crf = i * (1 + i) ** N / ((1 + i) ** N - 1)
    capex = t["CAPEX_kEUR"] * 1e3
    feed = t["feed_kg_h"]; w = fs.mass_fractions
    rdf_t = feed * w.get("RDF", 0) * pp["hours"] / 1000.0; pom_t = feed * w.get("olive_pomace", 0) * pp["hours"] / 1000.0
    elec = pp["elec_kWh_per_MJ"] * MJ / 1000.0
    fossil_t = (t["GWP_kg_MJ"] - pp["CH4_slip"] * pp["GWP_CH4"] - pp["elec_kWh_per_MJ"] * pp["grid_EF"]) * MJ / 1000.0
    cost = capex * crf + pp["om_frac"] * capex + pom_t * pp["pomace_cost"] + elec * pp["elec_price"] + fossil_t * sc["ets"] \
           - rdf_t * sc["rdf_avoided_cost"]
    lcoh = cost / heat
    c_ng = t["heat_price"]
    lf = landfill_EF()
    gwp_attr = t["GWP_kg_MJ"]
    gwp_net = gwp_attr - rdf_t * lf / MJ
    abat = (lcoh - c_ng) / ((P["NG_reference"] - gwp_net) * 3.6)            # EUR/t CO2-eq
    abat_attr = (lcoh - c_ng) / ((P["NG_reference"] - gwp_attr) * 3.6)
    feed_t = feed * pp["hours"] / 1000.0
    fu_waste = ((gwp_attr - P["NG_reference"]) * MJ - rdf_t * lf) / feed_t   # kg CO2-eq per t blend (negative = saving)
    return dict(NPV=t["NPV_kEUR"], IRR=t["IRR_pct"], LCOH=lcoh, cost_NG=c_ng, GWP_attr=gwp_attr, GWP_net=gwp_net, abat_net=abat, abat_attr=abat_attr,
                FU_waste=fu_waste, landfill_EF=lf, feed_kg_h=feed, rdf_t=rdf_t)

print("\n" + "=" * 78)
print("22. INTEGRATED TEA-LCA: LCOH, CO2 ABATEMENT COST, SYSTEM EXPANSION, SCENARIOS")
print("=" * 78)
print(f"  DOC of RDF = (1 - f_fossil) * C_RDF = {LF['DOC']:.4f};  landfill emission factor (IPCC L0): {landfill_EF():.0f} kg CO2-eq/t")
for f_ in (0.45, 0.85):
    print(f"  landfill EF with RDF fossil C fraction {f_}: {landfill_EF(dict(LF, DOC=(1 - f_) * C_RDF_WET)):.0f} kg CO2-eq/t")
PTS22 = {"Base (700 C, ER 0.26, 30 % RDF)": (700, er_b, 0.30), f"Optimum ({T_COMP:.0f} C, ER {ER_COMP:.2f}, {100 * X_COMP:.0f} % RDF)": (T_COMP, ER_COMP, X_COMP)}
rows22 = []
for lab, (T0, er0, x0) in PTS22.items():
    s_, f_ = run_point(T0, er0, x0)
    for sc_name, sc in SCEN.items():
        for Pd in (209.8, 1000.0):
            r = tea_lca2(s_, f_, sc, Pd); r.update(point=lab, scenario=sc_name, P_kWth=Pd); rows22.append(r)
T22 = pd.DataFrame(rows22)
print(T22[["point", "scenario", "P_kWth", "LCOH", "cost_NG", "GWP_attr", "GWP_net", "abat_attr", "abat_net", "FU_waste"]].round(3).to_string(index=False))
# sensitivity of the landfill factor (gas recovery 0.3-0.7)
for R_ in (0.3, 0.7):
    print(f"  landfill EF with gas recovery {R_}: {landfill_EF(dict(LF, R=R_)):.0f} kg CO2-eq/t")
T22.to_csv(DATA_OUT / "tea_lca_integrated.csv", index=False)

# ---- Integrated TEA-LCA (auxiliary figure) ----
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6))
b = T22[(T22.scenario == "2025") & (T22.P_kWth == 209.8)]
labs = ["Base", "Optimum"]; x = np.arange(2); w_ = 0.35
axs[0].bar(x - w_ / 2, b.GWP_attr, w_, color="#6c757d", label="Attributional")
axs[0].bar(x + w_ / 2, b.GWP_net, w_, color="#2ca02c", label="With avoided landfill")
axs[0].axhline(P["NG_reference"], color="k", ls="--", lw=0.7); axs[0].text(1.45, P["NG_reference"], "NG", fontsize=6, va="bottom")
axs[0].axhline(0, color="k", lw=0.5); axs[0].set_xticks(x); axs[0].set_xticklabels(labs)
axs[0].set_ylabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); axs[0].set_title("(a) Carbon footprint, 2025", loc="left"); axs[0].legend(frameon=False, fontsize=5.5)
o_ = T22[T22.point.str.startswith("Optimum")]
for (sc_name, Pd), st, c in [(("2025", 209.8), "o", "#1f77b4"), (("2025", 1000.0), "s", "#1f77b4"), (("2030", 209.8), "o", "#d62728"), (("2030", 1000.0), "s", "#d62728")]:
    g = o_[(o_.scenario == sc_name) & (o_.P_kWth == Pd)]
    axs[1].bar(f"{sc_name}\n{Pd/1000:.1f} MW" if Pd > 300 else f"{sc_name}\n0.21 MW", g.LCOH, color=c, alpha=0.8 if Pd > 300 else 0.45)
    axs[2].bar(f"{sc_name}\n{Pd/1000:.1f} MW" if Pd > 300 else f"{sc_name}\n0.21 MW", g.abat_net, color=c, alpha=0.8 if Pd > 300 else 0.45)
for sc_name, c in [("2025", "#1f77b4"), ("2030", "#d62728")]:
    axs[1].axhline(SCEN[sc_name]["ng_price"] / ETA_NG + SCEN[sc_name]["ets"] * EF_NG_DIRECT / ETA_NG, color=c, ls="--", lw=0.8)
    axs[2].axhline(SCEN[sc_name]["ets"], color=c, ls="--", lw=0.8)
axs[1].set_ylabel("LCOH (\u20ac MWh$^{-1}$)"); axs[1].set_title("(b) Cost of heat (optimum)", loc="left"); axs[1].tick_params(axis="x", labelsize=5.5)
axs[2].set_ylabel("Abatement cost (\u20ac t$^{-1}$ CO$_2$-eq)"); axs[2].set_title("(c) CO$_2$ abatement cost", loc="left"); axs[2].axhline(0, color="k", lw=0.5); axs[2].tick_params(axis="x", labelsize=5.5)
fig.tight_layout(); save(fig, "Fig10_TEA_LCA_integrated")

# =============================================================================
# 23. RESULTS RESTRUCTURING: R2, BASELINE PERFORMANCE, FREEBOARD PROFILE, INDUSTRIAL
#     REQUIREMENTS, LCOH BREAKDOWN, LCA CONTRIBUTIONS, LCOH-GWP MAP, EXTENDED
#     SENSITIVITY AND MONTE CARLO; FINAL FIGURE SET (Figs. 1-15)
# =============================================================================
import shutil
from dataclasses import replace as _dcr
FIG23 = AUX_OUT / "figures_aux_tea"
def save23(fig, name):
    fig.savefig(FIG23 / f"{name}.png", bbox_inches="tight"); fig.savefig(FIG23 / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
print("\n" + "=" * 78); print("23. RESULTS RESTRUCTURING"); print("=" * 78, flush=True)

# ---- 23a. R2 of the validation (all datasets except the published-air case) ----
R2 = {}
for sp in ["H2", "CO", "CO2", "CH4"]:
    e = pd.concat([df[[f"{sp}_exp", f"{sp}_mod"]].dropna() for ds, df in DATASETS.items() if ds != "Terrados 2026" and f"{sp}_exp" in df])
    t = DATASETS["Terrados 2026"][[f"{sp}_exp", f"{sp}_mod"]].dropna()
    r2 = lambda d: 1 - ((d.iloc[:, 1] - d.iloc[:, 0]) ** 2).sum() / ((d.iloc[:, 0] - d.iloc[:, 0].mean()) ** 2).sum()
    R2[sp] = dict(global_=r2(e), terrados=r2(t), n=len(e))
print("  23a. R2 (global / Terrados):", {k: (round(v["global_"], 3), round(v["terrados"], 3), v["n"]) for k, v in R2.items()})

# ---- integrated TEA-LCA evaluation with explicit parameters ----
def tl3(sim, fs, p=P, Pd=209.8):
    pp = dict(p, P_design_kWth=Pd, P_ref_kWth=209.8, capex_exp=0.6 if Pd != 209.8 else 1.0)
    t = tea_lca(sim, fs, pp)
    heat = t["heat_MWh_y"]; MJ = heat * 3600.0
    i, N = pp["discount_rate"], int(pp["lifetime"]); crf = i * (1 + i) ** N / ((1 + i) ** N - 1)
    capex = t["CAPEX_kEUR"] * 1e3; feed = t["feed_kg_h"]; w = fs.mass_fractions
    rdf_t = feed * w.get("RDF", 0) * pp["hours"] / 1000.0; pom_t = feed * w.get("olive_pomace", 0) * pp["hours"] / 1000.0
    elec = pp["elec_kWh_per_MJ"] * MJ / 1000.0
    fos_t = (t["GWP_kg_MJ"] - pp["CH4_slip"] * pp["GWP_CH4"] - pp["elec_kWh_per_MJ"] * pp["grid_EF"]) * MJ / 1000.0
    comp = dict(capex=capex * crf / heat, om=pp["om_frac"] * capex / heat, pomace=pom_t * pp["pomace_cost"] / heat,
                elec=elec * pp["elec_price"] / heat, co2=fos_t * pp["ets_price"] / heat, rdf=-rdf_t * pp["rdf_avoided_cost"] / heat)
    lcoh = sum(comp.values())
    lf = landfill_EF(dict(LF, DOC=(1 - pp["rdf_fossil_C_frac"]) * C_RDF_WET))
    gwp = t["GWP_kg_MJ"]; gwp_net = gwp - rdf_t * lf / MJ
    gwp_parts = dict(fossil=fos_t * 1000 / MJ, ch4=pp["CH4_slip"] * pp["GWP_CH4"], elec=pp["elec_kWh_per_MJ"] * pp["grid_EF"], landfill=-rdf_t * lf / MJ)
    abat = (lcoh - t["heat_price"]) / ((P["NG_reference"] - gwp_net) * 3.6)
    feed_t = feed * pp["hours"] / 1000.0
    return dict(NPV=t["NPV_kEUR"], IRR=t["IRR_pct"], PB=t["PB_y"], LCOH=lcoh, p_heat=t["heat_price"], GWP=gwp, GWP_net=gwp_net, abat=abat,
                FU_t=((gwp - P["NG_reference"]) * MJ - rdf_t * lf) / feed_t, comp=comp, gwp_parts=gwp_parts, lf=lf,
                heat=heat, feed_kg_h=feed, feed_t=feed_t, rdf_t=rdf_t, pom_t=pom_t, elec=elec, fos_t=fos_t, CAPEX=capex)

# ---- 23b. Baseline and compromise performance ----
PTS = {"base": (700, er_b, 0.30), "comp": (T_COMP, ER_COMP, X_COMP)}
SIMS = {k: run_point(*v) for k, v in PTS.items()}
def perf(sim, fs, feed=50.0):
    fe = sim["fe"]; F = sim["F_out"]
    C_gas = sum(F[IDX[s]] * ATOM[IDX[s], 0] for s in PERMANENT_GAS)
    Pth = sim["Q_dry_Nm3_h"] * sim["LHV"] / 3.6
    return dict(H2=sim["y"]["H2"], CO=sim["y"]["CO"], CO2=sim["y"]["CO2"], CH4=sim["y"]["CH4"], N2=sim["y"]["N2"],
                H2_CO=sim["y"]["H2"] / sim["y"]["CO"], Q=sim["Q_dry_Nm3_h"], Q_spec=sim["Q_dry_Nm3_h"] / feed, LHV=sim["LHV"],
                HHV=hhv_mix(sim["y"]), CGE=sim["CGE"], CCE=100 * C_gas / fe["el"]["C"], X_char=100 * (1 - sim["bed"]["F_char_out"] / (sim["bed"]["F_char_out"] + 0) if False else 1),
                tar=sim["tar_g_kg"], char=sim["char_g_kg"], Pth=Pth, Pth_spec=Pth / feed, tau_bed=sim["bed"]["tau_gas"], tau_fb=sim["tau_fb"])
PERF = {k: perf(*SIMS[k]) for k in SIMS}
for k in PERF:
    s_, f_ = SIMS[k]; fe = s_["fe"]
    n_in = fe["fc_kg_s"] * 1000 / AW["C"]; PERF[k]["X_char"] = 100 * (1 - s_["bed"]["F_char_out"] / n_in)
    PERF[k]["ER"] = PTS[k][1]; PERF[k]["T"] = PTS[k][0]
print("  23b. Baseline / compromise performance:"); print(pd.DataFrame(PERF).round(3).to_string())

# ---- 23c. Freeboard composition profile (baseline) ----
s_, f_ = SIMS["base"]; hyd = s_["hyd"]; T_K = 700 + 273.15; P_abs = P_ATM - 30.0 * 1000
kps_e = KPS; F0 = s_["bed"]["F_out"].copy()          # bed outlet; kinetic WGS in the freeboard (no equilibrium step)
zs = np.linspace(0, hyd["H_fb"], 25); prof = []
for z in zs:
    Fz = F0.copy() if z == 0 else freeboard_pfr(F0, T_K, P_abs, hyd["A"], z, kps_e, OPT_H1)
    dry = sum(Fz[IDX[k]] for k in ["H2", "N2", "CO", "CO2", "CH4", "C2H4", "C3H6", "O2"] if k in IDX)
    Qn = sum(Fz[IDX[k]] for k in SPECIES if k not in ("C_s",)) / MOL_PER_NM3 * 3600
    tar = sum(Fz[IDX[k]] * MW[IDX[k]] for k in ["C6H6", "C7H8", "C6H6O", "C10H8", "Tar"] if k in IDX)   # g/s
    prof.append(dict(z=z, **{k: 100 * Fz[IDX[k]] / dry for k in ["H2", "CO", "CO2", "CH4"]}, C2H4=100 * Fz[IDX["C2H4"]] / dry,
                     tar_g_Nm3=tar * 3600 / (dry / MOL_PER_NM3 * 3600)))
PROF = pd.DataFrame(prof)
print("  23c. Freeboard profile (inlet -> outlet):"); print(PROF.iloc[[0, 6, 12, 24]].round(3).to_string(index=False))

# ---- 23d. Industrial requirements, TEA and LCA at 209.8 kWth (2025) ----
TL = {k: tl3(*SIMS[k]) for k in SIMS}
IND = {}
for k in TL:
    t = TL[k]; pf = PERF[k]
    IND[k] = dict(useful_heat_MWh=t["heat"], syngas_heat_MWh=209.8 * P["hours"] / 1000, syngas_Nm3=pf["Q_spec"] * t["feed_kg_h"] * P["hours"],
                  feed_t=t["feed_t"], pomace_t=t["pom_t"], rdf_t=t["rdf_t"], elec_MWh=t["elec"], fossil_CO2_t=t["fos_t"],
                  landfill_avoided_tCO2e=t["rdf_t"] * t["lf"] / 1000, feed_kg_h=t["feed_kg_h"])
print("  23d. Annual requirements:"); print(pd.DataFrame(IND).round(1).to_string())
print("  LCOH components (EUR/MWh):"); print(pd.DataFrame({k: TL[k]["comp"] for k in TL}).round(2).to_string())
print("  GWP parts (kg/MJ):"); print(pd.DataFrame({k: TL[k]["gwp_parts"] for k in TL}).round(4).to_string())
pd.DataFrame({k: TL[k]["comp"] for k in TL}).to_csv(DATA_OUT / "lcoh_components.csv"); pd.DataFrame({k: TL[k]["gwp_parts"] for k in TL}).to_csv(DATA_OUT / "gwp_parts.csv")   # used by section 25
print("  Summary:", {k: {m: round(TL[k][m], 3) for m in ["NPV", "IRR", "PB", "LCOH", "p_heat", "GWP", "GWP_net", "abat", "FU_t", "lf"]} for k in TL})
pd.DataFrame(PERF).to_csv(DATA_OUT / "baseline_compromise_performance.csv"); PROF.to_csv(DATA_OUT / "freeboard_profile.csv", index=False)
pd.DataFrame(IND).to_csv(DATA_OUT / "industrial_requirements.csv")

# ---- 23e. LCOH-GWP map over the 125-point grid ----
rows = []
for T in T_GRID:
    for er in ER_GRID:
        for x in [0.10, 0.20, 0.30, 0.40, 0.50]:
            s1, f1 = run_point(T, er, x); r = tl3(s1, f1)
            rows.append(dict(T=T, ER=er, x_RDF=x, CGE=s1["CGE"], tar=s1["tar_g_kg"], **{m: r[m] for m in ["NPV", "IRR", "LCOH", "GWP", "GWP_net", "abat"]}))
    print(f"    grid T = {T} done", flush=True)
MAP23 = pd.DataFrame(rows); MAP23.to_csv(DATA_OUT / "grid_LCOH_GWP.csv", index=False)
mL = MAP23.loc[MAP23.LCOH.idxmin()]
print(f"  23e. LCOH range {MAP23.LCOH.min():.1f}-{MAP23.LCOH.max():.1f}; min LCOH at T {mL['T']:.0f}, ER {mL.ER:.2f}, RDF {100*mL.x_RDF:.0f} % "
      f"(LCOH {mL.LCOH:.1f}, NPV {mL.NPV:.1f}, GWP {mL.GWP:.4f}); LCOH < p_heat in {100*(MAP23.LCOH < TL['base']['p_heat']).mean():.0f} % of points")
SOL = {"Min LCOH": mL, "Min GWP": MAP23.loc[MAP23.GWP.idxmin()], "Max NPV": MAP23.loc[MAP23.NPV.idxmax()],
       "Compromise": MAP23[np.isclose(MAP23["T"], T_COMP) & np.isclose(MAP23.ER, ER_COMP) & np.isclose(MAP23.x_RDF, X_COMP)].iloc[0]}
print(pd.DataFrame(SOL).T[["T", "ER", "x_RDF", "CGE", "tar", "LCOH", "NPV", "IRR", "GWP", "GWP_net", "abat"]].round(3).to_string())

# ---- 23f. One-at-a-time sensitivity of NPV, LCOH, GWP_net and abatement cost (baseline) ----
# Each parameter is moved to the lower and upper bound of its range in Table 4.
SENS_RANGES = dict(capex_spec=(0.7 * 1087, 1.3 * 1087), om_frac=(0.03, 0.08), ng_price=(35, 60), ets_price=(60, 90),
                   discount_rate=(0.06, 0.10), lifetime=(10, 20), pomace_cost=(20, 40), rdf_avoided_cost=(35, 60),
                   elec_price=(110, 160), boiler_eff=(0.75, 0.85), rdf_fossil_C_frac=(0.45, 0.85), grid_EF=(0.12, 0.21))
SENS_K = ["capex_spec", "om_frac", "ng_price", "ets_price", "discount_rate", "lifetime", "pomace_cost", "rdf_avoided_cost",
          "elec_price", "boiler_eff", "rdf_fossil_C_frac", "grid_EF"]
b0 = TL["base"]; srows = []
for k in SENS_K:
    lo, hi = SENS_RANGES[k]
    rl, rh = tl3(*SIMS["base"], p=dict(P, **{k: lo})), tl3(*SIMS["base"], p=dict(P, **{k: hi}))
    srows.append(dict(param=k, **{f"{m}_lo": rl[m] for m in ["NPV", "LCOH", "GWP_net", "abat"]}, **{f"{m}_hi": rh[m] for m in ["NPV", "LCOH", "GWP_net", "abat"]}))
SENS23 = pd.DataFrame(srows).set_index("param"); SENS23.to_csv(DATA_OUT / "sensitivity_extended.csv")
for m in ["NPV", "LCOH", "abat"]:
    sp_ = (SENS23[f"{m}_hi"] - SENS23[f"{m}_lo"]).abs().sort_values(ascending=False)
    print(f"  23f. {m}: top drivers", {i: round(v, 1) for i, v in sp_.head(4).items()})

# ---- 23g. Extended Monte Carlo (60 kinetic runs x 33 draws = 1980; seed 2026) ----
rng23 = np.random.default_rng(2026)
def draw23(n):
    u = lambda a, b: rng23.uniform(a, b, n)
    return pd.DataFrame(dict(capex_spec=u(0.7 * 1087, 1.3 * 1087), om_frac=u(0.03, 0.08), ng_price=u(35, 60), ets_price=u(60, 90),
                             discount_rate=u(0.06, 0.10), lifetime=rng23.integers(10, 21, n), pomace_cost=u(20, 40),
                             rdf_avoided_cost=u(35, 60), elec_price=u(110, 160), boiler_eff=u(0.75, 0.85),
                             rdf_fossil_C_frac=u(0.45, 0.85), grid_EF=u(0.12, 0.21)))
mc = []
for lab, (T0, er0, x0) in PTS.items():
    for j in range(60):
        fh2_, fco_ = mc_factors(rng23); tau = rng23.uniform(0.7, 1.3) * 7200.0; fdol = np.exp(rng23.uniform(np.log(0.5), np.log(2.0)))
        kps_i = mc_kps(fh2_, fdol)
        s1, f1 = run_point(T0, er0, x0, opt=mc_opt(fco_), rc=_dcr(REACTOR_TERRADOS, tau_char_s=tau), kps=kps_i)
        for _, pr in draw23(33).iterrows():
            pp = dict(P, **pr.to_dict()); pp["lifetime"] = int(pp["lifetime"])
            r = tl3(s1, f1, p=pp)
            mc.append(dict(case=lab, NPV=r["NPV"], IRR=r["IRR"], LCOH=r["LCOH"], p_heat=r["p_heat"], GWP=r["GWP"], GWP_net=r["GWP_net"],
                           abat=r["abat"], i=pp["discount_rate"]))
    print(f"    MC {lab} done", flush=True)
MC23 = pd.DataFrame(mc); MC23.to_csv(DATA_OUT / "montecarlo_extended.csv", index=False)
for lab, g in MC23.groupby("case", sort=False):
    q = g[["NPV", "IRR", "LCOH", "GWP_net", "abat"]].quantile([0.05, 0.5, 0.95]).T.round(3)
    print(f"  23g. [{lab}] n = {len(g)}\n{q.to_string()}")
    print(f"     P(NPV>0) {100*(g.NPV > 0).mean():.1f} % | P(IRR>i) {100*(g.IRR / 100 > g.i).mean():.1f} % | "
          f"P(LCOH<p_heat) {100*(g.LCOH < g.p_heat).mean():.1f} % | P(GWP_net<NG) {100*(g.GWP_net < P['NG_reference']).mean():.1f} %")

# ---- 23h. Figures with the final numbering ----
for old, new in [("Fig1_framework", "Fig1_framework"), ("Fig2_parity", "Fig2_parity"), ("Fig3_composition_T", "Fig3_composition_T"),
                 ("Fig4_energy_performance", "Fig4_energy_performance"), ("Fig12_FABER_nonisothermal", "Fig5_FABER_nonisothermal"),
                 ("Fig5_T_ER_maps", "Fig7_T_ER_maps"), ("Fig7_Pareto", "Fig12_Pareto"), ("Fig11_energy_environment_opt", "Fig13_energy_environment_opt")]:
    for ext in ("png", "pdf"):
        src = FIG_DIR / f"{old}.{ext}"
        if src.exists(): shutil.copy(src, FIG23 / f"{new}.{ext}")
# Fig 6: freeboard profile
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
for sp, c in zip(["H2", "CO", "CO2", "CH4"], ["#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"]):
    axs[0].plot(PROF.z, PROF[sp], color=c, label=SPC[sp])
axs[0].set_xlabel("Height above the bed (m)"); axs[0].set_ylabel("vol%, dry"); axs[0].set_title("(a) Permanent gases", loc="left"); axs[0].legend(frameon=False, ncol=2)
axs[1].plot(PROF.z, PROF.tar_g_Nm3, "k-")
axs[1].set_xlabel("Height above the bed (m)"); axs[1].set_ylabel("Tar (g Nm$^{-3}$ dry gas)"); axs[1].set_title("(b) Tar", loc="left")
fig.tight_layout(); save23(fig, "Fig6_freeboard_profile")
# Fig 8: RDF fraction (4 panels incl. GWP)
rr = []
for lab, (T0, er0) in [("Base (700 \u00b0C, ER 0.26)", (700, er_b)), (f"Compromise ({T_COMP:.0f} \u00b0C, ER {ER_COMP:.2f})", (T_COMP, ER_COMP))]:
    for x in [0.10, 0.20, 0.30, 0.40, 0.50]:
        s1, f1 = run_point(T0, er0, x); r = tl3(s1, f1)
        rr.append(dict(case=lab, x=x, CGE=s1["CGE"], LHV=s1["LHV"], HHV=hhv_mix(s1["y"]), tar=s1["tar_g_kg"], GWP=r["GWP"], GWP_net=r["GWP_net"]))
RDF23 = pd.DataFrame(rr); RDF23.to_csv(DATA_OUT / "rdf_fraction_effect_tea.csv", index=False)
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3))
for (lab, g), st in zip(RDF23.groupby("case", sort=False), ["-o", "-s"]):
    axs[0].plot(100 * g.x, g.CGE, st, ms=3, label=lab.split(" (")[0]); axs[1].plot(100 * g.x, g.LHV, st, ms=3); axs[1].plot(100 * g.x, g.HHV, st.replace("-", "--"), ms=3, mfc="none")
    axs[2].plot(100 * g.x, g.tar, st, ms=3); axs[3].plot(100 * g.x, g.GWP, st, ms=3)
for ax, t_ in zip(axs, ["(a) CGE (%)", "(b) LHV (solid), HHV (dashed)", "(c) Tar (g kg$^{-1}$)", "(d) GWP$_{100}$ (kg MJ$^{-1}$)"]):
    ax.set_title(t_, loc="left", fontsize=7); ax.set_xlabel("RDF (wt%)")
axs[0].legend(frameon=False, fontsize=6); fig.tight_layout(); save23(fig, "Fig8_RDF_fraction")
# Fig 9: economics (LCOH breakdown + NPV vs capacity)
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
cols = [("capex", "Annualised CAPEX", "#1f3864"), ("om", "O&M", "#5b9bd5"), ("pomace", "Olive pomace", "#a9d18e"), ("elec", "Electricity", "#ffc000"),
        ("co2", "CO$_2$ allowances", "#c00000"), ("rdf", "Avoided landfill (credit)", "#70ad47")]
labs = ["Base", "Compromise"]
for j, k in enumerate(["base", "comp"]):
    pos, neg = 0.0, 0.0
    for key, lab, c in cols:
        v = TL[k]["comp"][key]
        if v >= 0: axs[0].bar(j, v, bottom=pos, color=c, width=0.5, label=lab if j == 0 else None); pos += v
        else: axs[0].bar(j, v, bottom=neg, color=c, width=0.5, label=lab if j == 0 else None); neg += v
    axs[0].plot(j, TL[k]["LCOH"], "kD", ms=5)
axs[0].axhline(TL["base"]["p_heat"], color="k", ls="--", lw=0.8); axs[0].text(2.2, TL["base"]["p_heat"] + 1.5, "NG heat", fontsize=6)
axs[0].axhline(0, color="k", lw=0.5); axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(labs); axs[0].set_xlim(-0.5, 3.0)
axs[0].set_ylabel("\u20ac MWh$^{-1}$ of useful heat"); axs[0].set_title("(a) LCOH breakdown (\u25c6 = LCOH)", loc="left"); axs[0].legend(frameon=False, fontsize=5.5, loc="center right")
for lab, st in [("base", "-o"), ("opt", "-s")]:
    g = SCALE16[SCALE16.case == lab]; axs[1].plot(g.P_kWth, g.NPV_kEUR, st, ms=3, label="Base" if lab == "base" else "Max-NPV point")
axs[1].axhline(0, color="k", lw=0.6); axs[1].set_xscale("log"); axs[1].set_xlabel("Capacity (kWth)"); axs[1].set_ylabel("NPV (k\u20ac)")
axs[1].set_title("(b) NPV vs. capacity", loc="left"); axs[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); save23(fig, "Fig9_economics")
# Fig 10: LCA contributions and landfill credit
fig, ax = plt.subplots(figsize=(W1 * 1.4, 2.7))
parts = [("fossil", "Fossil CO$_2$ of syngas", "#6c757d"), ("ch4", "CH$_4$ slip", "#e07a5f"), ("elec", "Electricity", "#3d405b")]
for j, k in enumerate(["base", "comp"]):
    pos = 0.0
    for key, lab, c in parts:
        v = TL[k]["gwp_parts"][key]; ax.bar(j * 2, v, bottom=pos, color=c, width=0.6, label=lab if j == 0 else None); pos += v
    ax.bar(j * 2 + 0.7, TL[k]["gwp_parts"]["landfill"], bottom=pos, color="#2ca02c", width=0.6, label="Avoided landfill" if j == 0 else None)
    ax.plot(j * 2 + 0.7, TL[k]["GWP_net"], "kD", ms=5)
ax.bar(4.2, P["NG_reference"], color="#b5b5b5", width=0.6, label="Natural gas")
ax.set_xticks([0, 0.7, 2, 2.7, 4.2]); ax.set_xticklabels(["Base\nattr.", "Base\nnet", "Comp.\nattr.", "Comp.\nnet", "NG"], fontsize=6)
ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.legend(frameon=False, fontsize=5.5, ncol=2)
fig.tight_layout(); save23(fig, "Fig10_LCA_contributions")
# Fig 11: integrated LCOH-GWP map + abatement cost
T22_ = pd.read_csv(DATA_OUT / "tea_lca_integrated.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
sc = axs[0].scatter(MAP23.GWP_net, MAP23.LCOH, c=MAP23["T"], s=12 + 40 * MAP23.x_RDF, cmap="plasma", edgecolors="k", lw=0.2)
fig.colorbar(sc, ax=axs[0], label="T (\u00b0C)")
axs[0].axhline(TL["base"]["p_heat"], color="k", ls="--", lw=0.7); axs[0].axvline(P["NG_reference"], color="k", ls=":", lw=0.7)
for k, mk in [("base", "o"), ("comp", "*")]:
    axs[0].plot(TL[k]["GWP_net"], TL[k]["LCOH"], mk, color="#2ca02c", ms=9, mfc="none", mew=1.3)
axs[0].set_xlabel("Net GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); axs[0].set_ylabel("LCOH (\u20ac MWh$^{-1}$)")
axs[0].set_title("(a) 125 operating points (size = RDF share)", loc="left", fontsize=7)
o_ = T22_[T22_.point.str.startswith("Optimum")]
for (scn, Pd), c, a in [(("2025", 209.8), "#1f77b4", 0.45), (("2025", 1000.0), "#1f77b4", 0.85), (("2030", 209.8), "#d62728", 0.45), (("2030", 1000.0), "#d62728", 0.85)]:
    g = o_[(o_.scenario == scn) & (o_.P_kWth == Pd)]
    axs[1].bar(f"{scn}\n{'0.21' if Pd < 300 else '1.0'} MW", g.abat_net, color=c, alpha=a)
axs[1].axhline(0, color="k", lw=0.5); axs[1].set_ylabel("Abatement cost (\u20ac t$^{-1}$ CO$_2$-eq)"); axs[1].tick_params(axis="x", labelsize=6)
axs[1].set_title("(b) CO$_2$ abatement cost, compromise", loc="left", fontsize=7)
fig.tight_layout(); save23(fig, "Fig11_integrated")
# Fig 14: sensitivity 2x2
lbl = {"capex_spec": "Specific CAPEX", "om_frac": "O&M fraction", "ng_price": "Natural-gas price", "ets_price": "EU ETS price",
       "discount_rate": "Discount rate", "lifetime": "Lifetime", "pomace_cost": "Pomace cost", "rdf_avoided_cost": "Avoided landfill fee",
       "elec_price": "Electricity price", "boiler_eff": "Boiler efficiency", "rdf_fossil_C_frac": "RDF fossil C fraction", "grid_EF": "Grid emission factor"}
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2))
for ax, (m, ttl, bv) in zip(axs.flat, [("NPV", "(a) NPV (k\u20ac)", b0["NPV"]), ("LCOH", "(b) LCOH (\u20ac MWh$^{-1}$)", b0["LCOH"]),
                                       ("GWP_net", "(c) Net GWP$_{100}$ (kg MJ$^{-1}$)", b0["GWP_net"]), ("abat", "(d) Abatement cost (\u20ac t$^{-1}$)", b0["abat"])]):
    S = SENS23.copy(); S["span"] = (S[f"{m}_hi"] - S[f"{m}_lo"]).abs(); S = S[S.span > 1e-9].sort_values("span")
    y = np.arange(len(S))
    ax.barh(y, S[f"{m}_lo"] - bv, left=bv, color="#1f77b4", height=0.6, label="lower bound (Table 4)"); ax.barh(y, S[f"{m}_hi"] - bv, left=bv, color="#ff7f0e", height=0.6, label="upper bound (Table 4)")
    ax.axvline(bv, color="k", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([lbl[i] for i in S.index], fontsize=6); ax.set_title(ttl, loc="left")
axs[0, 0].legend(frameon=False, fontsize=6, loc="lower right"); fig.tight_layout(); save23(fig, "Fig14_sensitivity")
# Fig 15: Monte Carlo
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
for lab, c in [("base", "#1f77b4"), ("comp", "#d62728")]:
    g = MC23[MC23.case == lab]; nm = "Base" if lab == "base" else "Compromise"
    axs[0].hist(g.NPV, bins=40, alpha=0.55, color=c, label=nm); axs[1].hist(g.LCOH, bins=40, alpha=0.55, color=c); axs[2].hist(g.GWP_net, bins=40, alpha=0.55, color=c)
axs[0].axvline(0, color="k", lw=0.6); axs[0].set_xlabel("NPV at 209.8 kWth (k\u20ac)"); axs[0].set_ylabel("Frequency"); axs[0].legend(frameon=False, fontsize=6)
axs[1].set_xlabel("LCOH (\u20ac MWh$^{-1}$)"); axs[2].axvline(P["NG_reference"], color="k", ls="--", lw=0.7); axs[2].set_xlabel("Net GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)")
for ax, t_ in zip(axs, ["(a)", "(b)", "(c)"]): ax.set_title(t_, loc="left")
fig.tight_layout(); save23(fig, "Fig15_montecarlo")
print("  23h. Final figure set:", sorted(p.name for p in FIG23.glob("*.png")))

# =============================================================================
# 24. RESULTS SET: validation table with model/experimental means, sensitivity, Monte Carlo and LCA figures
#     (auxiliary figures; the figures of the article are drawn in section 25)
# =============================================================================
FIG24 = AUX_OUT / "figures_aux_optimization"
def save24(fig, name):
    fig.savefig(FIG24 / f"{name}.png", bbox_inches="tight"); fig.savefig(FIG24 / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
print("\n" + "=" * 78); print("24. FINAL RESULTS SET"); print("=" * 78, flush=True)
# ---- 24a. Table 4: mean experimental / model values, RMSE 
DS_T4 = [("Terrados 2026", "Terrados (development)"), ("Calado 2023", "Calado"),
         ("FABER (Parrillo 2023)", "FABER"), ("Pio 2020", "Pio")]
rows = []
for key, lab in DS_T4:
    df = DATASETS[key]; r = dict(dataset=lab, n=len(df))
    for sp in ["H2", "CO", "CO2", "CH4"]:
        d = df[[f"{sp}_exp", f"{sp}_mod"]].dropna()
        e = d[f"{sp}_mod"] - d[f"{sp}_exp"]
        r.update({f"{sp}_exp": d[f"{sp}_exp"].mean(), f"{sp}_mod": d[f"{sp}_mod"].mean(),
                  f"{sp}_RMSE": np.sqrt((e ** 2).mean()), f"{sp}_n": len(d)})
    rows.append(r)
T4 = pd.DataFrame(rows)
avg = {f"{sp}_{m}": T4[f"{sp}_{m}"].mean() for sp in ["H2", "CO", "CO2", "CH4"] for m in ["RMSE"]}
print(T4.round(2).to_string(index=False)); print("  average:", {k: round(v, 2) for k, v in avg.items()})
T4.to_csv(DATA_OUT / "table4_validation.csv", index=False)
# ---- 24b. Auxiliary figure: Terrados (reference) composition vs temperature, grouped bars ----
rowsF = []
for T, r in TABLE4.iterrows():
    s_ = simulate(BLEND_TERRADOS, REACTOR_TERRADOS, OperatingPoint(T, r.P_gauge_kPa, r.feed_kg_h, air_n2_consistent(r, BLEND_TERRADOS)), KPS, OPT_H1)
    y = s_["y"]
    rowsF.append(dict(T=T, **{f"{k}_mod": y.get(k, 0.0) for k in ["H2", "CO", "CO2", "CH4", "N2"]},
                      C2_mod=y.get("C2H4", 0.0) + y.get("C2H6", 0.0) + y.get("C2H2", 0.0),
                      **{f"{k}_exp": r[k] for k in ["H2", "CO", "CO2", "CH4", "N2"]},
                      C2_exp=sum(r[k] for k in ["C2H4", "C2H6", "C2H2"] if k in r)))
F2 = pd.DataFrame(rowsF).set_index("T"); F2.to_csv(DATA_OUT / "fig2_terrados_bars.csv")
fig, axs = plt.subplots(2, 3, figsize=(W2, 4.6))
x = np.arange(len(F2)); w = 0.38
for ax, (sp, ttl) in zip(axs.flat, [("H2", "(a) H$_2$"), ("CO", "(b) CO"), ("CO2", "(c) CO$_2$"), ("CH4", "(d) CH$_4$"), ("N2", "(e) N$_2$"), ("C2", "(f) C$_2$H$_n$")]):
    ex_, mo_ = F2[f"{sp}_exp"].values, F2[f"{sp}_mod"].values
    ax.bar(x - w / 2, ex_, w, color="0.25", label="Experimental"); ax.bar(x + w / 2, mo_, w, color="#d62728", label="Model")
    for i in range(len(x)):
        dv = mo_[i] - ex_[i]; ax.text(x[i], max(ex_[i], mo_[i]) * 1.03 + 0.2, ("0.0" if abs(dv) < 0.05 else f"{dv:+.1f}"), ha="center", va="bottom", fontsize=5.5)
    rmse = np.sqrt(np.mean((mo_ - ex_) ** 2))
    ax.set_title(f"{ttl}  (RMSE {rmse:.2f} pp)", loc="left", fontsize=7); ax.set_xticks(x); ax.set_xticklabels([str(int(t)) for t in F2.index])
    ax.set_xlabel("T (\u00b0C)"); ax.set_ylabel("vol%, dry"); ax.set_ylim(0, max(ex_.max(), mo_.max()) * 1.25 + 0.5); ax.grid(axis="y", lw=0.3, alpha=0.6)
axs[0, 0].legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
fig.tight_layout(); save24(fig, "Fig2_validation_bars")
# ---- 24c. TEA-LCA at the compromise; sensitivity at the compromise ----
TLc = {k: tl3(*run_point(*v)) for k, v in {"base": (700, er_b, 0.30), "comp": (T_COMP, ER_COMP, X_COMP)}.items()}
SIMc = run_point(T_COMP, ER_COMP, X_COMP); b0 = TLc["comp"]; srows = []
for k in SENS_K:
    lo, hi = SENS_RANGES[k]
    rl, rh = tl3(*SIMc, p=dict(P, **{k: lo})), tl3(*SIMc, p=dict(P, **{k: hi}))
    srows.append(dict(param=k, **{f"{m}_lo": rl[m] for m in ["NPV", "LCOH", "GWP_net", "abat"]}, **{f"{m}_hi": rh[m] for m in ["NPV", "LCOH", "GWP_net", "abat"]}))
SENSc = pd.DataFrame(srows).set_index("param"); SENSc.to_csv(DATA_OUT / "sensitivity_compromise.csv")
for m in ["NPV", "LCOH", "GWP_net", "abat"]:
    sp_ = (SENSc[f"{m}_hi"] - SENSc[f"{m}_lo"]).abs().sort_values(ascending=False)
    print(f"  24c. {m} (base {b0[m]:.3f}): top drivers", {i: round(v, 3) for i, v in sp_.head(4).items()})
ng0 = SENSc.loc["ng_price"]; slope = (ng0.NPV_hi - ng0.NPV_lo) / (0.6 * P["ng_price"])
print(f"  NPV = 0 at natural-gas price ~ {P['ng_price'] - b0['NPV'] / slope:.1f} EUR/MWh")
# ---- 24d. Figures ----
FIGV12 = AUX_OUT / "figures_aux_tea"
for old, new in [("Fig1_framework", "Fig1_framework"), ("Fig5_FABER_nonisothermal", "Fig3_FABER_nonisothermal"), ("Fig7_T_ER_maps", "Fig4_T_ER_maps"),
                 ("Fig9_economics", "Fig6_economics"), ("Fig12_Pareto", "Fig9_Pareto"), ("Fig13_energy_environment_opt", "Fig10_energy_environment_opt")]:
    for ext in ("png", "pdf"):
        if (FIGV12 / f"{old}.{ext}").exists(): shutil.copy(FIGV12 / f"{old}.{ext}", FIG24 / f"{new}.{ext}")
# Fig 5: RDF fraction, compromise only
R5 = pd.read_csv(DATA_OUT / "rdf_fraction_effect_tea.csv"); R5 = R5[R5.case.str.startswith("Compromise")]
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.3))
axs[0].plot(100 * R5.x, R5.CGE, "-s", ms=3, color="#d62728"); axs[1].plot(100 * R5.x, R5.LHV, "-s", ms=3, color="#d62728", label="LHV")
axs[1].plot(100 * R5.x, R5.HHV, "--s", ms=3, mfc="none", color="#d62728", label="HHV"); axs[2].plot(100 * R5.x, R5.tar, "-s", ms=3, color="#d62728")
axs[3].plot(100 * R5.x, R5.GWP, "-s", ms=3, color="#d62728", label="Attributional"); axs[3].plot(100 * R5.x, R5.GWP_net, "--o", ms=3, mfc="none", color="#2ca02c", label="Net")
for ax, t_ in zip(axs, ["(a) CGE (%)", "(b) Heating value (MJ Nm$^{-3}$)", "(c) Tar (g kg$^{-1}$)", "(d) GWP$_{100}$ (kg MJ$^{-1}$)"]):
    ax.set_title(t_, loc="left", fontsize=7); ax.set_xlabel("RDF (wt%)"); ax.grid(lw=0.3, alpha=0.6)
axs[1].legend(frameon=False, fontsize=6); axs[3].legend(frameon=False, fontsize=6); fig.tight_layout(); save24(fig, "Fig5_RDF_fraction")
# Fig 7: LCA contributions with grid
fig, ax = plt.subplots(figsize=(W1 * 1.4, 2.7))
parts = [("fossil", "Fossil CO$_2$ of syngas", "#6c757d"), ("ch4", "CH$_4$ slip", "#e07a5f"), ("elec", "Electricity", "#3d405b")]
for j, k in enumerate(["base", "comp"]):
    pos = 0.0
    for key, lab, c in parts:
        v = TLc[k]["gwp_parts"][key]; ax.bar(j * 2, v, bottom=pos, color=c, width=0.6, label=lab if j == 0 else None, zorder=3); pos += v
    ax.bar(j * 2 + 0.7, TLc[k]["gwp_parts"]["landfill"], bottom=pos, color="#2ca02c", width=0.6, label="Avoided landfill" if j == 0 else None, zorder=3)
    ax.plot(j * 2 + 0.7, TLc[k]["GWP_net"], "kD", ms=5, zorder=4)
ax.bar(4.2, P["NG_reference"], color="#b5b5b5", width=0.6, label="Natural gas", zorder=3)
ax.set_xticks([0, 0.7, 2, 2.7, 4.2]); ax.set_xticklabels(["Base\nattr.", "Base\nnet", "Comp.\nattr.", "Comp.\nnet", "NG"], fontsize=6)
ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.grid(axis="y", lw=0.4, alpha=0.7, zorder=0)
ax.legend(frameon=False, fontsize=5.5, ncol=2); fig.tight_layout(); save24(fig, "Fig7_LCA_contributions")
# Fig 8: integrated (abatement panel fixed: scenario read as text)
T22_ = pd.read_csv(DATA_OUT / "tea_lca_integrated.csv", dtype={"scenario": str}); MAP23 = pd.read_csv(DATA_OUT / "grid_LCOH_GWP.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9))
sc = axs[0].scatter(MAP23.GWP_net, MAP23.LCOH, c=MAP23["T"], s=12 + 40 * MAP23.x_RDF, cmap="plasma", edgecolors="k", lw=0.2, zorder=3)
fig.colorbar(sc, ax=axs[0], label="T (\u00b0C)")
axs[0].axhline(TLc["comp"]["p_heat"], color="k", ls="--", lw=0.7); axs[0].axvline(P["NG_reference"], color="k", ls=":", lw=0.7)
for k, mk in [("base", "o"), ("comp", "*")]:
    axs[0].plot(TLc[k]["GWP_net"], TLc[k]["LCOH"], mk, color="#2ca02c", ms=9, mfc="none", mew=1.3, zorder=4)
axs[0].set_xlabel("Net GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); axs[0].set_ylabel("LCOH (\u20ac MWh$^{-1}$)"); axs[0].grid(lw=0.3, alpha=0.6, zorder=0)
axs[0].set_title("(a) 125 operating points (size = RDF share)", loc="left", fontsize=7)
o_ = T22_[T22_.point.str.startswith("Optimum")]; labs, vals, cols, alph = [], [], [], []
for scn, c in [("2025", "#1f77b4"), ("2030", "#d62728")]:
    for Pd in (209.8, 1000.0):
        g = o_[(o_.scenario == scn) & np.isclose(o_.P_kWth, Pd)]
        labs.append(f"{scn}\n{'0.21' if Pd < 300 else '1.0'} MWth"); vals.append(float(g.abat_net.iloc[0])); cols.append(c); alph.append(0.45 if Pd < 300 else 0.85)
for i, (l, v, c, a) in enumerate(zip(labs, vals, cols, alph)):
    axs[1].bar(i, v, color=c, alpha=a, zorder=3); axs[1].text(i, v - 8, f"{v:.0f}", ha="center", va="top", fontsize=6)
axs[1].set_xticks(range(len(labs))); axs[1].set_xticklabels(labs, fontsize=6); axs[1].axhline(0, color="k", lw=0.5)
axs[1].set_ylim(min(vals) * 1.2, 20); axs[1].set_ylabel("Abatement cost (\u20ac t$^{-1}$ CO$_2$-eq)"); axs[1].grid(axis="y", lw=0.3, alpha=0.6, zorder=0)
axs[1].set_title("(b) CO$_2$ abatement cost, compromise", loc="left", fontsize=7)
fig.tight_layout(); save24(fig, "Fig8_integrated"); print("  Fig 8b values:", dict(zip([l.replace(chr(10), ' ') for l in labs], [round(v, 1) for v in vals])))
# Fig 11: sensitivity at the compromise
lbl = {"capex_spec": "Specific CAPEX", "om_frac": "O&M fraction", "ng_price": "Natural-gas price", "ets_price": "EU ETS price",
       "discount_rate": "Discount rate", "lifetime": "Lifetime", "pomace_cost": "Pomace cost", "rdf_avoided_cost": "Avoided landfill fee",
       "elec_price": "Electricity price", "boiler_eff": "Boiler efficiency", "rdf_fossil_C_frac": "RDF fossil C fraction", "grid_EF": "Grid emission factor"}
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.2))
for ax, (m, ttl) in zip(axs.flat, [("NPV", "(a) NPV (k\u20ac)"), ("LCOH", "(b) LCOH (\u20ac MWh$^{-1}$)"), ("GWP_net", "(c) Net GWP$_{100}$ (kg MJ$^{-1}$)"), ("abat", "(d) Abatement cost (\u20ac t$^{-1}$)")]):
    bv = b0[m]; S = SENSc.copy(); S["span"] = (S[f"{m}_hi"] - S[f"{m}_lo"]).abs(); S = S[S.span > 1e-9].sort_values("span"); yy = np.arange(len(S))
    ax.barh(yy, S[f"{m}_lo"] - bv, left=bv, color="#1f77b4", height=0.6, label="lower bound (Table 4)", zorder=3); ax.barh(yy, S[f"{m}_hi"] - bv, left=bv, color="#ff7f0e", height=0.6, label="upper bound (Table 4)", zorder=3)
    ax.axvline(bv, color="k", lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([lbl[i] for i in S.index], fontsize=6); ax.set_title(ttl, loc="left"); ax.grid(axis="x", lw=0.3, alpha=0.6, zorder=0)
axs[0, 0].legend(frameon=False, fontsize=6, loc="lower right"); fig.tight_layout(); save24(fig, "Fig11_sensitivity")
# Fig 12: Monte Carlo, compromise only
MCc = pd.read_csv(DATA_OUT / "montecarlo_extended.csv"); MCc = MCc[MCc.case == "comp"]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
axs[0].hist(MCc.NPV, bins=40, color="#d62728", alpha=0.7, zorder=3); axs[1].hist(MCc.LCOH, bins=40, color="#d62728", alpha=0.7, zorder=3); axs[2].hist(MCc.GWP_net, bins=40, color="#d62728", alpha=0.7, zorder=3)
axs[0].axvline(0, color="k", lw=0.6); axs[1].axvline(MCc.p_heat.median(), color="k", ls="--", lw=0.7); axs[2].axvline(P["NG_reference"], color="k", ls="--", lw=0.7)
axs[0].set_xlabel("NPV at 209.8 kWth (k\u20ac)"); axs[0].set_ylabel("Frequency"); axs[1].set_xlabel("LCOH (\u20ac MWh$^{-1}$)"); axs[2].set_xlabel("Net GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)")
for ax, t_ in zip(axs, ["(a)", "(b)", "(c)"]): ax.set_title(t_, loc="left"); ax.grid(lw=0.3, alpha=0.6, zorder=0)
fig.tight_layout(); save24(fig, "Fig12_montecarlo")
print("  24d. Figures:", sorted(p.name for p in FIG24.glob("*.png")))


# =============================================================================
# 24b. COMPLEMENTARY ANALYSES: equilibrium reference, N2-free metrics, carbon/tar/char, TEA scenarios,
#      OAT with the Table 4 ranges, leave-one-dataset-out choice of the release zone, Monte Carlo check
# =============================================================================
import json as _json
_R = {}
print("\n" + "=" * 78); print("24b. COMPLEMENTARY ANALYSES"); print("=" * 78, flush=True)

# ---- 24b-a. Thermodynamic-equilibrium reference (baseline without kinetics) ------------------------------------------
#   Gas-phase equilibrium of CO, CO2, H2, H2O, CH4 with inert N2 at the bed temperature and pressure; all fuel C, H, O
#   to the gas (no char, no tar). Equations: element balances C, H, O; K_WGS and K for CO + 3 H2 = CH4 + H2O from
#   NASA-7 polynomials (NASA_EB). Solid carbon is not considered (all fuel carbon goes to the gas).
from scipy.optimize import least_squares as _lsq
def _s_molar(sp, T):
    tmid, hi, lo = NASA_EB[sp]; a = hi if T > tmid else lo
    return R_GAS * (a[0] * np.log(T) + a[1] * T + a[2] * T**2 / 2 + a[3] * T**3 / 3 + a[4] * T**4 / 4 + a[6])
_g = lambda sp, T: h_molar(sp, T) - T * _s_molar(sp, T)
def equilibrium_gas(sim, P_abs):
    fe = sim["fe"]; op = sim["op"]; T = op.T_C + 273.15
    air = op.air_Nm3_h * MOL_PER_NM3 / 3600.0; h2o_m = fe["moist_kg_s"] * 1000.0 / 18.015
    nC = fe["el"]["C"]; nH = fe["el"]["H"] + 2 * h2o_m; nO = fe["el"]["O"] + h2o_m + 2 * AIR_O2 * air
    nN2 = (1 - AIR_O2) * air + fe["el"]["N"] / 2
    K1 = np.exp(-(_g("CO2", T) + _g("H2", T) - _g("CO", T) - _g("H2O", T)) / (R_GAS * T))
    K2 = np.exp(-(_g("CH4", T) + _g("H2O", T) - _g("CO", T) - 3 * _g("H2", T)) / (R_GAS * T))
    p = P_abs / 101325.0
    def res(x):
        co, co2, h2, h2o, ch4 = np.exp(x); nt = co + co2 + h2 + h2o + ch4 + nN2
        return [(co + co2 + ch4) / nC - 1, (2 * h2 + 2 * h2o + 4 * ch4) / nH - 1, (co + 2 * co2 + h2o) / nO - 1,
                np.log(co2 * h2 / (co * h2o)) - np.log(K1),
                np.log(ch4 * h2o * nt**2 / (co * h2**3 * p**2)) - np.log(K2)]
    x0 = np.log(np.array([0.5 * nC, 0.4 * nC, 0.4 * nH / 2, 0.4 * nH / 2, 0.05 * nC]) + 1e-12)
    s = _lsq(res, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000)
    co, co2, h2, h2o, ch4 = np.exp(s.x); dry = co + co2 + h2 + ch4 + nN2
    y = dict(CO=100 * co / dry, CO2=100 * co2 / dry, H2=100 * h2 / dry, CH4=100 * ch4 / dry, N2=100 * nN2 / dry)
    return y, float(np.max(np.abs(res(s.x))))

# ---- collect every validation point once: experimental, formulation B, equilibrium ------------------------------
_pts = []   # (dataset, role, fs, rc, op, exp dict, kps, opt)
for p in pts_B: _pts.append(("Terrados-Cristos (development)", "dev", BLEND_TERRADOS, REACTOR_TERRADOS, OperatingPoint(p["T"], p["P"], p["feed"], p["air"]), p["exp"]))
for p in pts_C: _pts.append(("Carmo-Calado", "val", FEED_CALADO, REACTOR_CALADO, OperatingPoint(p["T"], p["P"], p["feed"], p["air"]), p["exp"]))
for p in PIO_POINTS: _pts.append(("Pio", "val", p["feed_cfg"], REACTOR_PIO, OperatingPoint(p["T"], p["P"], p["feed"], p["air"], fb_profile=pio_fb_profile(p["T"])), p["exp"]))
for _, r in TABLE_FABER.iterrows(): _pts.append(("Parrillo 2023 (plastics-eucalyptus)", "val", faber_feed(r.PL, r.BM), REACTOR_FABER,
                                                OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h, T_fb_out_C=r.T_fb_out_C),
                                                {k: r[k] for k in ["H2", "CO", "CO2", "CH4", "N2"] if k in r}))
for _, r in P21_BLIND[P21_BLIND.blind].iterrows(): _pts.append(("Parrillo 2021 (eucalyptus)", "val", faber_feed(0.0, 1.0), REACTOR_FABER,
                                                OperatingPoint(r.T_bed_C, P_GAUGE_FABER, r.feed_kg_h, r.air_Nm3_h), {k: r[k] for k in ["H2", "CO", "CO2", "CH4", "N2"] if k in r}))
for ds_, (fs_, rc_, pp_) in BLIND2.items():
    if "not part" in ds_: continue
    for p_ in pp_: _pts.append((ds_.split(",")[0] + (" (blind, olive pomace)" if "Campoy" in ds_ else " (blind, SRF)"), "blind", fs_, rc_,
                                OperatingPoint(float(p_["T"]), 0.0, p_["feed"], p_["air"]), {k: p_[k] for k in ["H2", "CO", "CO2", "CH4"]}))
SP4 = ["H2", "CO", "CO2", "CH4"]
_OPT_B = replace(OPT_PRED, bed_model="two_zone", tz_volatiles_to="emulsion")
rows24b = []
for ds_, role_, fs_, rc_, op_, ex_ in _pts:
    s_ = simulate(fs_, rc_, op_, KPS_PRED, _OPT_B)
    ye, resid_ = equilibrium_gas(s_, P_ATM + op_.P_gauge_kPa * 1000.0)
    rec = dict(dataset=ds_, role=role_, T=op_.T_C, ER=s_["ER"], eq_resid=resid_)
    for k in SP4 + ["N2"]:
        rec[f"{k}_exp"] = ex_.get(k, np.nan); rec[f"{k}_B"] = s_["y"].get(k, np.nan); rec[f"{k}_eq"] = ye.get(k, np.nan)
    rows24b.append(rec)
PT27 = pd.DataFrame(rows24b); PT27.to_csv(DATA_OUT / "analysis_points_B_equilibrium.csv", index=False)
def _metr(d, suf):
    D = np.column_stack([d[f"{k}_{suf}"] - d[f"{k}_exp"] for k in SP4]); return float(np.nanmean(np.abs(D))), np.sqrt(np.nanmean(D**2, 0)), np.nanmean(D, 0)
def _n2free(d, suf):
    a = d[[f"{k}_{suf}" for k in SP4]].values; a = 100 * a / a.sum(1, keepdims=True); return a
out24b = []
for ds_, d in PT27.groupby("dataset", sort=False):
    mB, rB, bB = _metr(d, "B"); mE, rE, bE = _metr(d, "eq")
    De = _n2free(d, "exp"); DB = _n2free(d, "B") - De
    h2co_e, h2co_B = (d.H2_exp / d.CO_exp), (d.H2_B / d.CO_B)
    out24b.append(dict(dataset=ds_, n=len(d), MAE_B=mB, MAE_eq=mE, maxRMSE_B=rB.max(), maxRMSE_eq=rE.max(),
                      bias_CO_B=bB[1], bias_CO2_B=bB[2], bias_CO_eq=bE[1], bias_CH4_eq=bE[3],
                      MAE_B_N2free=float(np.abs(DB).mean()), maxRMSE_B_N2free=float(np.sqrt((DB**2).mean(0)).max()),
                      bias_CO_N2free=float(DB[:, 1].mean()),
                      H2CO_exp=float(h2co_e.mean()), H2CO_B=float(h2co_B.mean()), eq_max_resid=float(d.eq_resid.max())))
T27A = pd.DataFrame(out24b).set_index("dataset"); T27A.to_csv(DATA_OUT / "analysis_metrics_B_equilibrium.csv")
print("\n  24b-a. Formulation B vs gas-phase equilibrium; N2-free metrics (four species normalised to 100 %); H2/CO")
print(T27A.round(2).to_string())
_R["24b-a"] = T27A.round(3).reset_index().to_dict("records")

# ---- 24b-b. Reference case: published air flow instead of the nitrogen-balance air; gauge pressure in mbar -----------
def _mae_terr(pts, pfac=1.0):
    D, ers = [], []
    for p in pts:
        s_ = simulate(BLEND_TERRADOS, REACTOR_TERRADOS, OperatingPoint(p["T"], p["P"] * pfac, p["feed"], p["air"]), KPS, OPT_H1)
        D.append([s_["y"][k] - p["exp"][k] for k in SP4]); ers.append(s_["ER"])
    D = np.array(D); return dict(MAE=float(np.abs(D).mean()), RMSE=np.sqrt((D**2).mean(0)).round(2).tolist(), bias=D.mean(0).round(2).tolist(), ER=[round(float(e), 3) for e in ers])
_R["24b-b"] = {"reconstructed air (adopted)": _mae_terr(pts_B), "published air 88-91 Nm3/h": _mae_terr(pts_A),
             "pressure read as mbar (gauge/10)": _mae_terr(pts_B, 0.1), "atmospheric (gauge 0)": _mae_terr(pts_B, 0.0)}
print("\n  24b-b. Reference case (H2, CO, CO2, CH4):"); [print(f"    {k}: {v}") for k, v in _R["24b-b"].items()]

# ---- 24b-c. Carbon distribution, tar and char of the reference case vs measured -------------------------------------
rows_c = []
for p in pts_B:
    s_ = simulate(BLEND_TERRADOS, REACTOR_TERRADOS, OperatingPoint(p["T"], p["P"], p["feed"], p["air"]), KPS, OPT_H1)
    F = s_["F_out"]; nC = s_["fe"]["el"]["C"]
    C_tar = sum(F[IDX[t]] * ATOM[IDX[t], 0] for t in TAR_SPECIES)
    rows_c.append(dict(T=p["T"], C_gas_pct=s_["C_gas_pct"], C_tar_pct=100 * C_tar / nC, C_char_pct=100 - s_["C_gas_pct"] - 100 * C_tar / nC,
                       tar_g_kg=s_["tar_g_kg"], tar_nobenzene_g_kg=s_["tar_nb_g_kg"], tar_meas=TABLE4.loc[p["T"], "tar_g_kg"],
                       char_g_kg=s_["char_g_kg"], char_meas=TABLE4.loc[p["T"], "char_g_kg"], CGE=s_["CGE"], CGE_meas=TABLE4.loc[p["T"], "CGE"],
                       **{f"tar_{t}_g_kg": F[IDX[t]] * MW[IDX[t]] / s_["fe"]["feed_kg_s"] for t in TAR_SPECIES}))
T27C = pd.DataFrame(rows_c).set_index("T"); T27C.to_csv(DATA_OUT / "analysis_carbon_tar_char.csv")
print("\n  24b-c. Reference case: carbon distribution (% of feed C), tar and char (g/kg fuel), CGE"); print(T27C.round(2).to_string())
_R["24b-c"] = T27C.round(3).reset_index().to_dict("records")

# ---- 24b-d. Techno-economic scenarios at the base case (tl3, Table 4 parameters) -------------------------------------
sb, fb = SIMS["base"]; b0 = tl3(sb, fb)
_ann = lambda i, n: (1 - (1 + i) ** -n) / i
A0 = _ann(P["discount_rate"], P["lifetime"]); CF0 = (b0["NPV"] * 1e3 + b0["CAPEX"]) / A0
ash_t = sb["fe"]["ash_kg_s"] / sb["fe"]["feed_kg_s"] * b0["feed_t"]
tar_t_model = sb["tar_g_kg"] * b0["feed_t"] / 1000.0; tar_t_meas = TABLE4.loc[700, "tar_g_kg"] * b0["feed_t"] / 1000.0
sc = {}
def _sc(name, **kw): r = tl3(sb, fb, p=dict(P, **kw)); sc[name] = dict(NPV=r["NPV"], IRR=r["IRR"], LCOH=r["LCOH"], GWP_net=r["GWP_net"], abat=r["abat"])
_sc("base")
for om in (0.10, 0.15, 0.20): _sc(f"maintenance {int(om*100)} %", om_frac=om)
_sc("no landfill credit (RDF to incineration or cement kiln)", rdf_avoided_cost=0.0)
_sc("landfill tax 2030 (60 EUR/t)", rdf_avoided_cost=60.0)
r_noLF = tl3(sb, fb, p=dict(P, rdf_avoided_cost=0.0)); sc["no landfill credit (RDF to incineration or cement kiln)"]["GWP_net"] = r_noLF["GWP"]
sc["no landfill credit (RDF to incineration or cement kiln)"]["abat"] = (r_noLF["LCOH"] - r_noLF["p_heat"]) / ((P["NG_reference"] - r_noLF["GWP"]) * 3.6)
extra = {"operator 0.5 FTE (22.5 kEUR/y)": 22.5e3, f"ash disposal {ash_t:.0f} t/y at the landfill tax (35 EUR/t)": ash_t * 35.0}
for k, c in extra.items(): sc[k] = dict(NPV=b0["NPV"] - c * A0 / 1e3, LCOH=b0["LCOH"] + c / b0["heat"])
T27D = pd.DataFrame(sc).T; T27D.to_csv(DATA_OUT / "analysis_tea_scenarios.csv")
print(f"\n  24b-d. Base case: CAPEX {b0['CAPEX']:.0f} EUR; annual net cash flow {CF0:.0f} EUR/y; annuity factor {A0:.3f};"
      f" simple payback {b0['CAPEX'] / CF0:.1f} y; discounted payback {b0['PB']} y")
print(f"       ash {ash_t:.1f} t/y; tar {tar_t_model:.1f} t/y (model) / {tar_t_meas:.1f} t/y (measured 47 g/kg);"
      f" extra annual cost that makes NPV = 0: {b0['NPV'] * 1e3 / A0:.0f} EUR/y")
print(T27D.round(3).to_string())
_R["24b-d"] = dict(CAPEX=b0["CAPEX"], CF=CF0, annuity=A0, simple_PB=b0["CAPEX"] / CF0, disc_PB=b0["PB"], ash_t=ash_t, tar_t_model=tar_t_model,
                 tar_t_meas=tar_t_meas, breakeven_extra_cost=b0["NPV"] * 1e3 / A0, scenarios=T27D.round(3).reset_index().to_dict("records"))

# ---- 24b-e. One-at-a-time sensitivity with the ranges of Table 4 ------------------------------------------------------
RANGES = SENS_RANGES
srows = []
for k, (lo, hi) in RANGES.items():
    rl, rh = tl3(sb, fb, p=dict(P, **{k: lo})), tl3(sb, fb, p=dict(P, **{k: hi}))
    srows.append(dict(param=k, lo=lo, hi=hi, NPV_lo=rl["NPV"], NPV_hi=rh["NPV"], dNPV=abs(rh["NPV"] - rl["NPV"]), LCOH_lo=rl["LCOH"], LCOH_hi=rh["LCOH"],
                      dLCOH=abs(rh["LCOH"] - rl["LCOH"]), GWPnet_lo=rl["GWP_net"], GWPnet_hi=rh["GWP_net"], dGWP=abs(rh["GWP_net"] - rl["GWP_net"])))
T27E = pd.DataFrame(srows).set_index("param").sort_values("dNPV", ascending=False); T27E.to_csv(DATA_OUT / "analysis_oat_table4_ranges.csv")
print("\n  24b-e. One-at-a-time sensitivity with the ranges of Table 4:"); print(T27E.round(3).to_string())
_R["24b-e"] = T27E.round(4).reset_index().to_dict("records")

# ---- 24b-g. Structural choice by leave-one-dataset-out (volatiles to the emulsion vs to the bubbles) ------------------
_c1, _c2 = "two-zone, volatiles to emulsion", "two-zone, volatiles to bubbles"
lodo = {}
for ds_ in TZ[_c1]:
    others = [d for d in TZ[_c1] if d != ds_]
    m1 = np.mean([TZ[_c1][d]["MAE"] for d in others]); m2 = np.mean([TZ[_c2][d]["MAE"] for d in others])
    lodo[ds_] = dict(MAE_emulsion=round(float(m1), 3), MAE_bubbles=round(float(m2), 3), choice="emulsion" if m1 < m2 else "bubbles")
print("\n  24b-g. Leave-one-dataset-out choice of the release zone:"); [print(f"    without {k}: {v}") for k, v in lodo.items()]
_R["24b-g"] = lodo
# ---- 24b-h. Monte Carlo design check ----------------------------------------------------------------------------------
_R["24b-h"] = dict(n_samples=int(len(MC23)), distinct_NPV_values=int(MC23.NPV.nunique()), GWP_net_negative_share_base=float((MC23[MC23.case == "base"].GWP_net < 0).mean()))
print("\n  24b-h. Monte Carlo:", _R["24b-h"])
_json.dump(_R, open(DATA_OUT / "analysis_results.json", "w"), indent=1, default=float)
print("\n  24b. Results written to analysis_*.csv and analysis_results.json")

# ===== CELL BREAK =====

# =============================================================================
# 25. FIGURES OF THE ARTICLE (Figs. 2-16): ONE FIGURE PER RESULTS SECTION
# =============================================================================
# Inputs: the CSV files written by sections 11-24 of this notebook in DATA_OUT, plus two quantities computed below:
#   (25a) NPV with the measured cold gas efficiency of the reference case (Eq. 9 unchanged; the product-gas flow is
#         scaled by CGE_exp / CGE_mod, everything else is the model's), and
#   (25b) landfill emission factor of the RDF for gas recovery R = 0.3-0.7 and fossil carbon fraction 0.45-0.85 (Eq. 18 of the article).
#   Fig. 2 also compares formulation A calibrated (perfectly mixed bed with f_H2 = 0.1, f_CO_bed = 0.1). Those results come
#   from a run of this notebook with LEVEL = "calibrated" (files with prefix m25g); set CAL_RUN to its folder.
#   If CAL_RUN is not found, panels of formulation A calibrated are skipped.
# Output: DATA_OUT / "figures" (PNG at 600 dpi and vector PDF).
print("\n" + "=" * 78); print("25. FIGURES OF THE ARTICLE (one figure per results section)"); print("=" * 78, flush=True)
import json as _json
from matplotlib.colors import LinearSegmentedColormap as _LSC
FIG25 = DATA_OUT / "figures"; FIG25.mkdir(exist_ok=True)
CAL_RUN = Path(globals().get("CAL_RUN", DATA_OUT.parent / "stored_formulation_A_calibrated"))
CAL_TAG = "m25g"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6, "savefig.dpi": 600, "figure.dpi": 110})
PAL = dict(blue="#9ec5e8", coral="#f4a3a0", green="#a8d5a2", lilac="#c7b3e5", peach="#f7c99b", teal="#9fd8d6", sand="#eadc9c",
           grey="#c4c4c4", exp="#7a7a7a", edge="#5a5a5a")
LINES = ["#7fb3e0", "#ef9a93", "#8fcb88", "#b39ddb", "#f4b77c"]        # five pastel series, slightly deeper for lines
MARK = ["o", "s", "^", "D", "v"]
CMAP_P = _LSC.from_list("pastel_seq", ["#fbf9f3", "#d9e9f6", "#a9cbea", "#7fb3e0", "#5b93c7"])
CMAP_W = _LSC.from_list("pastel_warm", ["#fbf9f3", "#fbe3cf", "#f6c49e", "#ef9f84", "#d97a6c"])
def save25(fig, name):
    fig.savefig(FIG25 / f"{name}.png", bbox_inches="tight"); fig.savefig(FIG25 / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("  saved", name, flush=True)
def _top_legend(fig, handles, labels, ncol, y=1.02):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.6, columnspacing=1.4)
def _grid(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
_rd = lambda f: pd.read_csv(DATA_OUT / f)
NG_REF = P["NG_reference"]

# ---- 25a. NPV with the measured CGE of the reference case ------------------------------------------------------
rows25 = []
for T_, r_ in TABLE4.iterrows():
    if "CGE" not in r_ or pd.isna(r_["CGE"]):
        continue
    air_ = air_n2_consistent(r_, BLEND_TERRADOS)
    s_ = simulate(BLEND_TERRADOS, REACTOR_TERRADOS, OperatingPoint(float(T_), float(r_.P_gauge_kPa) if "P_gauge_kPa" in r_ else -30.0, 50.0, air_), KPS, OPT_H1)
    t_mod = tea_lca(s_, BLEND_TERRADOS)
    s2 = dict(s_); s2["Q_dry_Nm3_h"] = s_["Q_dry_Nm3_h"] * float(r_["CGE"]) / s_["CGE"]
    t_exp = tea_lca(s2, BLEND_TERRADOS)
    rows25.append(dict(T=int(T_), CGE_mod=s_["CGE"], CGE_exp=float(r_["CGE"]), NPV_mod=t_mod["NPV_kEUR"], NPV_exp=t_exp["NPV_kEUR"],
                       feed_mod=t_mod["feed_kg_h"], feed_exp=t_exp["feed_kg_h"]))
NPVM = pd.DataFrame(rows25); NPVM.to_csv(DATA_OUT / "npv_measured_efficiency.csv", index=False)
print("  25a. NPV with predicted / measured CGE (kEUR):"); print(NPVM.round(1).to_string(index=False))

# ---- 25b. Landfill emission factor: gas recovery and fossil fraction ---------------------------------------------
rows25 = []
for ff_ in sorted({0.45, 0.65, float(P["rdf_fossil_C_frac"]), 0.85}):
    for R_ in np.round(np.linspace(0.3, 0.7, 9), 3):
        rows25.append(dict(fossil_frac=ff_, R=R_, EF=landfill_EF(dict(LF, R=R_, DOC=(1 - ff_) * C_RDF_WET))))
LFS = pd.DataFrame(rows25); LFS.to_csv(DATA_OUT / "landfill_EF_sensitivity.csv", index=False)
print(f"  25b. landfill EF (kg CO2-eq/t RDF) at fossil fraction {P['rdf_fossil_C_frac']}: R=0.3 -> "
      f"{landfill_EF(dict(LF, R=0.3)):.0f}, R=0.5 -> {landfill_EF():.0f}, R=0.7 -> {landfill_EF(dict(LF, R=0.7)):.0f}")

# ---- Fig. 2 (section 3.1): validation of formulation B (parity) and comparison with gas-phase equilibrium ----------
EPB = _rd("energy_performance_terrados.csv")
PTV = _rd("analysis_points_B_equilibrium.csv"); MTV = _rd("analysis_metrics_B_equilibrium.csv")
DSG = list(PTV.dataset.unique())
SHORT = {"Terrados-Cristos (development)": "Reference case", "Carmo-Calado": "Olive pomace (Calado)", "Pio": "Pine–RDF (Pio)",
         "Parrillo 2023 (plastics-eucalyptus)": "Plastics–eucalyptus", "Parrillo 2021 (eucalyptus)": "Eucalyptus",
         "Campoy 2011 (blind, olive pomace)": "Olive pomace, blind", "Arena 2014 (blind, SRF)": "SRF, blind"}
CL7 = LINES + ["#7fcfc9", "#c9a27e"]; MK7 = MARK + ["P", "X"]
SPT = {"H2": "H$_2$", "CO": "CO", "CO2": "CO$_2$", "CH4": "CH$_4$"}
fig = plt.figure(figsize=(W2, 5.3)); gs_ = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.62, wspace=0.45)
hs7 = []
for i_, k_ in enumerate(["H2", "CO", "CO2", "CH4"]):
    ax = fig.add_subplot(gs_[0, i_])
    hi_ = float(np.nanmax(PTV[[f"{k_}_exp", f"{k_}_B"]].values)) * 1.08; xx_ = np.array([0.0, hi_])
    ax.fill_between(xx_, xx_ - 5, xx_ + 5, color="#ececec", lw=0); ax.plot(xx_, xx_, color="#777", lw=0.6)
    for j_, ds_ in enumerate(DSG):
        d_ = PTV[PTV.dataset == ds_]; bl_ = (d_.role == "blind").all()
        h = ax.scatter(d_[f"{k_}_exp"], d_[f"{k_}_B"], s=11, marker=MK7[j_], lw=0.6, edgecolor=CL7[j_], facecolor="none" if bl_ else CL7[j_], zorder=3)
        if i_ == 0: hs7.append(h)
    ax.set_xlim(0, hi_); ax.set_ylim(0, hi_); ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"({'abcd'[i_]}) {SPT[k_]}", loc="left"); ax.set_xlabel("Measured (vol%, dry)", fontsize=6.5)
    if i_ == 0: ax.set_ylabel("Predicted, B (vol%, dry)", fontsize=6.5)
    ax.tick_params(labelsize=6)
ax = fig.add_subplot(gs_[1, :]); xx_ = np.arange(len(MTV)); w2_ = 0.36
ax.bar(xx_ - w2_ / 2, MTV.MAE_B, w2_, color=PAL["coral"], edgecolor=PAL["edge"], lw=0.3)
ax.bar(xx_ + w2_ / 2, MTV.MAE_eq, w2_, color=PAL["grey"], edgecolor=PAL["edge"], lw=0.3)
for xi_, v1, v2 in zip(xx_, MTV.MAE_B, MTV.MAE_eq):
    ax.text(xi_ - w2_ / 2, v1 + 0.2, f"{v1:.2f}", ha="center", fontsize=5.8, color="#444"); ax.text(xi_ + w2_ / 2, v2 + 0.2, f"{v2:.1f}", ha="center", fontsize=5.8, color="#444")
SHORT2 = {"Terrados-Cristos (development)": "Reference\ncase", "Carmo-Calado": "Olive pomace\n(Calado)", "Pio": "Pine–RDF\n(Pio)",
          "Parrillo 2023 (plastics-eucalyptus)": "Plastics–\neucalyptus", "Parrillo 2021 (eucalyptus)": "Eucalyptus",
          "Campoy 2011 (blind, olive pomace)": "Olive pomace,\nblind", "Arena 2014 (blind, SRF)": "SRF,\nblind"}
ax.set_xticks(xx_); ax.set_xticklabels([SHORT2.get(d, d) for d in MTV.dataset], fontsize=6.3)
ax.set_ylabel("Mean MAE of H$_2$, CO, CO$_2$, CH$_4$ (pp)"); ax.set_title("(e) Formulation B and gas-phase equilibrium", loc="left"); _grid(ax)
ax.set_ylim(0, float(MTV.MAE_eq.max()) * 1.18)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=PAL["coral"]), plt.Rectangle((0, 0), 1, 1, fc=PAL["grey"])], ["B, two-zone bed", "Gas-phase equilibrium"], frameon=False, loc="upper left", ncol=2)
_top_legend(fig, hs7, [SHORT.get(d, d) for d in DSG], 4, y=0.91)
save25(fig, "Fig02_validation")

# ---- Fig. 3 (section 3.2): maps over temperature and ER ------------------------------------------------------------
MP = _rd("map_T_ER_HHV.csv")
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); fig.subplots_adjust(hspace=0.42, wspace=0.32)
for ax, (col, ttl, cm, fmt) in zip(axs.ravel(), [("CGE", "(a) Cold gas efficiency (%)", CMAP_P, "{:.0f}"), ("LHV", "(b) LHV (MJ Nm$^{-3}$)", CMAP_P, "{:.1f}"),
                                                 ("HHV", "(c) HHV (MJ Nm$^{-3}$)", CMAP_P, "{:.1f}"), ("tar", "(d) Tar yield (g kg$^{-1}$)", CMAP_W, "{:.0f}")]):
    M_ = MP.pivot(index="ER", columns="T", values=col)
    im = ax.imshow(M_.values, origin="lower", aspect="auto", cmap=cm)
    for i_ in range(M_.shape[0]):
        for j_ in range(M_.shape[1]):
            ax.text(j_, i_, fmt.format(M_.values[i_, j_]), ha="center", va="center", fontsize=6.5, color="#333")
    ax.set_xticks(range(M_.shape[1])); ax.set_xticklabels(M_.columns.astype(int)); ax.set_yticks(range(M_.shape[0])); ax.set_yticklabels([f"{e:.2f}" for e in M_.index])
    ax.set_xlabel("Bed temperature (°C)"); ax.set_ylabel("Equivalence ratio"); ax.set_title(ttl, loc="left")
    for s_ in ax.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
save25(fig, "Fig03_T_ER_maps")

# ---- Fig. 4 (section 3.2.1) and Fig. 5 (section 3.2.2): effect of temperature and of ER ------------------------------
TM = _rd(f"{FILE_TAG}_T_ER_tea_lca_map.csv")
def _curves(by, other, panels, name, xlab, leg_fmt, exp=None):
    """Curves of the T x ER map. The maps of Fig. 3 already show CGE, LHV, HHV and tar, so these figures show only the gas
       composition and the energy consequences (to avoid repeating Fig. 3). exp: measured points of the reference case."""
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.6)); fig.subplots_adjust(hspace=0.48, wspace=0.38)
    levels = sorted(TM[other].unique()); hs = []; he = None
    for ax, (col, ttl) in zip(axs.ravel(), panels):
        for i_, lv in enumerate(levels):
            d_ = TM[np.isclose(TM[other], lv)].sort_values(by)
            h, = ax.plot(d_[by], d_[col], marker=MARK[i_], ms=3.6, lw=1.3, color=LINES[i_], mec="white", mew=0.4)
            if ax is axs.ravel()[0]: hs.append(h)
        if exp is not None and f"{col}_exp" in exp:
            he = ax.scatter(exp[by], exp[f"{col}_exp"], s=26, marker="s", facecolor="white", edgecolor="#333", lw=0.9, zorder=5)
        ax.set_title(ttl, loc="left"); ax.set_xlabel(xlab); _grid(ax)
    labs = [leg_fmt(lv) for lv in levels]
    if he is not None: hs.append(he); labs.append("Experiments, ER 0.26–0.29")
    _top_legend(fig, hs, labs, len(labs), y=0.97)
    save25(fig, name)
_curves("T", "ER", [("H2", "(a) H$_2$ (vol%, dry)"), ("CO", "(b) CO (vol%, dry)"), ("CH4", "(c) CH$_4$ (vol%, dry)"),
                    ("CO2", "(d) CO$_2$ (vol%, dry)"), ("Q_ext_kW", "(e) Heat needed by the bed (kW)"), ("eff_net_pct", "(f) Net efficiency (%)")],
        "Fig04_temperature", "Bed temperature (°C)", lambda v: f"ER = {v:.2f}")
_curves("ER", "T", [("H2", "(a) H$_2$ (vol%, dry)"), ("CO", "(b) CO (vol%, dry)"), ("CO2", "(c) CO$_2$ (vol%, dry)"),
                    ("N2", "(d) N$_2$ (vol%, dry)"), ("gas_Nm3_kg", "(e) Dry gas yield (Nm$^3$ kg$^{-1}$)"), ("feed_kg_h", "(f) Fuel for 209.8 kWth (kg h$^{-1}$)")],
        "Fig05_equivalence_ratio", "Equivalence ratio", lambda v: f"{v:.0f} °C")

# ---- Fig. 6 (section 3.2.3): RDF fraction at the base conditions (700 C, ER of the reference case) -----------------------
RF = _rd("rdf_fraction_effect_tea.csv"); RF = RF[RF.case.str.startswith("Base")].sort_values("x"); xr = 100 * RF.x
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5)); fig.subplots_adjust(wspace=0.38); col = LINES[0]
kw = dict(color=col, ms=3.6, mec="white", mew=0.4)
axs[0].plot(xr, RF.CGE, "-o", **kw)
h1, = axs[1].plot(xr, RF.LHV, "-o", **kw); h2, = axs[1].plot(xr, RF.HHV, "--s", color=LINES[1], ms=3.2, mec="white", mew=0.4)
axs[1].legend([h1, h2], ["LHV", "HHV"], frameon=False, loc="upper left")
h1, = axs[2].plot(xr, RF.GWP, "-o", **kw); h2, = axs[2].plot(xr, RF.GWP_net, "--s", color=LINES[2], ms=3.2, mec="white", mew=0.4)
h3 = axs[2].axhline(NG_REF, color=PAL["edge"], ls=":", lw=0.8); axs[2].set_ylim(0, 0.085)
axs[2].legend([h1, h2, h3], ["Attributional", "Net (avoided landfill)", "Natural gas"], frameon=False, loc="upper left", bbox_to_anchor=(0, 0.79))
for ax, ttl in zip(axs, ["(a) Cold gas efficiency (%)", "(b) Heating value (MJ Nm$^{-3}$)", "(c) GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"]):
    ax.set_title(ttl, loc="left"); ax.set_xlabel("RDF in the blend (wt%)"); ax.set_xticks([10, 20, 30, 40, 50]); _grid(ax)
fig.suptitle(f"Base conditions: 700 \u00b0C, ER = {er_b:.2f}", y=1.03, fontsize=8)
save25(fig, "Fig06_RDF_fraction")

# ---- Fig. 7 (section 3.3.1): components of the levelized cost of heat, base case ------------------------------------
LC = pd.read_csv(DATA_OUT / "lcoh_components.csv", index_col=0); TLI = _rd("tea_lca_integrated.csv")
p_heat = float(TLI[(TLI.scenario.astype(str) == "2025") & (TLI.P_kWth == 209.8)].cost_NG.iloc[0])
CL = {"capex": ("Annualized investment", PAL["blue"]), "om": ("Maintenance", PAL["teal"]), "pomace": ("Olive pomace", PAL["green"]),
      "co2": ("Allowances, fossil C of RDF", PAL["grey"]), "elec": ("Electricity", PAL["sand"]), "rdf": ("Avoided landfill charge", PAL["peach"])}
fig, ax = plt.subplots(figsize=(W2 * 0.75, 2.4)); keys9 = ["capex", "om", "pomace", "co2", "elec", "rdf"]; y9 = np.arange(len(keys9) + 1)[::-1]
cum = 0.0
for j_, c_ in enumerate(keys9):
    v_ = float(LC.loc[c_, "base"]); left_ = cum if v_ >= 0 else cum + v_
    ax.barh(y9[j_], abs(v_), 0.6, left=left_, color=CL[c_][1], edgecolor=PAL["edge"], lw=0.3)
    ax.text(max(cum, cum + v_) + 0.8, y9[j_], f"{v_:+.1f}", va="center", fontsize=6.5); cum += v_
ax.barh(y9[-1], cum, 0.6, color="#d9d9d9", edgecolor=PAL["edge"], lw=0.3); ax.text(cum + 0.8, y9[-1], f"LCOH {cum:.1f}", va="center", fontsize=6.5, weight="bold")
ax.axvline(p_heat, color=PAL["edge"], ls="--", lw=0.8); ax.text(p_heat + 0.8, y9[0] + 0.45, f"natural-gas heat {p_heat:.1f}", fontsize=6.5, color="#444", va="bottom")
ax.set_yticks(y9); ax.set_yticklabels([CL[c_][0] for c_ in keys9] + ["LCOH"]); ax.set_xlim(0, p_heat * 1.25); ax.set_ylim(-0.6, len(keys9) + 1.0)
ax.set_xlabel("€ MWh$^{-1}$ of useful heat"); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
save25(fig, "Fig07_LCOH_components")

# ---- Fig. 8 (section 3.3.2): NPV, scale and efficiency check -----------------------------------------------------------
SC = _rd("opt16_scale.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7)); fig.subplots_adjust(wspace=0.62)
ax = axs[0]
for i_, (c_, l_) in enumerate([("base", "Base case")]):
    d_ = SC[(SC.case == c_) & (SC.P_kWth <= 1000)].sort_values("P_kWth")
    ax.plot(d_.P_kWth, d_.NPV_kEUR, "-" + MARK[i_], color=LINES[i_], ms=3.6, mec="white", mew=0.4, label=l_)
ax.axhline(0, color="#999", lw=0.5); ax.set_xscale("log"); ax.set_xlabel("Capacity (kWth)"); ax.set_ylabel("NPV (k€)"); _grid(ax)
ax.set_title("(a) NPV and plant capacity, base case", loc="left")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
d_ = SC[(SC.case == "base") & (SC.P_kWth <= 1000)].sort_values("P_kWth"); ax2.plot(d_.P_kWth, d_.IRR_pct, ":", color=LINES[0], lw=1.0)
ax2.set_ylabel("IRR (%), dotted"); ax2.tick_params(labelsize=6.5)
ax = axs[1]; xt = np.arange(len(NPVM)); w5 = 0.36
ax.bar(xt - w5 / 2, NPVM.NPV_mod, w5, color=PAL["coral"], edgecolor=PAL["edge"], lw=0.3, label="Predicted CGE")
ax.bar(xt + w5 / 2, NPVM.NPV_exp, w5, color=PAL["exp"], edgecolor=PAL["edge"], lw=0.3, label="Measured CGE")
ax.set_xticks(xt); ax.set_xticklabels(NPVM["T"]); ax.set_xlabel("Bed temperature (°C)"); ax.set_ylabel("NPV (k€)"); _grid(ax)
ax.set_title("(b) NPV with predicted and measured CGE", loc="left"); ax.legend(frameon=False, loc="upper right", ncol=2); ax.set_ylim(0, NPVM[["NPV_mod", "NPV_exp"]].values.max() * 1.3)
save25(fig, "Fig08_NPV")

# ---- Fig. 9 (section 3.4.1): attributional GWP100 --------------------------------------------------------------------
GP = pd.read_csv(DATA_OUT / "gwp_parts.csv", index_col=0); GR = _rd("opt16_T_ER_xRDF.csv")
_t25 = TLI[(TLI.scenario.astype(str) == "2025") & (TLI.P_kWth == 209.8)].reset_index(drop=True)
GWP_TOT = {"base": float(_t25.GWP_attr.iloc[0]), "comp": float(_t25.GWP_attr.iloc[1])}; GWP_NET = {"base": float(_t25.GWP_net.iloc[0]), "comp": float(_t25.GWP_net.iloc[1])}
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25])); fig.subplots_adjust(wspace=0.35)
ax = axs[0]; xg = np.arange(2); GL = {"fossil": ("Fossil CO$_2$ (RDF)", PAL["grey"]), "ch4": ("CH$_4$ slip", PAL["lilac"]), "elec": ("Electricity", PAL["sand"])}
b_ = 0.0
for c_ in ["fossil", "ch4", "elec"]:
    ax.bar(0, GP.loc[c_, "base"], 0.55, bottom=b_, color=GL[c_][1], edgecolor=PAL["edge"], lw=0.3, label=GL[c_][0]); b_ += GP.loc[c_, "base"]
ax.text(0, b_ + 0.0015, f"{GWP_TOT['base']:.3f}", ha="center", fontsize=6.5)
ax.bar(1, NG_REF, 0.55, color=PAL["blue"], edgecolor=PAL["edge"], lw=0.3, label="Natural gas"); ax.text(1, NG_REF + 0.0015, f"{NG_REF:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(xg); ax.set_xticklabels(["Base case", "Natural\ngas"]); ax.set_ylabel("kg CO$_2$-eq MJ$^{-1}$"); _grid(ax); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, NG_REF * 1.35)
ax.set_title("(a) Contributions", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=6.2)
ax = axs[1]; sc_ = ax.scatter(100 * GR.fossil_C_share, GR.GWP_kg_MJ, c=100 * GR.x_RDF, cmap=CMAP_W, s=16, edgecolor=PAL["edge"], lw=0.3)
cb = fig.colorbar(sc_, ax=ax, pad=0.02); cb.set_label("RDF (wt%)", fontsize=6.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
ax.axhline(NG_REF, color=PAL["edge"], ls=":", lw=0.8); ax.set_xlabel("Fossil share of the carbon in the gas (%)"); ax.set_ylabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)")
ax.set_title("(b) 125 operating conditions", loc="left"); _grid(ax)
save25(fig, "Fig09_GWP_attributional")

# ---- Fig. 10 (section 3.4.2): avoided landfill ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7)); fig.subplots_adjust(wspace=0.32)
ax = axs[0]; xg = np.arange(3)
v12 = [GWP_TOT["base"], GWP_NET["base"], NG_REF]
ax.bar(xg, v12, 0.55, color=[PAL["grey"], PAL["green"], PAL["blue"]], edgecolor=PAL["edge"], lw=0.3)
for xi_, v_ in zip(xg, v12): ax.text(xi_, v_ + 0.001, f"{v_:.3f}", ha="center", fontsize=6.3)
ax.text(1, GWP_NET["base"] + 0.0055, f"landfill credit\n\u2212{GWP_TOT['base'] - GWP_NET['base']:.3f}", fontsize=6, va="bottom", ha="center", color="#555")
ax.set_xticks(xg); ax.set_xticklabels(["Attributional", "With avoided\nlandfill", "Natural\ngas"]); ax.set_ylabel("kg CO$_2$-eq MJ$^{-1}$"); _grid(ax); ax.set_ylim(0, NG_REF * 1.15)
ax.set_title("(a) GWP$_{100}$ of the useful heat, base case", loc="left")
ax = axs[1]
for i_, ff_ in enumerate(sorted(LFS.fossil_frac.unique())):
    d_ = LFS[LFS.fossil_frac == ff_]; ax.plot(d_.R, d_.EF, "-" + MARK[i_], color=LINES[i_], ms=3.4, mec="white", mew=0.4, label=f"fossil C fraction {ff_:.2f}" + (" (base)" if np.isclose(ff_, P["rdf_fossil_C_frac"]) else ""))
ax.axvline(LF["R"], color="#999", lw=0.5, ls="--"); ax.set_xlabel("Landfill gas recovery fraction"); ax.set_ylabel("kg CO$_2$-eq t$^{-1}$ RDF")
ax.set_title("(b) Landfill emission factor of the RDF", loc="left"); ax.legend(frameon=False); _grid(ax)
save25(fig, "Fig10_avoided_landfill")

# ---- Fig. 11 (section 3.5): integrated performance -------------------------------------------------------------------
GL_ = _rd("grid_LCOH_GWP.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1])); fig.subplots_adjust(wspace=0.38)
ax = axs[0]; sc_ = ax.scatter(GL_.GWP_net, GL_.LCOH, c=100 * GL_.x_RDF, cmap=CMAP_W, s=16, edgecolor=PAL["edge"], lw=0.3)
cb = fig.colorbar(sc_, ax=ax, pad=0.02); cb.set_label("RDF (wt%)", fontsize=6.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
ax.axhline(p_heat, color=PAL["edge"], ls="--", lw=0.8); ax.text(GL_.GWP_net.min(), p_heat - 0.6, "natural-gas heat", fontsize=6, va="top", color="#555")
ax.set_xlabel("Net GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.set_ylabel("LCOH (€ MWh$^{-1}$)"); ax.set_title("(a) 125 operating conditions", loc="left"); _grid(ax)
ax = axs[1]; T9 = TLI.copy(); T9["scenario"] = T9.scenario.astype(str)
grp = [("2025", 209.8), ("2025", 1000.0), ("2030", 209.8), ("2030", 1000.0)]; xg = np.arange(len(grp)); w7 = 0.36
pt_ = T9.point.unique()[0]
v_ = [float(T9[(T9.point == pt_) & (T9.scenario == s) & (T9.P_kWth == pk)].abat_net.iloc[0]) for s, pk in grp]
ax.bar(xg, v_, 0.55, color=[PAL["blue"], PAL["blue"], PAL["coral"], PAL["coral"]], edgecolor=PAL["edge"], lw=0.3)
for xi_, vv in zip(xg, v_): ax.text(xi_, vv - 6, f"{vv:.0f}", ha="center", va="top", fontsize=6.3)
ax.axhline(0, color="#999", lw=0.5); ax.set_xticks(xg); ax.set_xticklabels([f"{s}\n{pk / 1000:.2f} MWth" if pk < 300 else f"{s}\n1 MWth" for s, pk in grp]); ax.set_ylim(min(v_) * 1.2, 0)
ax.set_ylabel("€ t$^{-1}$ CO$_2$-eq"); ax.set_title("(b) CO$_2$ abatement cost, base case", loc="left"); _grid(ax)
save25(fig, "Fig11_integrated")

# ---- Fig. 12 and Fig. 13 (section 3.6): optimization ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(W2 * 0.7, 3.0))
nd = GR[GR.pareto.astype(bool)]; dd = GR[~GR.pareto.astype(bool)]
ax.scatter(dd.GWP_kg_MJ, dd.NPV_kEUR, s=13, color=PAL["grey"], edgecolor="white", lw=0.3, label="Dominated")
sc_ = ax.scatter(nd.GWP_kg_MJ, nd.NPV_kEUR, s=26, c=nd["T"], cmap=CMAP_W, edgecolor=PAL["edge"], lw=0.4, label="Non-dominated")
cb = fig.colorbar(sc_, ax=ax, pad=0.02); cb.set_label("Bed temperature of the non-dominated points (°C)", fontsize=6.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
opt_ = GR[np.isclose(GR["T"], T_COMP) & np.isclose(GR.ER, ER_COMP) & np.isclose(GR.x_RDF, X_COMP)]
if len(opt_): ax.scatter(opt_.GWP_kg_MJ, opt_.NPV_kEUR, s=70, marker="*", color=LINES[1], edgecolor="#333", lw=0.5, zorder=5, label="Constrained optimum")
ax.set_xlabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.set_ylabel("NPV (k€)"); ax.legend(frameon=False, loc="lower left", fontsize=6.3); _grid(ax)
save25(fig, "Fig12_Pareto")
EO = _rd("opt19_energy_environment.csv")
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.8)); fig.subplots_adjust(wspace=0.42)
ax = axs[0]; sc_ = ax.scatter(EO["T"], EO.ER, c=EO.D, cmap=CMAP_P, s=14, edgecolor=PAL["edge"], lw=0.3)
b_ = EO.loc[EO.D.idxmax()]; ax.scatter([b_["T"]], [b_.ER], marker="*", s=80, color=LINES[1], edgecolor="#333", lw=0.5, zorder=5)
cb = fig.colorbar(sc_, ax=ax, pad=0.02); cb.set_label("Desirability", fontsize=6.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
ax.set_xlabel("Bed temperature (°C)"); ax.set_ylabel("Equivalence ratio"); ax.set_title(f"(a) {len(EO)} evaluated points", loc="left")
ax = axs[1]; nd_ = EO[EO.pareto.astype(bool)]
ax.scatter(EO.GWP, EO.LHV, s=11, color=PAL["grey"], edgecolor="white", lw=0.3, label="Evaluated")
ax.scatter(nd_.GWP, nd_.LHV, s=18, color=LINES[0], edgecolor=PAL["edge"], lw=0.3, label=f"Non-dominated ({len(nd_)})")
ax.scatter([b_.GWP], [b_.LHV], marker="*", s=80, color=LINES[1], edgecolor="#333", lw=0.5, zorder=5, label="Highest desirability")
ax.set_xlabel("GWP$_{100}$ (kg CO$_2$-eq MJ$^{-1}$)"); ax.set_ylabel("LHV (MJ Nm$^{-3}$)"); ax.set_title("(b) Heating value and climate impact", loc="left")
ax.legend(frameon=False, fontsize=6.2, loc="lower right"); _grid(ax)
save25(fig, "Fig13_desirability")

# ---- Fig. 14 and Fig. 15 (section 3.7): uncertainty and sensitivity -----------------------------------------------------
MC = _rd("montecarlo_extended.csv")
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4)); fig.subplots_adjust(wspace=0.35); hs = []
for i_, (c_, l_) in enumerate([("base", "Base case"), (sorted(set(MC.case) - {"base"})[0], "Constrained optimum")]):
    g_ = MC[MC.case == c_]
    for ax, col in zip(axs, ["NPV", "LCOH", "GWP_net"]):
        h = ax.hist(g_[col], bins=40, color=[PAL["blue"], PAL["coral"]][i_], alpha=0.75, edgecolor="white", lw=0.2)
    hs.append(plt.Rectangle((0, 0), 1, 1, fc=[PAL["blue"], PAL["coral"]][i_]))
axs[0].axvline(0, color=PAL["edge"], lw=0.7, ls="--"); axs[1].axvline(p_heat, color=PAL["edge"], lw=0.7, ls="--"); axs[2].axvline(NG_REF, color=PAL["edge"], lw=0.7, ls="--")
for ax, ttl, xl in zip(axs, ["(a) NPV", "(b) LCOH", "(c) Net GWP$_{100}$"], ["k€", "€ MWh$^{-1}$", "kg CO$_2$-eq MJ$^{-1}$"]):
    ax.set_title(ttl, loc="left"); ax.set_xlabel(xl); _grid(ax)
axs[0].set_ylabel("Samples"); _top_legend(fig, hs, ["Base case", "Constrained optimum"], 2, y=0.98)
save25(fig, "Fig14_MonteCarlo")
SE = pd.read_csv(DATA_OUT / "sensitivity_extended.csv", index_col=0)
BASE_ = TLI[(TLI.scenario.astype(str) == "2025") & (TLI.P_kWth == 209.8)].iloc[0]
SLB = {"capex_spec": "Specific investment", "boiler_eff": "Boiler efficiency", "ng_price": "Natural-gas price", "lifetime": "Service life",
       "discount_rate": "Discount rate", "om_frac": "Maintenance", "ets_price": "Allowance price", "pomace_cost": "Pomace cost",
       "rdf_fossil_C_frac": "Fossil C of RDF", "elec_price": "Electricity price", "rdf_avoided_cost": "Landfill charge", "grid_EF": "Grid emission factor"}
fig, axs = plt.subplots(1, 3, figsize=(W2, 3.2)); fig.subplots_adjust(wspace=0.95)
for ax, (m_, bv, ttl) in zip(axs, [("NPV", BASE_.NPV, "(a) NPV (k€)"), ("LCOH", BASE_.LCOH, "(b) LCOH (€ MWh$^{-1}$)"), ("GWP_net", BASE_.GWP_net, "(c) Net GWP$_{100}$")]):
    S_ = SE[[f"{m_}_lo", f"{m_}_hi"]].copy(); S_["rng"] = (S_.iloc[:, 0] - S_.iloc[:, 1]).abs(); S_ = S_.sort_values("rng")
    y_ = np.arange(len(S_))
    ax.barh(y_, S_[f"{m_}_lo"] - bv, left=bv, color=PAL["blue"], edgecolor=PAL["edge"], lw=0.3, height=0.62, label="Lower bound")
    ax.barh(y_, S_[f"{m_}_hi"] - bv, left=bv, color=PAL["coral"], edgecolor=PAL["edge"], lw=0.3, height=0.62, label="Upper bound")
    ax.axvline(bv, color="#555", lw=0.6); ax.set_yticks(y_); ax.set_yticklabels([SLB.get(i, i) for i in S_.index], fontsize=6.3)
    ax.set_title(ttl, loc="left"); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
_top_legend(fig, [plt.Rectangle((0, 0), 1, 1, fc=PAL["blue"]), plt.Rectangle((0, 0), 1, 1, fc=PAL["coral"])], ["Lower bound of the Table 4 range", "Upper bound of the Table 4 range"], 2, y=0.98)
save25(fig, "Fig15_sensitivity")

# ---- Fig. 16 (section 3.8): 2025 versus 2030 -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.5)); fig.subplots_adjust(wspace=0.55)
pts_ = list(T9.point.unique()); xs_ = np.arange(2); w8 = 0.36
for ax, (col, ttl) in zip(axs, [("NPV", "(a) NPV (k€)"), ("IRR", "(b) IRR (%)"), ("LCOH", "(c) LCOH (€ MWh$^{-1}$)"), ("abat_net", "(d) Abatement cost (€ t$^{-1}$ CO$_2$-eq)")]):
    for j_, (s_, c_) in enumerate([("2025", PAL["blue"]), ("2030", PAL["green"])]):
        v_ = [float(T9[(T9.point == p_) & (T9.scenario == s_) & (T9.P_kWth == 209.8)][col].iloc[0]) for p_ in pts_]
        ax.bar(xs_ + (j_ - 0.5) * w8, v_, w8, color=c_, edgecolor=PAL["edge"], lw=0.3, label=s_)
        if col == "LCOH":
            cng = float(T9[(T9.scenario == s_) & (T9.P_kWth == 209.8)].cost_NG.iloc[0])
            ax.axhline(cng, color=c_, ls="--", lw=0.9)
    ax.set_xticks(xs_); ax.set_xticklabels(["Base", "Optimum"]); ax.set_title(ttl, loc="left"); _grid(ax); ax.axhline(0, color="#999", lw=0.4)
_top_legend(fig, [plt.Rectangle((0, 0), 1, 1, fc=PAL["blue"]), plt.Rectangle((0, 0), 1, 1, fc=PAL["green"])],
            ["2025 scenario", "2030 scenario (dashed in (c): value of the natural-gas heat)"], 2, y=0.98)
save25(fig, "Fig16_scenarios_2025_2030")
print(f"  25. {len(list(FIG25.glob('*.png')))} figures in {FIG25}")

# =============================================================================
# 26. ARTICLE OUTPUT: TABLES 1-11 AND FIGURES 1-16
# =============================================================================
# Only this section writes to the notebook. Everything printed by sections 0-25 is in DATA_OUT/full_output.txt and
# the figures of the article are in DATA_OUT/figures (see ARTICLE_ONLY and SAVE_AUX in section 1).
# Tables 3 and 4 are documentary (they describe corrections and datasets); their numbers are taken from the code.
# Two quantities come from runs that are not repeated here, as in section 14e: the calibrated variant of formulation A
# (CAL_RUN, LEVEL = "calibrated") and its leave-one-reactor-out MAE (LORO_STORED, same run).
if ARTICLE_ONLY:
    _sys.stdout = _STDOUT; _LOG.close()
plt.close("all")
try:
    from IPython.display import display as _display, Markdown as _Md, Image as _Img
    _IN_NB = get_ipython() is not None
except Exception:
    _IN_NB = False
def _table(df, title):
    if _IN_NB:
        _display(_Md(f"**{title}**")); _display(df)
    else:
        print("\n" + title); print(df.to_string()); print()
_f1 = lambda v: f"{v:.1f}"; _f2 = lambda v: f"{v:.2f}"
from decimal import Decimal as _Dec, ROUND_HALF_UP as _HUP
_f1h = lambda v: str(_Dec(repr(round(float(v), 9))).quantize(_Dec("0.1"), rounding=_HUP))   # half-up, as in the article (13.25 -> 13.3)
_sg = lambda v: "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
SPX = ["H2", "CO", "CO2", "CH4"]
print("=" * 78); print(f"ARTICLE OUTPUT - model {MODEL_NAME} (formulation B, two-zone bed)"); print("=" * 78)

# ---- Table 1. Feedstock properties ----------------------------------------------------------------------------
_c = BLEND_TERRADOS.components; _w = BLEND_TERRADOS.mass_fractions; _kp, _kr = "olive_pomace", "RDF"
_rows = []
for lab, get in [("Carbon (wt%)", lambda c: c.ultimate_dry["C"]), ("Hydrogen (wt%)", lambda c: c.ultimate_dry["H"]), ("Nitrogen (wt%)", lambda c: c.ultimate_dry["N"]),
                 ("Oxygen (wt%)", lambda c: c.ultimate_dry["O"]), ("Ash (wt%)", lambda c: c.ash_ar), ("Moisture (wt%, as received)", lambda c: c.moisture_ar),
                 ("Volatile matter (wt%, as received)", lambda c: c.VM_ar), ("Fixed carbon (wt%, as received)", lambda c: c.FC_ar), ("HHV (MJ/kg)", lambda c: c.HHV_MJ_kg)]:
    a, b = get(_c[_kp]), get(_c[_kr]); _rows.append([lab, _f1h(a), _f1h(b), _f1h(_w[_kp] * a + _w[_kr] * b)])
_table(pd.DataFrame(_rows, columns=["Parameter", "Olive pomace", "RDF", "70/30 blend"]).set_index("Parameter"), "Table 1. Feedstock properties")

# ---- Table 2. Base conditions of the reference case -----------------------------------------------------------
_rc = REACTOR_TERRADOS
_air_m = [air_n2_consistent(TABLE4.loc[T_], BLEND_TERRADOS) for T_ in TABLE4.index]
_er_m = [a_ / (stoich_air_Nm3_per_kg(BLEND_TERRADOS) * TABLE4.loc[T_].feed_kg_h) for a_, T_ in zip(_air_m, TABLE4.index)]
_rows = [["Reactor internal diameter / height", "m", f"{_rc.D_m:.2f} / {_rc.H_total_m:.2f}"], ["Bed material / inventory", "- / kg", f"{_rc.bed_material.capitalize()} / {_rc.bed_mass_kg:.0f}"],
         ["Dolomite particle size (reported range)", "mm", "0-2"], ["Particle size used in the model (assumed)", "mm", f"{1000 * _rc.dp_bed_m:.1f}"],
         ["Number of air nozzles", "-", f"{_rc.n_orifices}"], ["Fuel feed rate", "kg/h", f"{TABLE4.feed_kg_h.iloc[0]:.0f}"],
         ["Bed temperature", "°C", f"{TABLE4.index.min():.0f}-{TABLE4.index.max():.0f}"],
         ["Operating pressure (gauge)", "kPa", f"{TABLE4.P_gauge_kPa.max():.0f} to {TABLE4.P_gauge_kPa.min():.0f}"],
         ["Air flow rate, as listed in the source", "Nm3/h", f"{TABLE4.air_Nm3_h.min():.0f}-{TABLE4.air_Nm3_h.max():.0f}"],
         ["Air flow rate used in the model (nitrogen balance)", "Nm3/h", f"{min(_air_m):.1f}-{max(_air_m):.1f}"],
         ["Equivalence ratio with the air used in the model", "-", f"{min(_er_m):.2f}-{max(_er_m):.2f}"]]
_table(pd.DataFrame(_rows, columns=["Description", "Unit", "Value"]).set_index("Description"), "Table 2. Base conditions of the reference case")

# ---- Supplementary table. Audit of the reaction network (documentary; corrections implemented in section 9) -----------------
_rows = [["Oxidation of C2H4 and C3H6 (four reactions)", "Pre-exponential factors listed in m3-mol-s units", "Same values in cm-s-mol units", "Factors multiplied by 3.16e-5"],
         ["Phenol + steam", "Rate of third order in steam", "First order in phenol, no steam term", "Steam order removed; reaction switched off"],
         ["Water-gas shift", "Rate constant only", "Constant with a fitting factor of 0.1", "Factor restored"],
         ["H2 oxidation", "Rate constant only", "Constant with a fitting factor of 0.001", "Factor restored"],
         ["Eucalyptus devolatilization yields", "Add up to 111 wt%", "Not traceable", "Kept, with elemental closure; flagged as an assumption"]]
_table(pd.DataFrame(_rows, columns=["Item", "As compiled", "Primary source", "Action"]).set_index("Item"), "Supplementary table. Audit of the reaction network against its primary sources")

# ---- Table 3. Experimental datasets ---------------------------------------------------------------------------
_B2 = pd.read_csv(DATA_OUT / f"{FILE_TAG}_blind2_summary.csv"); _B2b = _B2[_B2.formulation.str.startswith("B")]
_nb = lambda s: int(_B2b[_B2b.dataset.str.contains(s)].n.iloc[0])
_rows = [["Terrados-Cristos et al. (2026)", "BFB, 0.50 m, dolomite", "Olive pomace 70 % + RDF 30 %", len(DATASETS["Terrados 2026"]), "Development"],
         ["Carmo-Calado et al. (2023)", "BFB, 0.50 m, dolomite", "Olive pomace", len(DATASETS["Calado 2023"]), "External validation"],
         ["Pio et al. (2020)", "BFB, 0.25 m, sand", "Pine chips or pellets with 0-100 % RDF", len(DATASETS["Pio 2020"]), "External validation"],
         ["Parrillo et al. (2023)", "BFB, 0.489 m, olivine", "Plastics and eucalyptus (7 blends)", len(DATASETS["FABER (Parrillo 2023)"]), "External validation"],
         ["Parrillo et al. (2021)", "BFB, 0.489 m, olivine", "Eucalyptus chips", int(P21_BLIND.blind.sum()), "Blind test of A; external validation of B"],
         ["Campoy et al. (2011)", "BFB, 0.150 m, ofite", "Exhausted olive pomace", _nb("orujillo"), "Blind test of B"],
         ["Arena and Di Gregorio (2014)", "BFB, 0.381 m, olivine", "Solid recovered fuel", _nb("SRF"), "Blind test of B"]]
_table(pd.DataFrame(_rows, columns=["Dataset", "Reactor and bed", "Fuel", "Points", "Role"]).set_index("Dataset"), "Table 3. Experimental validation sets")

# ---- Table 4. Techno-economic and carbon-footprint parameters --------------------------------------------------
_EFR = lambda **kw: landfill_EF(dict(LF, **kw))
_rows = [["Thermal demand (kWth) / operating hours (h/y)", f"{P['P_design_kWth']} / {P['hours']:.0f}", "-"],
         ["Specific investment (€/kWth)", f"{P['capex_spec']:.0f}", "±30 %"], ["Maintenance (% of investment per year)", f"{100 * P['om_frac']:.0f}", "3-8"],
         ["Discount rate (%) / service life (years)", f"{100 * P['discount_rate']:.0f} / {P['lifetime']}", "6-10 / 10-20"], ["Olive pomace cost (€/t)", f"{P['pomace_cost']:.0f}", "20-40"],
         ["Fossil fraction of RDF carbon", f"{P['rdf_fossil_C_frac']:.2f}", "0.45-0.85"], ["GWP100 of CH4", f"{P['GWP_CH4']}", "-"],
         ["Electricity price (€/MWh)", f"{P['elec_price']:.0f}", "110-160"], ["Avoided RDF landfill cost (€/t)", f"{P['rdf_avoided_cost']:.0f}", f"35-60; {SCEN['2030']['rdf_avoided_cost']:.0f} in 2030"],
         ["Grid emission factor (kg CO2-eq/kWh)", f"{P['grid_EF']:.3f}", f"0.12-0.21; {SCEN['2030']['grid_EF']:.3f} in 2030"], ["Syngas boiler efficiency (%)", f"{100 * P['boiler_eff']:.0f}", "75-85"],
         ["Pretreatment electricity (kWh per MJ of useful heat)", f"{P['elec_kWh_per_MJ']}", "-"], ["Unburned CH4 (kg per MJ of useful heat)", f"{P['CH4_slip']:.1e}", "-"],
         ["Natural-gas reference (kg CO2-eq/MJ)", f"{P['NG_reference']:.3f}", "-"], ["Natural-gas price (€/MWh)", f"{P['ng_price']}", "35-60"],
         ["CO2 allowance price (€/t)", f"{P['ets_price']:.0f}", f"60-90; {SCEN['2030']['ets']:.0f} in 2030"],
         ["Direct natural-gas emissions (t CO2/MWh) / boiler efficiency (%)", f"{P['EF_NG_direct']} / {100 * P['eta_NG']:.0f}", "-"],
         ["RDF landfill emissions (kg CO2-eq/t)", f"{landfill_EF():.0f}",
          f"{_EFR(R=0.7):.0f}-{_EFR(R=0.3):.0f} (gas recovery); {_EFR(DOC=(1 - 0.85) * C_RDF_WET):.0f}-{_EFR(DOC=(1 - 0.45) * C_RDF_WET):.0f} (fossil fraction)"]]
_table(pd.DataFrame(_rows, columns=["Parameter", "Value (2025)", "Range / 2030"]).set_index("Parameter"), "Table 4. Techno-economic and carbon-footprint parameters")

# ---- Tables 5 and 6. Validation of the three variants -----------------------------------------------------------
_S = pd.read_csv(DATA_OUT / f"{FILE_TAG}_summary_by_dataset_species.csv")
_CAL = pd.read_csv(CAL_RUN / f"{CAL_TAG}_summary_by_dataset_species.csv") if (CAL_RUN / f"{CAL_TAG}_summary_by_dataset_species.csv").exists() else None
_BLC = None
if (CAL_RUN / f"{CAL_TAG}_blind_parrillo2021_points.csv").exists():
    _BLC = pd.read_csv(CAL_RUN / f"{CAL_TAG}_blind_parrillo2021_points.csv"); _BLC = _BLC[_BLC.blind & _BLC.level.str.contains("calibrated")]
_P21b = P21_BLIND[P21_BLIND.blind]
_DS = [("Terrados 2026", "Terrados", "Terrados-Cristos et al. (2026)"), ("Calado 2023", "Calado", "Carmo-Calado et al. (2023)"),
       ("Pio 2020", "Pio", "Pio et al. (2020)"), ("FABER (Parrillo 2023)", "FABER", "Parrillo et al. (2023)"), ("Parrillo 2021 (not blind here)", None, "Parrillo et al. (2021)")]
_A0, _BB = TZ["reference (mixed bed)"], TZ["two-zone, volatiles to emulsion"]
_rows, _mae = [], {}
for key, short, lab in _DS:
    _mae[key] = [_A0[key]["MAE"], np.nan, _BB[key]["MAE"]]; _cm = []
    for i, k in enumerate(SPX):
        if short is not None:
            r_ = _S[(_S.dataset == short) & (_S.species == k)].iloc[0]; n_, exp_ = int(r_.n), float(r_.exp_mean)
            if _CAL is not None:
                c_ = _CAL[(_CAL.dataset == short) & (_CAL.species == k)].iloc[0]; cal = (exp_ + c_.bias, c_.RMSE, c_.bias); _cm.append(c_.MAE)
        else:
            n_, exp_ = len(_P21b), float(_P21b[k].mean())
            if _BLC is not None:
                d_ = (_BLC[f"{k}_mod"] - _BLC[f"{k}_exp"]).values; cal = (exp_ + d_.mean(), np.sqrt((d_ ** 2).mean()), d_.mean()); _cm.append(np.abs(d_).mean())
        a0 = (exp_ + _A0[key]["bias"][i], _A0[key]["RMSE"][i], _A0[key]["bias"][i]); bb = (exp_ + _BB[key]["bias"][i], _BB[key]["RMSE"][i], _BB[key]["bias"][i])
        cell = lambda t: (_f2(t[0]), f"{_f1(t[1])} ({_sg(t[2])})")
        _rows.append([lab if i == 0 else "", k, n_ if i == 0 else "", _f2(exp_), *cell(a0), *(cell(cal) if _CAL is not None else ("-", "-")), *cell(bb)])
    if _cm: _mae[key][1] = float(np.mean(_cm))
_table(pd.DataFrame(_rows, columns=["Dataset", "Gas", "n", "Exp. mean", "A0 mean", "A0 RMSE (bias)", "Acal mean", "Acal RMSE (bias)", "B mean", "B RMSE (bias)"]),
       "Table 5. Experimental and predicted mean (vol%, dry), RMSE and bias (pp) of the three variants (A0: mixed bed, nothing fitted; Acal: mixed bed, calibrated; B: two-zone bed)")
LORO_STORED = {"Terrados 2026": 0.87, "Calado 2023": 1.76, "Pio 2020": 3.67, "FABER (Parrillo 2023)": 1.62}   # mean MAE, leave-one-reactor-out, LEVEL = "calibrated" run
# Portalegre values: the pair fitted without Portalegre is (0.1, 0.1), the same as the global pair, so they equal the
# in-sample values.
_MAPE_B = {short: float(_S[(_S.dataset == short) & _S.species.isin(SPX)].MAPE.mean()) for _, short, _ in _DS if short}
_ape = []                                              # Parrillo 2021 with formulation B: relative errors of the blind points
for _, r_ in _P21b.iterrows():
    s_ = simulate(faber_feed(0.0, 1.0), REACTOR_FABER, OperatingPoint(r_.T_bed_C, P_GAUGE_FABER, r_.feed_kg_h, r_.air_Nm3_h), KPS_PRED, OPT_REF)
    _ape.append([100 * abs(s_["y"][k] - r_[k]) / r_[k] for k in SPX])
_MAPE_B[None] = float(np.mean(_ape))
_rows = []
for key, short, lab in _DS:
    rm_, i_ = max((v_, j_) for j_, v_ in enumerate(_BB[key]["RMSE"]))
    _rows.append([lab, len(_P21b) if short is None else int(_S[(_S.dataset == short)].n.iloc[0]), _f2(_mae[key][0]), _f2(_mae[key][1]),
                  _f2(LORO_STORED[key]) if key in LORO_STORED else "-", _f2(_mae[key][2]), f"{rm_:.1f} ({SPX[i_]})",
                  "Yes" if rm_ <= RMSE_LIMIT else f"No ({SPX[i_]})", "Yes" if _mae[key][2] <= MAE_LIMIT else "No"])
_table(pd.DataFrame(_rows, columns=["Dataset", "Points", "A0", "Acal", "Acal, leave-one-reactor-out", "B", "Largest RMSE of B", f"B: RMSE <= {RMSE_LIMIT:.0f} pp", f"B: mean MAE <= {MAE_LIMIT:.0f} pp"]).set_index("Dataset"),
       "Table 6. Mean MAE of H2, CO, CO2 and CH4 (pp) of the three variants, and largest RMSE of formulation B")
print("  MAPE of formulation B (%; reported in the log only, not used as a criterion): " + ", ".join(f"{lab} {_MAPE_B[short]:.1f}" for _, short, lab in _DS))

# ---- Table 7. Blind test of formulation B ----------------------------------------------------------------------
_BP = pd.read_csv(DATA_OUT / f"{FILE_TAG}_blind2_points.csv")
_rows = []
for s_, lab in [("orujillo", "Exhausted olive pomace (Campoy et al., 2011)"), ("SRF", "Solid recovered fuel (Arena and Di Gregorio, 2014)")]:
    r_ = _B2b[_B2b.dataset.str.contains(s_)].iloc[0]
    for i, k in enumerate(SPX):
        exp_ = None
        for c_ in ("dataset", "case", "fuel"):
            if c_ in _BP.columns:
                g_ = _BP[_BP[c_].astype(str).str.contains(s_) & (_BP.get("formulation", pd.Series("B", index=_BP.index)).astype(str).str.startswith("B"))]
                if len(g_) and f"{k}_exp" in g_: exp_ = float(g_[f"{k}_exp"].mean()); break
        _rows.append([lab if i == 0 else "", k, int(r_.n) if i == 0 else "", "-" if exp_ is None else _f2(exp_), "-" if exp_ is None else _f2(exp_ + r_[f"{k}_bias"]),
                      _f2(r_[f"{k}_MAE"]), _f2(r_[f"{k}_RMSE"]), f"{r_[f'{k}_bias']:+.2f}"])
_table(pd.DataFrame(_rows, columns=["Dataset", "Gas", "n", "Exp. mean", "Model mean", "MAE", "RMSE", "Bias"]),
       "Table 7. Blind test of formulation B (MAE, RMSE and bias in pp)")

# ---- Table 8. Plant sizing and annual balance of the base case ---------------------------------------------------
_PF = pd.read_csv(DATA_OUT / "baseline_compromise_performance.csv", index_col=0); _IR = pd.read_csv(DATA_OUT / "industrial_requirements.csv", index_col=0)
_tq = {k: tea_lca(*SIMS[k]) for k in ("base", "comp")}
_op = lambda k: f"{_PF.loc['T', k]:.0f} / {_PF.loc['ER', k]:.2f} / {100 * PTS[k][2]:.0f}"
_lab9 = ["Feed rate (kg/h)", "Blend / olive pomace / RDF (t/y)", "Product gas (10^3 Nm3/y)", "LHV of the gas (MJ/Nm3)", "Chemical energy of the gas (MWh/y)",
         "Useful heat (MWh/y)", "Electricity (MWh/y)", "Fossil CO2 emitted (t/y)", "Emissions avoided at the landfill (t CO2-eq/y)"]
_c9 = [_f1(_IR.loc["feed_kg_h", "base"]), f"{_IR.loc['feed_t', 'base']:.0f} / {_IR.loc['pomace_t', 'base']:.0f} / {_IR.loc['rdf_t', 'base']:.0f}",
       f"{_IR.loc['syngas_Nm3', 'base'] / 1e3:.0f}", _f2(_PF.loc["LHV", "base"]), f"{_IR.loc['syngas_heat_MWh', 'base']:.0f}", f"{_IR.loc['useful_heat_MWh', 'base']:.0f}",
       _f1(_IR.loc["elec_MWh", "base"]), f"{_IR.loc['fossil_CO2_t', 'base']:.0f}", f"{_IR.loc['landfill_avoided_tCO2e', 'base']:.0f}"]
_o9 = ["50 kg/h x 209.8 kWth / chemical energy of the gas predicted by the model", "Feed rate x operating hours; blend ratio 70/30",
       "Gas yield predicted by the model x fuel consumed", "Predicted by the model", "Demand x operating hours",
       "Chemical energy x boiler efficiency (0.80)", "Useful heat x pretreatment electricity (Table 4)",
       "Carbon in the gas predicted by the model x fossil fraction of the carbon", "RDF consumed x landfill emission factor (Eq. 18)"]
_table(pd.DataFrame({"Quantity": _lab9, "Base case": _c9, "Origin": _o9}).set_index("Quantity"),
       "Table 8. Annual balance of the base case for a heat demand of 209.8 kWth (7200 h/y), with the origin of each quantity")

# ---- Table 9. Base case and constrained optimum (Section 3.6) ------------------------------------------------------
_TL = pd.read_csv(DATA_OUT / "tea_lca_integrated.csv"); _TL = _TL[(_TL.scenario.astype(str) == "2025") & (_TL.P_kWth == 209.8)].reset_index(drop=True)
_tl = {"base": _TL.iloc[0], "comp": _TL.iloc[1]}
_c10 = lambda k: [_op(k), " / ".join(_f1(_PF.loc[s, k]) for s in SPX), _f2(_PF.loc["LHV", k]), _f1(_PF.loc["CGE", k]), _f1(_tq[k]["Q_ext_kW"]),
                  _f1(_tq[k]["eff_net_pct"]), _f1(_PF.loc["tar", k]), f"{_IR.loc['feed_t', k]:.0f}", f"{_IR.loc['syngas_Nm3', k] / 1e3:.0f}", f"{_IR.loc['fossil_CO2_t', k]:.0f}",
                  f"{_tl[k].NPV:.0f}", _f1(_tl[k].IRR), _f1(_tl[k].LCOH), f"{_tl[k].GWP_attr:.3f} / {_tl[k].GWP_net:.3f}", f"{_tl[k].abat_net:.0f}"]
_lab10 = ["Temperature (°C) / ER / RDF (%)", "H2 / CO / CO2 / CH4 (vol%, dry)", "LHV (MJ/Nm3)", "Cold gas efficiency (%)", "Heat demand of the bed (kW)", "Net efficiency (%)",
          "Tar yield (g/kg, not calibrated)", "Blend (t/y)", "Syngas (10^3 Nm3/y)", "Fossil CO2 (t/y)", "NPV (k€)", "IRR (%)", "LCOH (€/MWh)",
          "GWP100 attributional / net (kg CO2-eq/MJ)", "Abatement cost (€/t CO2-eq)"]
_table(pd.DataFrame({"Quantity": _lab10, "Base case": _c10("base"), "Constrained optimum": _c10("comp")}).set_index("Quantity"),
       "Table 9. Base case and constrained optimum at 209.8 kWth (2025 scenario)")

# ---- Table 10. Monte Carlo -------------------------------------------------------------------------------------
_MC = pd.read_csv(DATA_OUT / "montecarlo_extended.csv")
_pc = lambda g, c, fmt: " / ".join(fmt(v) for v in np.percentile(g[c].dropna(), [5, 50, 95]))
_rows = []
for lab, c_, fmt in [("NPV (k€)", "NPV", lambda v: f"{v:.0f}"), ("IRR (%)", "IRR", _f1), ("LCOH (€/MWh)", "LCOH", _f1)]:
    _rows.append([lab] + [_pc(_MC[_MC.case == k], c_, fmt) for k in ("base", "comp")])
_rows.append(["Probability of NPV > 0 (%)"] + [_f1(100 * (_MC[_MC.case == k].NPV > 0).mean()) for k in ("base", "comp")])
_table(pd.DataFrame(_rows, columns=["Indicator", "Base case", "Constrained optimum"]).set_index("Indicator"),
       f"Table 10. Monte Carlo results ({int((_MC.case == 'base').sum())} samples) at 209.8 kWth: 5th / 50th / 95th percentiles")

# ---- Figures 1-16 ---------------------------------------------------------------------------------------------
# ---- Table 7b. Reference case: tar, carbon distribution and CGE --------------------------------------------------
_c = pd.read_csv(DATA_OUT / "analysis_carbon_tar_char.csv")
_table(pd.DataFrame({"T (°C)": _c["T"].astype(int), "Tar model (g/kg)": _c.tar_g_kg.map(_f1), "Tar measured (g/kg)": _c.tar_meas.map(_f1),
                     "C to gas (%)": _c.C_gas_pct.map(_f1), "C to tar (%)": _c.C_tar_pct.map(_f1), "C to char (%)": _c.C_char_pct.map(_f1),
                     "CGE model (%)": _c.CGE.map(_f1), "CGE measured (%)": _c.CGE_meas.map(_f1)}).set_index("T (°C)"),
       "Table 7b. Reference case: tar yield, carbon distribution (% of fuel C) and cold gas efficiency")
# ---- Table 8b. Conservative techno-economic scenarios of the base case ----------------------------------------------
_s = pd.read_csv(DATA_OUT / "analysis_tea_scenarios.csv", index_col=0)
_table(_s[["NPV", "LCOH"]].apply(lambda c: c.map(lambda v: "" if pd.isna(v) else f"{v:.1f}")).rename(columns={"NPV": "NPV (k€)", "LCOH": "LCOH (€/MWh)"}),
       "Table 8b. Base case under conservative operating assumptions (209.8 kWth, 2025)")
# ---- Table S9b. Formulation B, gas-phase equilibrium, N2-free metrics and H2/CO --------------------------------------
_m = pd.read_csv(DATA_OUT / "analysis_metrics_B_equilibrium.csv", index_col=0)
_table(pd.DataFrame({"n": _m.n, "MAE B (pp)": _m.MAE_B.map(_f2), "MAE equilibrium (pp)": _m.MAE_eq.map(_f2), "MAE B, N2-free (pp)": _m.MAE_B_N2free.map(_f2),
                     "bias CO B, N2-free (pp)": _m.bias_CO_N2free.map(_f1), "H2/CO measured": _m.H2CO_exp.map(_f2), "H2/CO B": _m.H2CO_B.map(_f2)}),
       "Table S9b. Formulation B and gas-phase equilibrium; N2-free metrics and H2/CO ratio")

_FIGS = [("Fig01_framework", "Fig. 1. Structure of the model and of the assessment"),
         ("Fig02_validation", "Fig. 2. Validation of formulation B and comparison with gas-phase equilibrium"), ("Fig03_T_ER_maps", "Fig. 3. Performance over temperature and equivalence ratio"),
         ("Fig04_temperature", "Fig. 4. Effect of bed temperature"), ("Fig05_equivalence_ratio", "Fig. 5. Effect of the equivalence ratio"),
         ("Fig06_RDF_fraction", "Fig. 6. Effect of the RDF fraction at the base conditions"),
         ("Fig07_LCOH_components", "Fig. 7. Components of the levelized cost of heat"), ("Fig08_NPV", "Fig. 8. NPV, plant capacity and measured efficiency"),
         ("Fig09_GWP_attributional", "Fig. 9. Attributional GWP100"), ("Fig10_avoided_landfill", "Fig. 10. Avoided landfill"),
         ("Fig11_integrated", "Fig. 11. Integrated techno-economic and environmental performance"), ("Fig12_Pareto", "Fig. 12. Non-dominated operating conditions"),
         ("Fig13_desirability", "Fig. 13. Energy-environmental optimization"), ("Fig14_MonteCarlo", "Fig. 14. Monte Carlo distributions"),
         ("Fig15_sensitivity", "Fig. 15. One-at-a-time sensitivity"), ("Fig16_scenarios_2025_2030", "Fig. 16. 2025 and 2030 scenarios")]
for f_, cap in _FIGS:
    p_ = FIG25 / f"{f_}.png"
    if _IN_NB:
        _display(_Md(f"**{cap}**")); _display(_Img(filename=str(p_), width=900))
    else:
        print(f"{cap}: {p_}")
print(f"\nFull log of the run: {DATA_OUT / 'full_output.txt'}  |  figures of the article: {FIG25}")
