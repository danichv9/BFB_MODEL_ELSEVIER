# Core of the BFB model used by the web app: sections 2-14, 14d and 15 of BFB_modelo_MODEL27_dos_zonas_TEA.ipynb, unchanged.
import json, warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, nnls
warnings.filterwarnings("ignore")
class _PDStub:
    """Minimal stand-in for pandas (only the literature data tables use it; not needed by simulate())."""
    class DataFrame(dict):
        def __init__(self, d): super().__init__({k: np.asarray(v) for k, v in d.items()})
        def set_index(self, k): return self
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: raise AttributeError(k)
        def __setitem__(self, k, v): super().__setitem__(k, np.asarray(v))
pd = _PDStub()
from pathlib import Path
def _find(fname):
    for p in [Path("data")/fname, Path(fname)]:
        if p.exists(): return p
    raise FileNotFoundError(fname)
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

