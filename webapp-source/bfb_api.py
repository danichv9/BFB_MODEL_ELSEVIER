# Web interface of the BFB model: builds the configuration from the inputs of the page, runs simulate() (formulation B,
# two-zone bed) and the integrated techno-economic and carbon-footprint evaluation (section 23 of the notebook), and
# returns the results as JSON.
import json, math
import numpy as np
from dataclasses import replace
from bfb_core import (FeedstockConfig, ReactorConfig, OperatingPoint, OLIVE_POMACE, RDF, DOLOMITE_ORIO,
                      simulate, stoich_air_Nm3_per_kg, tea_lca, bed_surplus_W, KPS, OPT_H1, P,
                      IDX, MW, GAS, TAR_SPECIES, PERMANENT_GAS, ATOM, MOL_PER_NM3)

HHV_NM3 = {"H2": 12.75, "CO": 12.63, "CH4": 39.82, "C2H4": 63.40, "C3H6": 91.80}


def _f(x):
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def _component(base, d):
    ult = dict(C=d["C"], H=d["H"], N=d["N"], S=d["S"], O=d["O"], Ash=d["Ash"])
    return replace(base, ultimate_dry={k: float(v) for k, v in ult.items()}, moisture_ar=float(d["M"]),
                   VM_ar=float(d["VM"]), FC_ar=float(d["FC"]), ash_ar=float(d["ash_ar"]), HHV_MJ_kg=float(d["HHV"]))


LF = dict(DOCf=0.5, MCF=1.0, F=0.5, OX=0.1, R=0.5)   # IPCC 2019 first-order decay defaults (section 22 of the notebook)


def _landfill_EF(rdf, f_fossil, gwp_ch4_bio):
    c_wet = rdf.ultimate_dry["C"] / sum(rdf.ultimate_dry.values()) * (1 - rdf.moisture_ar / 100.0)
    lf = dict(LF, DOC=(1 - f_fossil) * c_wet)
    ch4_gen = 1000.0 * lf["DOC"] * lf["DOCf"] * lf["MCF"] * lf["F"] * 16.0 / 12.0
    return ch4_gen * (1 - lf["R"]) * (1 - lf["OX"]) * gwp_ch4_bio     # biogenic CH4


def _tl3(sim, fs, rdf, p):
    """Integrated techno-economic and carbon-footprint evaluation (section 23 of the notebook): NPV, IRR, payback, LCOH and
    its parts, GWP100 and CO2 abatement cost."""
    Pd = p["P_design_kWth"]
    pp = dict(p, P_ref_kWth=209.8, capex_exp=0.6 if abs(Pd - 209.8) > 1e-9 else 1.0)
    t = tea_lca(sim, fs, pp)
    heat = t["heat_MWh_y"]; MJ = heat * 3600.0
    i, N = pp["discount_rate"], int(pp["lifetime"]); crf = i * (1 + i) ** N / ((1 + i) ** N - 1)
    capex = t["CAPEX_kEUR"] * 1e3; feed = t["feed_kg_h"]; w = fs.mass_fractions
    rdf_t = feed * w.get("RDF", 0) * pp["hours"] / 1000.0; pom_t = feed * w.get("olive_pomace", 0) * pp["hours"] / 1000.0
    elec = pp["elec_kWh_per_MJ"] * MJ / 1000.0
    fos_t = (t["GWP_kg_MJ"] - pp["CH4_slip"] * pp["GWP_CH4"] - pp["elec_kWh_per_MJ"] * pp["grid_EF"]) * MJ / 1000.0
    comp = dict(capex=capex * crf / heat, om=pp["om_frac"] * capex / heat, pomace=pom_t * pp["pomace_cost"] / heat,
                elec=elec * pp["elec_price"] / heat, co2=fos_t * pp["ets_price"] / heat,
                rdf=-rdf_t * pp["rdf_avoided_cost"] / heat)
    lcoh = sum(comp.values())
    lf = _landfill_EF(rdf, pp["rdf_fossil_C_frac"], pp["GWP_CH4_biogenic"])
    gwp = t["GWP_kg_MJ"]; gwp_net = gwp - rdf_t * lf / MJ
    gwp_parts = dict(fossil=fos_t * 1000 / MJ, ch4=pp["CH4_slip"] * pp["GWP_CH4"],
                     elec=pp["elec_kWh_per_MJ"] * pp["grid_EF"], landfill=-rdf_t * lf / MJ)
    den = (pp["NG_reference"] - gwp_net) * 3.6
    abat = (lcoh - t["heat_price"]) / den if abs(den) > 1e-12 else float("nan")
    return dict(NPV=t["NPV_kEUR"], IRR=t["IRR_pct"], PB=t["PB_y"], LCOH=lcoh, p_heat=t["heat_price"], GWP=gwp,
                GWP_net=gwp_net, GWP_vs_NG=t["GWP_vs_NG_pct"], abat=abat, comp=comp, gwp_parts=gwp_parts, lf=lf,
                heat=heat, feed_kg_h=feed, rdf_t=rdf_t, pom_t=pom_t, CAPEX=capex / 1e3, rev=t["rev_kEUR_y"],
                opex=t["opex_kEUR_y"], Q_ext_kW=t["Q_ext_kW"], eff_net=t["eff_net_pct"], fossil_C=t["fossil_C_share"])


def run(inp_json):
    inp = json.loads(inp_json)
    fd, opd, rd, ec = inp["feed"], inp["op"], inp["reactor"], inp["econ"]
    op_c, rdf_c = _component(OLIVE_POMACE, fd["op"]), _component(RDF, fd["rdf"])
    x = float(fd["x_rdf"])
    comps, fr = {}, {}
    if x < 1.0: comps["olive_pomace"] = op_c; fr["olive_pomace"] = 1.0 - x
    if x > 0.0: comps["RDF"] = rdf_c; fr["RDF"] = x
    fs = FeedstockConfig(comps, fr, f"OP{100 * (1 - x):.0f}_RDF{100 * x:.0f}")

    mat = rd["material"]
    rc = ReactorConfig(float(rd["D"]), float(rd["H"]), float(rd["W"]), mat, float(rd["dp_um"]) * 1e-6,
                       float(rd["rho_p"]), float(rd["eps_mf"]), int(rd["n_or"]), float(rd["tau_char"]),
                       "user_BFB", tar_catalyst=DOLOMITE_ORIO if mat == "dolomite" else None)
    feed = float(opd["feed"])
    air_st = stoich_air_Nm3_per_kg(fs)
    air = float(opd["ER"]) * air_st * feed if opd["air_mode"] == "ER" else float(opd["air"])
    op = OperatingPoint(float(opd["T"]), float(opd["P"]), feed, air, float(opd.get("air_T", 25.0)))
    sim = simulate(fs, rc, op, KPS, OPT_H1)

    p = dict(P)
    for k, v in ec.items():
        p[k] = float(v)
    tl = _tl3(sim, fs, rdf_c, p)

    y = sim["y"]; F = sim["F_out"]; hyd = sim["hyd"]; bed = sim["bed"]
    hhv = sum(HHV_NM3[k] * y.get(k, 0.0) for k in HHV_NM3) / 100.0
    tars = {s: F[IDX[s]] * MW[IDX[s]] * 3600.0 / feed for s in TAR_SPECIES}       # g per kg fuel
    Q_bed_kW = bed_surplus_W(sim, float(opd.get("air_T", 25.0))) / 1e3
    P_gas_kW = sim["Q_dry_Nm3_h"] * sim["LHV"] / 3.6
    warn = []
    T = op.T_C; ER = sim["ER"]
    if not (700 <= T <= 1100): warn.append(f"Bed temperature {T:.0f} °C is outside the validated range (700-1100 °C).")
    if not (0.20 <= ER <= 0.40): warn.append(f"ER = {ER:.3f} is outside the explored range (0.20-0.40); validated window is 0.26-0.29.")
    if not (0.10 <= x <= 0.50) and x not in (0.0,): warn.append(f"RDF fraction {100 * x:.0f} % is outside the explored range (10-50 %).")
    if hyd["U_Umf"] <= 1.0: warn.append("U0/Umf ≤ 1: the bed is not fluidized at these conditions.")
    if bed.get("d_b", 0) > 0.6 * rc.D_m: warn.append("Bubble diameter > 0.6 D: slugging regime, outside the bubbling-bed correlations.")
    if hyd["H_fb"] <= 0: warn.append("Expanded bed is taller than the reactor: no freeboard left.")
    cl = sim["closure"]
    if max(abs(v - 100) for v in cl.values()) > 1.0: warn.append("Elemental closure deviates more than 1 % (check inputs).")

    out = dict(
        y={k: _f(v) for k, v in y.items()},
        y_wet={k: _f(v) for k, v in sim["y_wet"].items()},
        y_nfree={k: _f(100 * v / (100 - y["N2"])) for k, v in y.items() if k != "N2"},
        Q_dry=_f(sim["Q_dry_Nm3_h"]), gas_yield=_f(sim["Q_dry_Nm3_h"] / feed), LHV=_f(sim["LHV"]), HHV=_f(hhv),
        CGE=_f(sim["CGE"]), LHV_fuel=_f(sim["LHV_fuel"]), tar_g_kg=_f(sim["tar_g_kg"]), tar_g_Nm3=_f(sim["tar_g_Nm3"]),
        tar_nb_g_kg=_f(sim["tar_nb_g_kg"]), tars={k: _f(v) for k, v in tars.items()}, char_g_kg=_f(sim["char_g_kg"]),
        C_gas=_f(sim["C_gas_pct"]), FC_conv=_f(sim["FC_conv"]), O2_conv=_f(sim["O2_conv"]), ER=_f(ER),
        air=_f(air), air_st=_f(air_st), closure={k: _f(v) for k, v in cl.items()},
        hyd=dict(U0=_f(hyd["U0"]), Umf=_f(hyd["Umf"]), U_Umf=_f(hyd["U_Umf"]), H_mf=_f(hyd["H_mf"]), H_f=_f(hyd["H_f"]),
                 H_fb=_f(hyd["H_fb"]), delta=_f(bed.get("delta", float("nan"))), d_b=_f(bed.get("d_b", float("nan"))),
                 tau_bed=_f(bed.get("tau_gas", float("nan"))), tau_fb=_f(sim["tau_fb"]),
                 Kbe=_f(bed.get("K_be_mean", float("nan"))), phi_e=_f(np.mean(bed["phi_e"]) if "phi_e" in bed else float("nan"))),
        energy=dict(Q_bed_kW=_f(Q_bed_kW), Q_bed_MJ_kg=_f(Q_bed_kW * 3.6 / feed), P_gas_kW=_f(P_gas_kW),
                    P_fuel_kW=_f(feed * sim["LHV_fuel"] / 3.6)),
        tea={k: (_f(v) if not isinstance(v, dict) else {a: _f(b) for a, b in v.items()}) for k, v in tl.items()},
        warnings=warn)
    return json.dumps(out)

