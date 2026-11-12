# BFB gasifier model

Open-source kinetic model of the air co-gasification of olive pomace and refuse-derived fuel (RDF) in a bubbling fluidized bed (BFB), with techno-economic and carbon-footprint (GWP100) assessment.

> Chuquín-Vasco, D., Lo-Iacono-Ferreira, V. G., Terrados-Cristos, M., Carmo-Calado, L., Torregrosa-López, J. I. Open-source kinetic modeling of olive pomace–RDF co-gasification in a bubbling fluidized bed: validation, techno-economic and carbon-footprint assessment. *Biomass and Bioenergy* (submitted).

## Web app

**https://danichv9.github.io/BFB_MODEL_ELSEVIER/**

Runs the model in the browser, with nothing to install. Enter the feedstock, fuel analyses, operating conditions and reactor data, and get the syngas composition, LHV/HHV, cold gas efficiency, tar, bed energy balance, NPV, IRR, LCOH and GWP100.

## Model

Two-zone bubbling bed (bubble and emulsion phases), isothermal plug-flow freeboard, instantaneous devolatilization with elemental closure, and 26 reactions traced to their primary sources. The gas-phase kinetics are not fitted; the only parameter fitted to gasification data is the activity of the in-bed dolomite (apparent kinetics in the pore-diffusion regime), calibrated on the tar yields of the reference case.

## Contents

| Folder | Content |
|:---|:---|
| `docs/` | Web app (`index.html`) |
| `model/` | Jupyter notebook, the same code as a Python script, reaction network and species (JSON), stored results of the calibrated formulation A |
| `webapp-source/` | Source files used to build the web app |

## Run the notebook

Python ≥ 3.10 with `numpy`, `scipy`, `pandas` and `matplotlib`. Open `model/BFB_modelo_MODEL27_dos_zonas_TEA.ipynb` and run its single cell (about 45 min). It shows the tables and figures of the article.

## Limitations

Tar yields are calibrated only for dolomite beds of the reference case and are indicative elsewhere. Results outside 700–1100 °C and ER 0.20–0.40 are extrapolations.

## Authors

Daniel Chuquín-Vasco ᵃ ᵇ \* · Vanesa G. Lo-Iacono-Ferreira ᵃ · Marta Terrados-Cristos ᶜ · Luís Carmo-Calado ᵈ ᵉ · Juan Ignacio Torregrosa-López ᵃ \*

ᵃ Project Management, Innovation and Sustainability Research Center (PRINS), Universitat Politècnica de València – Alcoy Campus, 03690 Alcoi, Spain
ᵇ Safety and Resources Valorization Research Group (INVAGRO), Universidad Nacional de Chimborazo (UNACH), 060108 Riobamba, Ecuador
ᶜ Project Engineering Department, University of Oviedo, 33004 Oviedo, Spain
ᵈ Polytechnic Institute of Portalegre, Portugal
ᵉ Department of Graphic Engineering Design and Projects, Universidad de Jaén, 23071 Jaén, Spain

\* Corresponding authors: dachuvas@upv.edu.es, jitorreg@iqn.upv.es

## License

MIT — see [`LICENSE`](LICENSE).
