# Builds docs/index.html: inserts the model core, its JSON interface, the reaction network, the species registry and the
# logos into template.html. Run from this folder.
import os, base64

t = open("template.html", encoding="utf-8").read()
rep = {"__PY_CORE__": open("bfb_core.py", encoding="utf-8").read(),
       "__PY_API__": open("bfb_api.py", encoding="utf-8").read(),
       "__NET_JSON__": open("../model/active_network.json", encoding="utf-8").read(),
       "__REG_JSON__": open("../model/species_registry.json", encoding="utf-8").read()}


def _logo(fn):
    return "data:image/png;base64," + base64.b64encode(open(fn, "rb").read()).decode()


rep["__LOGO_UPV__"] = _logo("logo_upv.png")
rep["__LOGO_PRINS__"] = _logo("logo_prins.png")
for k, v in rep.items():
    assert "</script" not in v.lower(), k
    t = t.replace(k, v)
open("../docs/index.html", "w", encoding="utf-8").write(t)
print(f"{len(t) / 1024:.0f} KB")
