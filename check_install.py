"""Run from the project folder:  python check_install.py
Checks that every file is present, complete (not cut off) and working."""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LAST = {
    "app.py": "st.dataframe(export.head(20))",
    "geoflux_core.py": "return float(1.0 - np.exp(-n_expected))",
    "geoflux_mine.py": "return cat, blast_log.reset_index(drop=True), workings, truth",
    "geoflux_frac.py": "# END OF FILE: geoflux_frac.py",
    "geoflux_io.py": "# END OF FILE: geoflux_io.py",
    "geoflux_dca.py": "# END OF FILE: geoflux_dca.py",
    "geoflux_demo.py": "# END OF FILE: geoflux_demo.py",
    "tabs_seis.py": "# END OF FILE: tabs_seis.py",
    "tabs_prod.py": "# END OF FILE: tabs_prod.py",
    os.path.join("pages", "1_Oil_and_Gas.py"): "# END OF FILE: 1_Oil_and_Gas.py",
}
problems = []
for path, last in LAST.items():
    full = os.path.join(HERE, path)
    if not os.path.exists(full):
        problems.append("MISSING FILE: " + path)
        continue
    text = open(full, encoding="utf-8").read()
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    print("%-28s %4d lines" % (path, len(text.splitlines())))
    try:
        ast.parse(text)
    except SyntaxError as e:
        problems.append("SYNTAX ERROR in %s at line %s: %s" % (path, e.lineno, e.msg))
    if not rows or rows[-1] != last:
        problems.append("CUT OFF: %s stops early (it should end with: %s)" % (path, last))

if not problems:
    try:
        import geoflux_core as core, geoflux_demo as demo, geoflux_dca as dca
        import geoflux_frac as frac, geoflux_mine as mine
        cat, stages, well, fault, _ = demo.make_synthetic_frac()
        cat["Stage"] = frac.assign_stages(cat["Time"], stages, 60)
        mb = core.detect_mag_bin(cat["Magnitude"])
        mc = core.estimate_mc_maxc(cat["Magnitude"], mb)
        b = core.estimate_b_value(cat["Magnitude"], mc, mb)
        summ = frac.stage_summary(cat, stages, mc, b["b"])
        prod = demo.make_synthetic_production(months=24)
        fit = dca.fit_arps(dca.production_time_axis(prod["Date"]), prod["Rate_boed"])
        q = dca.forecast_monthly(fit["qi"], fit["di"], fit["b"], 6.0, 20)
        eco = dca.economics(q, 60.0, 12.5, 8.0, 15000.0, 10e6, 10.0)
        mine.make_synthetic_mine(days=5)
        print("Smoke test: %d stages, hot stage index %.2f, NPV %.1f million" % (
            len(summ), summ["Sigma"].max(), eco["npv"] / 1e6))
    except Exception as exc:
        problems.append("SMOKE TEST FAILED: %r" % (exc,))

print()
if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print("  - " + p)
else:
    print("ALL GOOD. Run:  streamlit run app.py")