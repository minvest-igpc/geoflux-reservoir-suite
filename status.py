import os

HERE = os.path.dirname(os.path.abspath(__file__))
END = "# END OF FILE: "
EXPECT = {
    "app.py": "st.dataframe(export.head(20))",
    "geoflux_core.py": "return float(1.0 - np.exp(-n_expected))",
    "geoflux_mine.py": "return cat, blast_log.reset_index(drop=True), workings, truth",
}
for name in ["geoflux_frac", "geoflux_io", "geoflux_dca", "geoflux_demo", "tabs_seis", "tabs_prod",
             "geoflux_models", "geoflux_backtest", "geoflux_agg", "geoflux_runs", "tabs_models",
             "geoflux_econ", "geoflux_risk", "tabs_econ", "geoflux_wells", "geoflux_typecurve",
             "geoflux_port", "tabs_tc", "tabs_tc2", "tabs_port"]:
    EXPECT[name + ".py"] = END + name + ".py"
for name in ["1_Oil_and_Gas", "2_Decline_Models", "3_Economics_Risk", "4_Type_Curves", "5_Portfolio"]:
    EXPECT[os.path.join("pages", name + ".py")] = END + name + ".py"

bad = 0
for path, last in EXPECT.items():
    full = os.path.join(HERE, path)
    if not os.path.exists(full):
        print("MISSING         " + path)
        bad += 1
        continue
    rows = [r.strip() for r in open(full, encoding="utf-8").read().splitlines() if r.strip()]
    ok = bool(rows) and rows[-1] == last
    print(("OK              " if ok else "OLD or CUT OFF  ") + path)
    bad += 0 if ok else 1
print()
print("ALL FILES OK. Now run:  streamlit run app.py" if bad == 0 else "%d file(s) need attention." % bad)