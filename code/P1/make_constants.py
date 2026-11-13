"""П1.0: таблиця констант data/P1/constants.csv — кожне число з джерелом.

Стовпчики: system, d, quantity, value, err (1σ останніх знаків як у джерелі; '' — точне), status,
source (ключ BibTeX), where (місце в праці), pdf_checked (yes — звірено з PDF у library/; no — PDF немає),
note. status: exact | measured | derived (обчислено тут зі значень джерела; формула в note).
Запуск: .venv/bin/python code/P1/make_constants.py
"""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = []


def row(system, d, q, v, err="", status="measured", src="", where="", pdf="yes", note=""):
    R.append(dict(system=system, d=d, quantity=q, value=v, err=err, status=status, source=src, where=where,
                  pdf_checked=pdf, note=note))


def derived(system, d, q, v, e, src, note):
    row(system, d, q, f"{v:.6g}", f"{e:.2g}", "derived", src, "", "yes", note)


# ---- перколяція, d = 2
row("perc_bond_square", 2, "pc", "0.5", status="exact", src="Stauffer1994", where="табл. порогів", pdf="no",
    note="точний (Кестен 1980); PDF книги немає")
row("perc_site_square", 2, "pc", "0.59274621", "13", src="Newman2001", where="розд. про ймовірність обгортання, R_L^(h)")
row("perc_bond_triangular", 2, "pc", f"{2 * math.sin(math.pi / 18):.10f}", status="exact", src="Stauffer1994",
    where="табл. порогів", pdf="no", note="2 sin(π/18)")
for q, v, s in (("beta", "5/36", 5 / 36), ("nu", "4/3", 4 / 3), ("d_f", "91/48", 91 / 48)):
    row("perc_2D", 2, q, f"{s:.10f}", status="exact", src="Stauffer1994", where="табл. показників", pdf="no", note=v)
row("perc_2D", 2, "d_min", "1.13077", "2", src="Zhou2012", where="анотація")
# ---- d = 3
row("perc_bond_cubic", 3, "pc", "0.24881182", "10", src="Wang2013", where="анотація")
row("perc_site_cubic", 3, "pc", "0.3116077", "2", src="Wang2013", where="анотація")
row("perc_3D", 3, "1/nu", "1.1410", "15", src="Wang2013", where="анотація")
row("perc_3D", 3, "beta/nu", "0.47705", "15", src="Wang2013", where="анотація")
row("perc_3D", 3, "d_min", "1.3756", "3", src="Wang2013", where="анотація", note="Zhou2012: 1.3756(6)")
yt, eyt, bn, ebn = 1.1410, 0.0015, 0.47705, 0.00015
nu = 1 / yt; enu = eyt / yt ** 2
derived("perc_3D", 3, "nu", nu, enu, "Wang2013", "ν = 1/(1/ν)")
derived("perc_3D", 3, "beta", bn * nu, math.hypot(ebn * nu, bn * enu), "Wang2013", "β = (β/ν)·ν")
derived("perc_3D", 3, "d_f", 3 - bn, ebn, "Wang2013", "d_f = d − β/ν")
# ---- d = 3, інші решітки (П1.3b: однакове z, різні короткі петлі)
row("perc_bond_diamond", 3, "pc", "0.3895892", "5", src="Xu2014", where="табл. V", note="z = 4")
row("perc_bond_bcc", 3, "pc", "0.18028762", "20", src="Xu2014", where="табл. V",
    note="z = 8; LorenzZiff1998: 0.1802875(10)")
row("perc_bond_fcc", 3, "pc", "0.12016377", "15", src="Xu2014", where="табл. V",
    note="z = 12; LorenzZiff1998: 0.1201635(10)")
# ---- d = 4, 5 (Zhang2021: bond, періодичні гіперкуби)
for d, (ps, eps, pb, epb), (yt, eyt, df, edf, dm, edm) in (
        (4, ("0.19688561", "3", "0.16013122", "6"), (1.461, 0.005, 3.0446, 0.0007, "1.6042", "5")),
        (5, ("0.14079633", "4", "0.11817145", "3"), (1.743, 0.010, 3.5260, 0.0014, "1.8137", "16"))):
    row(f"perc_site_hypercubic", d, "pc", ps, eps, src="MertensMoore2018", where="табл. II")
    row(f"perc_bond_hypercubic", d, "pc", pb, epb, src="MertensMoore2018", where="табл. II")
    row(f"perc_{d}D", d, "y_t=1/nu", f"{yt}", f"{eyt}", src="Zhang2021", where="анотація")
    row(f"perc_{d}D", d, "d_f", f"{df}", f"{edf}", src="Zhang2021", where="анотація")
    row(f"perc_{d}D", d, "d_min", dm, edm, src="Zhang2021", where="анотація")
    nu = 1 / yt; enu = eyt / yt ** 2
    derived(f"perc_{d}D", d, "nu", nu, enu, "Zhang2021", "ν = 1/y_t")
    derived(f"perc_{d}D", d, "beta", nu * (d - df), math.hypot(enu * (d - df), nu * edf), "Zhang2021",
            "β = ν(d − d_f), гіперскейлінг (d < 6)")
# ---- d = 6 (верхня критична розмірність)
row("perc_site_hypercubic", 6, "pc", "0.109016661", "8", src="MertensMoore2018", where="табл. II")
row("perc_bond_hypercubic", 6, "pc", "0.09420165", "2", src="MertensMoore2018", where="табл. II")
for q, v in (("beta", "1"), ("nu", "0.5"), ("d_f", "4"), ("d_min", "2")):
    row("perc_6D", 6, q, v, status="exact", src="Stauffer1994", pdf="no",
        note="середнє поле; при d = 6 логарифмічні поправки (MertensMoore2018, Grassberger2003)")
# ---- Ізінг і Поттс
row("ising_square", 2, "Kc", f"{math.log(1 + math.sqrt(2)) / 2:.10f}", status="exact", src="Wu1982", pdf="no",
    note="Онсагер, Kc = ln(1+√2)/2; PDF Wu1982 — вручну")
row("ising_2D", 2, "beta", "0.125", status="exact", src="Wu1982", pdf="no", note="1/8")
row("ising_2D", 2, "nu", "1", status="exact", src="Wu1982", pdf="no")
row("ising_sc", 3, "Kc", "0.221654626", "5", src="FerrenbergXuLandau2018", where="анотація")
row("ising_3D", 3, "nu", "0.629912", "86", src="FerrenbergXuLandau2018", where="анотація", note="Монте-Карло")
row("ising_3D", 3, "Delta_sigma", "0.5181489", "10", src="Kos2016", where="анотація / ф-ла (1.1)")
row("ising_3D", 3, "Delta_epsilon", "1.412625", "10", src="Kos2016", where="анотація / ф-ла (1.1)")
ds, eds, de, ede = 0.5181489, 1e-6, 1.412625, 1e-5
nu = 1 / (3 - de); enu = ede * nu ** 2
derived("ising_3D", 3, "nu", nu, enu, "Kos2016", "ν = 1/(3 − Δε) (бутстреп)")
derived("ising_3D", 3, "beta", ds * nu, math.hypot(eds * nu, ds * enu), "Kos2016", "β = Δσ·ν")
for q, b, n in ((3, "1/9", "5/6"), (4, "1/12", "2/3")):
    row(f"potts{q}_square", 2, "Kc", f"{math.log(1 + math.sqrt(q)):.10f}", status="exact", src="Wu1982", pdf="no",
        note="Kc = ln(1+√q) (самодуальність)")
    row(f"potts{q}_2D", 2, "beta", f"{eval(b):.10f}", status="exact", src="Wu1982", pdf="no", note=b)
    row(f"potts{q}_2D", 2, "nu", f"{eval(n):.10f}", status="exact", src="Wu1982", pdf="no",
        note=n + ("; логарифмічні поправки" if q == 4 else ""))
for q in (7, 8):
    row(f"potts{q}_square", 2, "Kc", f"{math.log(1 + math.sqrt(q)):.10f}", status="exact", src="Wu1982", pdf="no",
        note="перехід першого роду; стрибок — Baxter1982 (П1.6)")

out = ROOT / "data" / "P1" / "constants.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(R[0]))
    w.writeheader()
    w.writerows(R)
print(f"{len(R)} рядків → {out.relative_to(ROOT)}")
for r in R:
    if r["status"] == "derived":
        print(f"  {r['system']:14s} {r['quantity']:6s} {r['value']:>10s} ± {r['err']}")
