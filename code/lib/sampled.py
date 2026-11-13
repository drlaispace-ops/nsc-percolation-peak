"""П1.2. Вибірковий оцінювач NSC для великих графів і спряжені реалізації перколяції.

Оцінювач (`nsc_sampled`):
  * компоненти розміром ≤ 2 мають NodeSC = 0 точно (у кулі k = 1 з двох вузлів p = 1, δ = 0);
  * компоненти розміром < s0 (або ≤ M) рахуються точно, з кожного вузла;
  * у більших компонентах беремо M джерел без повернення і оцінюємо середнє NodeSC компоненти;
    NSC = Σ_c N_c·ḡ_c / n, дисперсія Σ_c N_c²·(S_c²/M)(1 − M/N_c) / n²
    (вибірка без повернення, Eppstein–Wang 2004 для BFS-джерел).
Джерела вибираються за фіксованою перестановкою вузлів `perm`: у кожній великій компоненті —
перші M вузлів у порядку perm. Якщо perm спільна для сітки p, джерела спряжені: ті самі вузли,
поки вони лежать у великих компонентах.

Ядро `_sources_sc` рахує NodeSC і профіль δ_k для заданих джерел; робочі масиви int32,
лише два на потік (dist і черга BFS: шар k — суцільний відрізок черги), тож пам'ять ≈ 8·n байт
на потік. Кожне джерело також повертає обсяг роботи (вузли + переглянуті ребра) для моделі
вартості.

Спряжені реалізації (`coupled_bonds`, `at_p`): одне рівномірне число на зв'язок решітки,
граф при p — зв'язки з w < p (стандартна конструкція Ньюмена–Зіффа).
"""
import numba as nb
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg


@nb.njit(parallel=True, cache=True)
def _sources_sc(indptr, indices, sources, K, nthreads):
    """NodeSC, профіль δ_k (k = 1..K, хвіст k > K — в останньому стовпчику) і обсяг роботи
    для кожного джерела. Джерела розподілено між потоками почергово (t, t+T, …)."""
    m = len(sources)
    n = len(indptr) - 1
    sc = np.zeros(m)
    prof = np.zeros((m, K))
    work = np.zeros(m, dtype=np.int64)
    for t in nb.prange(nthreads):
        dist = np.full(n, -1, dtype=np.int32)
        q = np.empty(n, dtype=np.int32)
        for j in range(t, m, nthreads):
            src = sources[j]
            dist[src] = 0
            q[0] = src
            lo, hi = 0, 1          # поточний шар — q[lo:hi]
            s = 1
            b2 = 0
            c = 0.0
            k = 0
            w = 0
            while True:
                k += 1
                top = hi
                for a in range(lo, hi):
                    u = q[a]
                    for e in range(indptr[u], indptr[u + 1]):
                        v = indices[e]
                        if dist[v] == -1:
                            dist[v] = k
                            q[top] = v
                            top += 1
                    w += indptr[u + 1] - indptr[u]
                if top == hi:
                    break
                for a in range(hi, top):
                    v = q[a]
                    for e in range(indptr[v], indptr[v + 1]):
                        d = dist[indices[e]]
                        if d == k - 1:
                            b2 += 2
                        elif d == k:
                            b2 += 1
                s += top - hi
                p = (b2 / 2.0) / (s * (s - 1) / 2.0)
                dl = p * (1.0 - p)
                c += dl
                prof[j, min(k, K) - 1] += dl
                lo, hi = hi, top
            sc[j] = c
            work[j] = w + hi
            for a in range(hi):
                dist[q[a]] = -1
    return sc, prof, work


def csr(A):
    """scipy sparse (будь-яка орієнтація) -> симетрична CSR без петель і кратних ребер, int32."""
    A = sp.csr_array(A)
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A.indptr.astype(np.int64), A.indices.astype(np.int32), A


def node_sc_exact(A, K=64, nthreads=None):
    """NodeSC і профіль усіх вузлів (контроль для вибіркового оцінювача)."""
    ip, ix, _ = csr(A)
    nthreads = nthreads or nb.get_num_threads()
    return _sources_sc(ip, ix, np.arange(len(ip) - 1, dtype=np.int64), K, nthreads)


NBIN = 40                            # біни розмірів кластерів ⌊log₂ N⌋ (2⁴⁰ > будь-якого n)


def nsc_sampled(A, M=500, s0=1000, perm=None, rng=None, K=64, occ=None, nthreads=None, keep_raw=False):
    """Вибіркова NSC з розкладом за кластерами.

    A    — граф (scipy sparse); occ — маска зайнятих для site-перколяції (None — усі зайняті);
    M    — кількість джерел у великій компоненті; s0 — компоненти менші за s0 рахуються точно;
    perm — перестановка вузлів (для спряжених джерел); інакше з rng.
    Повертає dict: nsc, se (на всі вузли), nsc_occ, se_occ, P (частка найбільшої компоненти серед
    усіх вузлів), g_giant, se_giant, g_fin, se_fin, delta (середній профіль на всі вузли),
    кількості й обсяг роботи; chi = Σ' N_c²/n (середній розмір скінченного кластера, без гіганта;
    chi_occ — на зайняті); журнал скінченних кластерів за розміром у бінах ⌊log₂ N_c⌋ (NBIN бінів):
    sbin_c (кластерів), sbin_n (вузлів), sbin_sc (Σ NodeSC), sbin_var (дисперсія Σ NodeSC);
    з keep_raw — сирі NodeSC і профілі джерел великих компонент, raw_giant_sc — джерела гіганта.
    """
    ip, ix, A = csr(A)
    n = len(ip) - 1
    nthreads = nthreads or nb.get_num_threads()
    if perm is None:
        perm = (rng or np.random.default_rng()).permutation(n)
    rank = np.empty(n, np.int64)
    rank[perm] = np.arange(n)
    ncomp, lab = cg.connected_components(A, directed=False)
    size = np.bincount(lab, minlength=ncomp)
    occ = np.ones(n, bool) if occ is None else np.asarray(occ, bool)
    gi = int(np.argmax(np.where(np.bincount(lab[occ], minlength=ncomp) > 0, size, -1)))
    big = (size >= s0) & (size > M)
    trivial = size <= 2
    node_big = big[lab]
    exact_nodes = np.flatnonzero(~node_big & ~trivial[lab])
    cand = np.flatnonzero(node_big)
    order = cand[np.lexsort((rank[cand], lab[cand]))]
    if len(order):
        lab_o = lab[order]
        start = np.r_[0, np.flatnonzero(np.diff(lab_o)) + 1]
        pos = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
        samp = order[pos < M]
    else:
        samp = np.empty(0, np.int64)
    src = np.concatenate([exact_nodes, samp]).astype(np.int64)
    sc, prof, work = _sources_sc(ip, ix, src, K, nthreads)
    ne = len(exact_nodes)
    # точна частина
    tot = np.zeros(ncomp)            # Σ NodeSC компоненти (оцінка)
    var = np.zeros(ncomp)            # дисперсія цієї суми
    np.add.at(tot, lab[exact_nodes], sc[:ne])
    dsum = np.zeros((ncomp, K))
    np.add.at(dsum, lab[exact_nodes], prof[:ne])
    # вибіркова частина
    ls = lab[samp]
    bc = np.flatnonzero(big)
    big_W = []                       # обсяг роботи BFS за кожною великою компонентою (модель вартості)
    for c in bc:
        sel = np.flatnonzero(ls == c) + ne
        x = sc[sel]
        m = len(x)
        N = size[c]
        tot[c] = N * x.mean()
        var[c] = N ** 2 * x.var(ddof=1) / m * (1 - m / N)
        dsum[c] = N * prof[sel].mean(0)
        big_W.append(int(work[sel].sum()))
    n_occ = int(occ.sum())
    isg = np.arange(ncomp) == gi
    # скінченні кластери: компоненти із зайнятими вузлами, крім гіганта (порожні вузли site — не кластери)
    fin = (np.bincount(lab[occ], minlength=ncomp) > 0) & ~isg
    sb = np.minimum(np.log2(np.maximum(size, 1)).astype(np.int64), NBIN - 1)[fin]
    sbin = dict(sbin_c=np.bincount(sb, minlength=NBIN), sbin_n=np.bincount(sb, size[fin], NBIN),
                sbin_sc=np.bincount(sb, tot[fin], NBIN), sbin_var=np.bincount(sb, var[fin], NBIN))
    chi2 = float(np.sum(size[fin].astype(float) ** 2))
    Ng = size[gi]
    S, V = tot.sum(), var.sum()
    Sf, Vf = tot[~isg].sum(), var[~isg].sum()
    nfin = n_occ - Ng
    out = dict(nsc=S / n, se=np.sqrt(V) / n,
               nsc_occ=S / n_occ if n_occ else 0.0, se_occ=np.sqrt(V) / n_occ if n_occ else 0.0,
               P=Ng / n, P_occ=Ng / n_occ if n_occ else 0.0,
               g_giant=tot[gi] / Ng, se_giant=np.sqrt(var[gi]) / Ng,
               g_fin=Sf / nfin if nfin > 0 else 0.0, se_fin=np.sqrt(Vf) / nfin if nfin > 0 else 0.0,
               delta=dsum.sum(0) / n, n=n, n_occ=n_occ, n_big=int(big.sum()), n_src_exact=ne,
               n_src_sampled=len(samp), work=int(work.sum()), work_exact=int(work[:ne].sum()),
               work_sampled=int(work[ne:].sum()), big_N=size[bc].tolist(), big_W=big_W,
               chi=chi2 / n, chi_occ=chi2 / n_occ if n_occ else 0.0, giant_sampled=bool(big[gi]), **sbin)
    if keep_raw:
        out.update(raw_comp=ls, raw_sc=sc[ne:], raw_prof=prof[ne:], raw_giant_sc=sc[ne:][ls == gi])
    return out


# ------------------------------------------------------------------ спряжені реалізації
def coupled_bonds(L, d, rng, kind="bond"):
    """Усі зв'язки Z_L^d з періодичними межами і рівномірні числа на них.
    kind="bond": число на зв'язок; kind="site": число на вузол (зв'язок зайнятий, якщо обидва
    кінці зайняті). Повертає (n, u, v, w) — граф при p: зв'язки з w < p (`at_p`)."""
    n = L ** d
    idx = np.arange(n, dtype=np.int64).reshape((L,) * d)
    u = np.concatenate([idx.ravel()] * d)
    v = np.concatenate([np.roll(idx, -1, axis=ax).ravel() for ax in range(d)])
    if kind == "bond":
        return n, u, v, rng.random(len(u)), None
    ws = rng.random(n)
    return n, u, v, np.maximum(ws[u], ws[v]), ws


def coupled_triangular(L, rng):
    """Трикутна решітка L×L (квадратна + одна діагональ, як perc.triangular_bond), bond, спряжена."""
    n = L * L
    idx = np.arange(n, dtype=np.int64).reshape(L, L)
    u = np.concatenate([idx.ravel()] * 3)
    v = np.concatenate([np.roll(idx, sh, axis=(0, 1)).ravel() for sh in ((-1, 0), (0, -1), (-1, -1))])
    return n, u, v, rng.random(len(u)), None


def at_p(n, u, v, w, p, ws=None):
    """Граф (і маска зайнятих для site) при значенні p зі спряженої реалізації."""
    k = w < p
    A = sp.coo_array((np.ones(int(k.sum()), np.int8), (u[k], v[k])), shape=(n, n)).tocsr()
    return A, (None if ws is None else ws < p)
