# -*- coding: utf-8 -*-
"""
SAĞLAMLIK ANALİZİ: AĞIRLIK DUYARLILIĞI + BASELINE KIYASI + İŞ YÜKÜ
===================================================================
Makalenin hakem itirazlarını peşinen cevaplayan üç analiz:

1. AĞIRLIK DUYARLILIĞI — CHI'den esinlenen önem ağırlıkları keyfi mi?
   Aynı p-Median problemi üç ağırlıklama ile çözülür (CHI, tekdüze sayım,
   CHI-karekök); seçilen konumların örtüşmesi (Jaccard) ve CHI referans
   metriğindeki bozulma raporlanır.

2. BASELINE ÇEŞİTLENDİRME — p-Median naif yaklaşımlardan gerçekten iyi mi?
   Kıyaslananlar: mevcut tesisler (OSM), K-Means ağırlıklı merkezler
   (eski projenin yaklaşımı), en yoğun p hücre (naif hotspot) ve rastgele
   yerleşim (100 tekrar, %95 aralık).

3. İŞ YÜKÜ DAĞILIMI — tek karakol aşırı talep mi topluyor?
   Her çözüm için en-yakın-tesis ataması, karakol başına talep payı,
   Gini katsayısı ve Lorenz eğrileri.

Çalıştırma:  python saglamlik_analizi.py [şehir | all]
Çıktılar:    sonuclar/<şehir>/saglamlik/
"""

import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from polis_optimizasyon import (
    CITY_CONFIGS, CrimeDataLoader, DemandGrid, PMedianSolver,
    GRID_SIZE_M, CAND_GRID_M, RANDOM_SEED,
    C_SURFACE, C_INK, C_INK2, C_BLUE, C_AQUA, C_YELLOW, C_GRAY, C_MUTED,
    set_projection, project_to_meters, evaluate_solution,
    fetch_existing_stations, _style_axes,
)

N_RANDOM_REPS = 100

# Ağırlıklama şemaları: severity sütununa uygulanan dönüşüm
WEIGHT_SCHEMES = {
    'CHI (ana)': lambda s: s,
    'Tekdüze (sayım)': lambda s: np.ones_like(s),
    'CHI-karekök': lambda s: np.sqrt(s),
}


def gini(vals):
    """Gini katsayısı (0 = tam eşit, 1 = tek noktada yoğunlaşmış)."""
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def station_loads(fac_xy, demand_xy, weights):
    """En-yakın-tesis atamasıyla karakol başına toplam talep."""
    from polis_optimizasyon import dist_matrix
    nearest = dist_matrix(demand_xy, fac_xy).argmin(axis=1)
    loads = np.zeros(len(fac_xy))
    np.add.at(loads, nearest, weights)
    return loads


# ---------------------------------------------------------------
# GRAFİKLER
# ---------------------------------------------------------------
def plot_baselines(rows, rand_stats, city_name, out):
    """Baseline kıyası: talep-ağırlıklı ortalama mesafe (düşük = iyi)."""
    colors = {'Mevcut karakollar': C_GRAY, 'p-Median': C_BLUE,
              'K-Means (ağırlıklı)': C_AQUA, 'En yoğun p hücre': C_YELLOW,
              'Rastgele (100 tekrar)': C_MUTED}
    names = [r['method'] for r in rows]
    vals = [r['weighted_mean_km'] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=C_SURFACE)
    _style_axes(ax)
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, height=0.62,
            color=[colors.get(n, C_MUTED) for n in names])
    # rastgele baseline'a %95 aralık çubuğu
    for y, r in zip(ypos, rows):
        if r['method'] == 'Rastgele (100 tekrar)' and rand_stats:
            ax.errorbar(r['weighted_mean_km'], y,
                        xerr=[[r['weighted_mean_km'] - rand_stats['lo']],
                              [rand_stats['hi'] - r['weighted_mean_km']]],
                        fmt='none', ecolor=C_INK2, capsize=3, linewidth=1.2)
        ax.annotate(f"{r['weighted_mean_km']:.2f} km",
                    (r['weighted_mean_km'], y), xytext=(5, 0),
                    textcoords='offset points', va='center',
                    fontsize=9, color=C_INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5, color=C_INK2)
    ax.set_xlabel('Talep-ağırlıklı ort. mesafe (km) — düşük = iyi',
                  color=C_INK2, fontsize=10)
    ax.set_title(f'Baseline kıyası — {city_name} (eşit p)',
                 color=C_INK, fontsize=11, loc='left')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"[GRAFIK] {out}")


def plot_lorenz(load_dict, city_name, out):
    """Karakol iş yükü Lorenz eğrileri (köşegene yakın = dengeli)."""
    colors = {'Mevcut karakollar': C_GRAY, 'p-Median': C_BLUE,
              'K-Means (ağırlıklı)': C_AQUA, 'En yoğun p hücre': C_YELLOW}
    fig, ax = plt.subplots(figsize=(6.2, 5.0), facecolor=C_SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=C_INK2, linewidth=1, linestyle=':')
    for name, loads in load_dict.items():
        v = np.sort(loads)
        cx = np.arange(1, len(v) + 1) / len(v)
        cy = np.cumsum(v) / v.sum()
        g = gini(loads)
        ax.plot(np.concatenate([[0], cx]), np.concatenate([[0], cy]),
                color=colors.get(name, C_MUTED), linewidth=2,
                label=f'{name} (Gini {g:.2f})')
    ax.set_xlabel('Karakolların kümülatif payı', color=C_INK2, fontsize=10)
    ax.set_ylabel('Talebin kümülatif payı', color=C_INK2, fontsize=10)
    ax.set_title(f'İş yükü dengesi (Lorenz) — {city_name}',
                 color=C_INK, fontsize=11, loc='left')
    ax.legend(fontsize=8.5, frameon=False, labelcolor=C_INK2, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"[GRAFIK] {out}")


# ---------------------------------------------------------------
# ANA AKIŞ
# ---------------------------------------------------------------
def main(city='london'):
    cfg = CITY_CONFIGS[city]
    set_projection(cfg.get('epsg', 27700))
    outdir = os.path.join('sonuclar', city, 'saglamlik')
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    print('=' * 70)
    print(f"SAĞLAMLIK ANALİZİ — {cfg['name'].upper()}")
    print('=' * 70)

    df = CrimeDataLoader(cfg).load()
    grid_chi = DemandGrid(df, GRID_SIZE_M)           # CHI referans talebi
    cand_grid = DemandGrid(df, CAND_GRID_M)
    cand_xy = cand_grid.xy                            # tüm şemalarda ortak aday kümesi

    existing = fetch_existing_stations(cfg['bbox'])
    if existing is not None:
        ex_x, ex_y = project_to_meters(existing[:, 1], existing[:, 0])
        existing_xy = np.column_stack([ex_x, ex_y])
        p_star = len(existing)
    else:
        existing_xy = None
        p_star = 25
    print(f"[KURULUM] p* = {p_star}, {grid_chi.n} talep hücresi, "
          f"{len(cand_xy)} aday konum")

    # ---- 1) AĞIRLIK DUYARLILIĞI --------------------------------
    print('\n--- 1) AĞIRLIK DUYARLILIĞI ---')
    scheme_rows, selections = [], {}
    for name, fn in WEIGHT_SCHEMES.items():
        dfx = df.copy()
        dfx['severity'] = fn(df['severity'].values)
        g = DemandGrid(dfx, GRID_SIZE_M)
        pm = PMedianSolver(g.xy, g.weights, cand_xy)
        sel, _ = pm.teitz_bart(p_star, verbose=False)
        selections[name] = set(sel)
        met = evaluate_solution(cand_xy[sel], grid_chi.xy, grid_chi.weights)
        scheme_rows.append({'scheme': name, **met})
        print(f"   {name:18s} CHI-metrikte ort={met['weighted_mean_km']:.4f} km, "
              f"3km kapsama=%{met['coverage_3km_pct']:.2f}")

    base_sel = selections['CHI (ana)']
    base_km = scheme_rows[0]['weighted_mean_km']
    for r, (name, sel) in zip(scheme_rows, selections.items()):
        r['jaccard_vs_chi'] = len(base_sel & sel) / len(base_sel | sel)
        r['chi_metric_degradation_pct'] = 100 * (r['weighted_mean_km'] / base_km - 1)
        if name != 'CHI (ana)':
            print(f"   {name:18s} Jaccard={r['jaccard_vs_chi']:.3f}, "
                  f"CHI-metrik bozulması=%{r['chi_metric_degradation_pct']:+.2f}")
    pd.DataFrame(scheme_rows).to_csv(
        os.path.join(outdir, 'agirlik_duyarliligi.csv'), index=False)

    # ---- 2) BASELINE ÇEŞİTLENDİRME -----------------------------
    print('\n--- 2) BASELINE ÇEŞİTLENDİRME ---')
    solutions = {}   # ad -> tesis koordinatları (iş yükü analizi de kullanır)
    rows = []

    if existing_xy is not None:
        solutions['Mevcut karakollar'] = existing_xy
    solutions['p-Median'] = cand_xy[sorted(base_sel)]

    km = KMeans(n_clusters=p_star, n_init=4, random_state=RANDOM_SEED)
    km.fit(grid_chi.xy, sample_weight=grid_chi.weights)
    solutions['K-Means (ağırlıklı)'] = km.cluster_centers_

    solutions['En yoğun p hücre'] = cand_grid.top_candidates(p_star)

    for name, xy in solutions.items():
        met = evaluate_solution(xy, grid_chi.xy, grid_chi.weights)
        rows.append({'method': name, **met})
        print(f"   {name:22s} ort={met['weighted_mean_km']:.3f} km, "
              f"3km kapsama=%{met['coverage_3km_pct']:.1f}")

    rng = np.random.RandomState(RANDOM_SEED)
    rand_km, rand_cov = [], []
    for _ in range(N_RANDOM_REPS):
        idx = rng.choice(len(cand_xy), size=p_star, replace=False)
        met = evaluate_solution(cand_xy[idx], grid_chi.xy, grid_chi.weights)
        rand_km.append(met['weighted_mean_km'])
        rand_cov.append(met['coverage_3km_pct'])
    rand_stats = {'lo': float(np.percentile(rand_km, 2.5)),
                  'hi': float(np.percentile(rand_km, 97.5))}
    rows.append({'method': 'Rastgele (100 tekrar)',
                 'weighted_mean_km': float(np.mean(rand_km)),
                 'coverage_3km_pct': float(np.mean(rand_cov)),
                 'ci95_km': f"[{rand_stats['lo']:.3f}, {rand_stats['hi']:.3f}]"})
    print(f"   {'Rastgele (100 tekrar)':22s} ort={np.mean(rand_km):.3f} km "
          f"(%95: {rand_stats['lo']:.3f}–{rand_stats['hi']:.3f}), "
          f"3km kapsama=%{np.mean(rand_cov):.1f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, 'baseline_karsilastirmasi.csv'), index=False)
    plot_baselines(rows, rand_stats, cfg['name'],
                   os.path.join(outdir, 'baseline_karsilastirmasi.png'))

    # ---- 3) İŞ YÜKÜ DAĞILIMI -----------------------------------
    print('\n--- 3) İŞ YÜKÜ DAĞILIMI ---')
    load_rows, load_dict = [], {}
    for name, xy in solutions.items():
        loads = station_loads(xy, grid_chi.xy, grid_chi.weights)
        load_dict[name] = loads
        share = loads / loads.sum()
        load_rows.append({
            'method': name, 'n_stations': len(xy),
            'gini': gini(loads),
            'cv': float(loads.std() / loads.mean()),
            'max_share_pct': float(100 * share.max()),
            'top10pct_share_pct': float(
                100 * np.sort(share)[-max(1, len(share) // 10):].sum()),
        })
        print(f"   {name:22s} Gini={load_rows[-1]['gini']:.3f}, "
              f"CV={load_rows[-1]['cv']:.2f}, "
              f"en yüklü karakol=%{load_rows[-1]['max_share_pct']:.1f}")
    pd.DataFrame(load_rows).to_csv(
        os.path.join(outdir, 'is_yuku_dagilimi.csv'), index=False)
    plot_lorenz(load_dict, cfg['name'],
                os.path.join(outdir, 'is_yuku_lorenz.png'))

    print(f"\n[TAMAM] {outdir}/ — süre {time.time() - t0:.0f} s")


if __name__ == '__main__':
    args = sys.argv[1:] or ['london']
    cities = list(CITY_CONFIGS) if args == ['all'] else args
    for c in cities:
        if c not in CITY_CONFIGS:
            print(f"Bilinmeyen şehir: {c} (geçerli: {', '.join(CITY_CONFIGS)})")
            continue
        main(c)
