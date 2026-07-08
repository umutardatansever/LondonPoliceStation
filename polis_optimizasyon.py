# -*- coding: utf-8 -*-
"""
LONDRA POLİS KARAKOLU YERLEŞİM VE DEVRİYE ROTASI OPTİMİZASYONU
===============================================================
Suç-talep ağırlıklı tesis yerleşim problemi (p-median + MCLP) ve
gerçek yol ağı üzerinde devriye turu (Christofides TSP yaklaşımı).

Bilimsel katkılar (Q1/Q2 hedefli tasarım):
  1. Suç önem derecesi (Crime Harm Index'ten esinlenen ağırlıklar) ile
     ağırlıklandırılmış talep modeli (500 m ızgara, EPSG:27700 projeksiyonu).
  2. p-median problemi: Teitz-Bart yerel arama sezgiseli + PuLP/CBC ile
     kesin MILP çözümü (optimalite açığı raporlanır).
  3. MCLP (Maximal Covering Location Problem): açgözlü (1-1/e garantili)
     + kesin MILP çözümü.
  4. OpenStreetMap'ten alınan GERÇEK mevcut polis karakolu konumlarıyla
     nicel kıyas (talep-ağırlıklı ortalama mesafe, kapsama oranları).
  5. Zamansal dış-örneklem doğrulama: ilk aylarda eğitilen yerleşimin
     sonraki aylardaki performansı (kararlılık analizi).
  6. p duyarlılık analizi (karakol sayısı - hizmet kalitesi ödünleşimi).
  7. Devriye rotası: gerçek yol ağında MST alt sınırı + Christofides
     TSP turu (kapalı devriye).

Çalıştırma:  python polis_optimizasyon.py
Girdi:       london_crime.csv
"""

import glob as globmod
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import folium
from folium import plugins

from pyproj import Transformer

warnings.filterwarnings('ignore')

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
    ox.settings.use_cache = True
    ox.settings.log_console = False
except ImportError:
    OSMNX_AVAILABLE = False
    import networkx as nx

# ---------------------------------------------------------------
# SABİTLER
# ---------------------------------------------------------------
# Şehir yapılandırmaları. data_glob: veri dosyası deseni (çoklu aylık CSV
# desteklenir); bbox: coğrafi aykırı değer filtresi. Tüm Britanya şehirleri
# için EPSG:27700 projeksiyonu geçerlidir.
CITY_CONFIGS = {
    'london': dict(
        name='Londra',
        bbox=dict(lat_min=51.28, lat_max=51.70, lon_min=-0.55, lon_max=0.35),
        data_glob=['data/london/*.csv'],
        epsg=27700,   # British National Grid
        schema='uk',
    ),
    'west-midlands': dict(
        name='West Midlands (Birmingham)',
        bbox=dict(lat_min=52.30, lat_max=52.70, lon_min=-2.25, lon_max=-1.55),
        data_glob=['data/west-midlands/*.csv'],
        epsg=27700,
        schema='uk',
    ),
    'chicago': dict(
        name='Chicago',
        bbox=dict(lat_min=41.62, lat_max=42.05, lon_min=-87.95, lon_max=-87.50),
        data_glob=['data/chicago/*.csv'],
        epsg=26916,   # UTM 16N (metre)
        schema='chicago',
    ),
}
GRID_SIZE_M = 500          # talep ızgarası çözünürlüğü (metre)
CAND_GRID_M = 1000         # aday tesis ızgarası (tüm şehre yayılı)
COARSE_GRID_M = 1500       # kesin MILP için kaba talep ızgarası
N_CANDIDATES_EXACT = 150   # kesin MILP'de aday sayısı
COVERAGE_RADIUS_M = 3000   # MCLP kapsama yarıçapı (3 km)
COVERAGE_LEVELS_M = [1000, 2000, 3000, 5000]
P_SENSITIVITY = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
PATROL_P = 20              # devriye senaryosunda karakol sayısı
MILP_TIME_LIMIT = 240      # saniye
RANDOM_SEED = 42

# Cambridge Crime Harm Index'ten esinlenen göreli önem ağırlıkları.
# Mutlak CHI gün-değerleri değil; kategoriler arası göreli sıralamayı korur.
CRIME_SEVERITY = {
    'Violence and sexual offences': 10.0,
    'Robbery': 8.0,
    'Possession of weapons': 8.0,
    'Burglary': 5.0,
    'Theft from the person': 4.0,
    'Vehicle crime': 3.0,
    'Criminal damage and arson': 3.0,
    'Public order': 2.0,
    'Drugs': 2.0,
    'Other theft': 2.0,
    'Shoplifting': 1.0,
    'Bicycle theft': 1.0,
    'Anti-social behaviour': 1.0,
    'Other crime': 1.0,
}
DEFAULT_SEVERITY = 1.0

# Chicago (Illinois UCR 'primary_type') için aynı göreli ölçeğe eşlenmiş
# önem ağırlıkları — Birleşik Krallık kategorileriyle hizalı tutulmuştur.
CRIME_SEVERITY_CHICAGO = {
    'HOMICIDE': 10.0,
    'CRIMINAL SEXUAL ASSAULT': 10.0,
    'CRIM SEXUAL ASSAULT': 10.0,
    'ASSAULT': 10.0,
    'BATTERY': 10.0,
    'KIDNAPPING': 10.0,
    'ROBBERY': 8.0,
    'WEAPONS VIOLATION': 8.0,
    'SEX OFFENSE': 8.0,
    'HUMAN TRAFFICKING': 10.0,
    'BURGLARY': 5.0,
    'MOTOR VEHICLE THEFT': 3.0,
    'CRIMINAL DAMAGE': 3.0,
    'ARSON': 3.0,
    'THEFT': 2.0,
    'NARCOTICS': 2.0,
    'PUBLIC PEACE VIOLATION': 2.0,
    'OFFENSE INVOLVING CHILDREN': 8.0,
    'STALKING': 4.0,
    'INTIMIDATION': 4.0,
    'CRIMINAL TRESPASS': 1.0,
    'DECEPTIVE PRACTICE': 1.0,
    'PROSTITUTION': 1.0,
    'GAMBLING': 1.0,
    'LIQUOR LAW VIOLATION': 1.0,
    'INTERFERENCE WITH PUBLIC OFFICER': 1.0,
    'OBSCENITY': 1.0,
}

# Grafik paleti (dataviz referans paleti, açık tema)
C_SURFACE = '#fcfcfb'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_BLUE = '#2a78d6'    # p-median / birincil seri
C_AQUA = '#1baf7a'    # MCLP / ikincil seri
C_YELLOW = '#eda100'
C_GRAY = '#898781'    # mevcut durum (baz çizgi)

# WGS84 <-> şehre uygun metrik projeksiyon (main içinde set_projection ile
# şehir yapılandırmasındaki EPSG koduna göre kurulur; varsayılan: Britanya)
_TO_METERS = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True)
_TO_WGS = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True)


def set_projection(epsg):
    global _TO_METERS, _TO_WGS
    _TO_METERS = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    _TO_WGS = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)


def project_to_meters(lon, lat):
    x, y = _TO_METERS.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def unproject_to_wgs(x, y):
    lon, lat = _TO_WGS.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


# ---------------------------------------------------------------
# 1. VERİ YÜKLEME
# ---------------------------------------------------------------
class CrimeDataLoader:
    """police.uk biçimli CSV'ler -> temiz, projeksiyonlu, önem-ağırlıklı
    nokta verisi. Birden çok aylık dosya (data/<şehir>/*.csv) desteklenir."""

    def __init__(self, city_cfg):
        self.cfg = city_cfg
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.files = []
        for pattern in city_cfg['data_glob']:
            hits = sorted(globmod.glob(pattern)) or \
                sorted(globmod.glob(os.path.join(script_dir, pattern)))
            if hits:
                self.files = hits
                break

    def load(self):
        if not self.files:
            raise FileNotFoundError(
                f"{self.cfg['name']} için veri bulunamadı "
                f"(aranan desenler: {self.cfg['data_glob']})")
        print(f"[VERI] {len(self.files)} CSV dosyası yükleniyor "
              f"({os.path.basename(self.files[0])} .. "
              f"{os.path.basename(self.files[-1])})")
        schema = self.cfg.get('schema', 'uk')
        if schema == 'chicago':
            parts = [pd.read_csv(f, usecols=['date', 'primary_type',
                                             'latitude', 'longitude'])
                     for f in self.files]
            df = pd.concat(parts, ignore_index=True)
            n_raw = len(df)
            df = df.dropna(subset=['latitude', 'longitude'])
            df['month'] = df['date'].str[:7]
            df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon',
                                    'primary_type': 'type'})
            severity_map = CRIME_SEVERITY_CHICAGO
        else:  # police.uk şeması
            parts = [pd.read_csv(f, usecols=['Month', 'Longitude', 'Latitude',
                                             'Crime type'])
                     for f in self.files]
            df = pd.concat(parts, ignore_index=True)
            n_raw = len(df)
            df = df.dropna(subset=['Latitude', 'Longitude'])
            df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon',
                                    'Crime type': 'type', 'Month': 'month'})
            severity_map = CRIME_SEVERITY
        b = self.cfg['bbox']
        df = df[(df['lat'] > b['lat_min']) & (df['lat'] < b['lat_max']) &
                (df['lon'] > b['lon_min']) & (df['lon'] < b['lon_max'])].copy()

        df['severity'] = df['type'].map(severity_map).fillna(DEFAULT_SEVERITY)
        x, y = project_to_meters(df['lon'].values, df['lat'].values)
        df['x'], df['y'] = x, y

        print(f"[VERI] Ham kayıt: {n_raw} | {self.cfg['name']} içi geçerli kayıt: "
              f"{len(df)} (örnekleme YOK, tam veri)")
        print(f"[VERI] Ay aralığı: {df['month'].min()} .. {df['month'].max()} "
              f"({df['month'].nunique()} ay) | Suç türü sayısı: {df['type'].nunique()}")
        return df


# ---------------------------------------------------------------
# 2. TALEP IZGARASI MODELİ
# ---------------------------------------------------------------
class DemandGrid:
    """Suç noktalarını sabit boyutlu ızgara hücrelerinde toplulaştırır.
    Talep düğümü = hücredeki suçların ağırlıklı ortalama konumu,
    talep ağırlığı = hücredeki toplam önem (severity) puanı."""

    def __init__(self, df, cell_m=GRID_SIZE_M):
        self.cell_m = cell_m
        gx = np.floor(df['x'].values / cell_m).astype(np.int64)
        gy = np.floor(df['y'].values / cell_m).astype(np.int64)
        key = gx * 10_000_000 + gy

        agg = pd.DataFrame({
            'key': key,
            'w': df['severity'].values,
            'wx': df['severity'].values * df['x'].values,
            'wy': df['severity'].values * df['y'].values,
            'cnt': 1,
        }).groupby('key').sum()

        self.weights = agg['w'].values.astype(float)
        self.counts = agg['cnt'].values
        self.xy = np.column_stack([agg['wx'].values / agg['w'].values,
                                   agg['wy'].values / agg['w'].values])
        self.n = len(self.weights)

    def top_candidates(self, m):
        """En yüksek talepli m hücrenin merkezleri -> aday tesis konumları."""
        idx = np.argsort(-self.weights)[:m]
        return self.xy[idx]


def dist_matrix(a_xy, b_xy):
    """Öklid mesafe matrisi (metre). Projeksiyonlu koordinatlarda Öklid,
    Haversine büyük-daire mesafesine ~%0.1 hata ile eşdeğerdir.
    float32: bellek/hız için (metre ölçeğinde hassasiyet kaybı < 1 m)."""
    a = a_xy.astype(np.float32)
    b = b_xy.astype(np.float32)
    d = a[:, None, :] - b[None, :, :]
    return np.sqrt((d ** 2).sum(axis=2))


def evaluate_solution(fac_xy, demand_xy, weights):
    """Bir tesis kümesinin hizmet kalitesi metrikleri."""
    D = dist_matrix(demand_xy, fac_xy)
    nearest = D.min(axis=1)
    wsum = weights.sum()
    out = {
        'weighted_mean_km': float((weights * nearest).sum() / wsum / 1000.0),
        'max_km': float(nearest.max() / 1000.0),
        'median_km': float(np.median(nearest) / 1000.0),
    }
    for r in COVERAGE_LEVELS_M:
        out[f'coverage_{r // 1000}km_pct'] = float(
            100.0 * weights[nearest <= r].sum() / wsum)
    return out


# ---------------------------------------------------------------
# 3. p-MEDIAN ÇÖZÜCÜLERİ
# ---------------------------------------------------------------
class PMedianSolver:
    """min  sum_i w_i * min_{j in S} d_ij ,  |S| = p
    Açgözlü kurulum + Teitz-Bart (vertex substitution) yerel arama;
    isteğe bağlı PuLP/CBC ile kesin MILP doğrulaması."""

    def __init__(self, demand_xy, weights, cand_xy):
        self.demand_xy = demand_xy
        self.w = weights.astype(np.float32)
        self.cand_xy = cand_xy
        self.D = dist_matrix(demand_xy, cand_xy)  # N x M
        self.N, self.M = self.D.shape

    def _objective(self, sel):
        return float((self.w * self.D[:, sel].min(axis=1)).sum())

    def greedy(self, p):
        sel = []
        best_d = np.full(self.N, np.inf)
        for _ in range(p):
            # her aday için: eklenirse yeni toplam maliyet
            cand_cost = (self.w[:, None] * np.minimum(best_d[:, None], self.D)).sum(axis=0)
            cand_cost[sel] = np.inf
            j = int(np.argmin(cand_cost))
            sel.append(j)
            best_d = np.minimum(best_d, self.D[:, j])
        return sel

    def teitz_bart(self, p, max_pass=8, verbose=True):
        t0 = time.time()
        sel = self.greedy(p)
        obj = self._objective(sel)
        for pas in range(max_pass):
            improved = False
            sub = self.D[:, sel]                       # N x p
            order = np.argsort(sub, axis=1)
            d1 = sub[np.arange(self.N), order[:, 0]]   # en yakın
            d2 = sub[np.arange(self.N), order[:, 1]]   # ikinci en yakın
            nearest_pos = order[:, 0]
            for pos in range(p):
                # sel[pos] çıkarıldığında her düğümün kalan en yakın mesafesi
                base = np.where(nearest_pos == pos, d2, d1)
                # tüm adaylar için yeni maliyet (vektörel)
                new_costs = (self.w[:, None] * np.minimum(base[:, None], self.D)).sum(axis=0)
                new_costs[sel] = np.inf
                j = int(np.argmin(new_costs))
                if new_costs[j] + 1e-6 < obj:
                    sel[pos] = j
                    obj = new_costs[j]
                    improved = True
                    sub = self.D[:, sel]
                    order = np.argsort(sub, axis=1)
                    d1 = sub[np.arange(self.N), order[:, 0]]
                    d2 = sub[np.arange(self.N), order[:, 1]]
                    nearest_pos = order[:, 0]
            if not improved:
                break
        if verbose:
            print(f"   [p-median TB] p={p}  amaç={obj / self.w.sum() / 1000:.4f} km "
                  f"(ağırlıklı ort.)  süre={time.time() - t0:.1f}s")
        return sel, obj

    def solve_exact(self, p, time_limit=MILP_TIME_LIMIT, dist_cutoff_m=20000):
        """Kesin p-median MILP (CBC). Küçültülmüş örneklemde çalıştırılmalı."""
        if not PULP_AVAILABLE:
            return None
        t0 = time.time()
        allowed = self.D <= dist_cutoff_m
        # her talep düğümü için en az bir adaya izin ver (fizibilite garantisi)
        nearest_j = self.D.argmin(axis=1)
        allowed[np.arange(self.N), nearest_j] = True

        prob = pulp.LpProblem('p_median', pulp.LpMinimize)
        y = [pulp.LpVariable(f'y_{j}', cat='Binary') for j in range(self.M)]
        x = {}
        for i in range(self.N):
            for j in np.where(allowed[i])[0]:
                x[(i, j)] = pulp.LpVariable(f'x_{i}_{j}', lowBound=0, upBound=1)

        prob += pulp.lpSum(self.w[i] * self.D[i, j] * x[(i, j)] for (i, j) in x)
        for i in range(self.N):
            prob += pulp.lpSum(x[(i, j)] for j in np.where(allowed[i])[0]) == 1
        for (i, j) in x:
            prob += x[(i, j)] <= y[j]
        prob += pulp.lpSum(y) == p

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        obj = pulp.value(prob.objective)
        sel = [j for j in range(self.M) if y[j].value() and y[j].value() > 0.5]
        print(f"   [p-median MILP] durum={status}  amaç={obj / self.w.sum() / 1000:.4f} km"
              f"  değişken={len(x) + self.M}  süre={time.time() - t0:.1f}s")
        return {'status': status, 'objective': obj, 'selected': sel,
                'time': time.time() - t0}


# ---------------------------------------------------------------
# 4. MCLP ÇÖZÜCÜSÜ
# ---------------------------------------------------------------
class MCLPSolver:
    """max  sum_i w_i z_i ;  z_i <= sum_{j: d_ij<=R} y_j ;  sum y_j = p
    Açgözlü (1-1/e yaklaşım garantili) + kesin MILP (y ikili, z sürekli —
    maksimizasyonda z otomatik bütünler olduğundan formülasyon kesindir)."""

    def __init__(self, demand_xy, weights, cand_xy, radius_m=COVERAGE_RADIUS_M):
        self.w = weights
        self.cand_xy = cand_xy
        self.radius = radius_m
        self.cover = dist_matrix(demand_xy, cand_xy) <= radius_m  # N x M bool
        self.N, self.M = self.cover.shape

    def greedy(self, p):
        sel = []
        covered = np.zeros(self.N, dtype=bool)
        for _ in range(p):
            gain = (self.w[:, None] * (self.cover & ~covered[:, None])).sum(axis=0)
            gain[sel] = -1
            j = int(np.argmax(gain))
            sel.append(j)
            covered |= self.cover[:, j]
        return sel, float(self.w[covered].sum())

    def solve_exact(self, p, time_limit=120):
        if not PULP_AVAILABLE:
            return None
        t0 = time.time()
        prob = pulp.LpProblem('mclp', pulp.LpMaximize)
        y = [pulp.LpVariable(f'y_{j}', cat='Binary') for j in range(self.M)]
        z = [pulp.LpVariable(f'z_{i}', lowBound=0, upBound=1) for i in range(self.N)]
        prob += pulp.lpSum(self.w[i] * z[i] for i in range(self.N))
        for i in range(self.N):
            js = np.where(self.cover[i])[0]
            if len(js) == 0:
                prob += z[i] == 0
            else:
                prob += z[i] <= pulp.lpSum(y[j] for j in js)
        prob += pulp.lpSum(y) == p
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        status = pulp.LpStatus[prob.status]
        sel = [j for j in range(self.M) if y[j].value() and y[j].value() > 0.5]
        obj = pulp.value(prob.objective)
        print(f"   [MCLP MILP] durum={status}  kapsanan ağırlık={obj:.0f} "
              f"({100 * obj / self.w.sum():.2f}%)  süre={time.time() - t0:.1f}s")
        return {'status': status, 'objective': obj, 'selected': sel}


# ---------------------------------------------------------------
# 5. MEVCUT KARAKOLLAR (OpenStreetMap)
# ---------------------------------------------------------------
def fetch_existing_stations(bbox):
    """OSM'den şehir sınır kutusu içindeki amenity=police tesislerini çeker.
    Dönen değer: (lat, lon) dizisi (yaklaşık 200 m içinde tekilleştirilmiş)."""
    if not OSMNX_AVAILABLE:
        print("[MEVCUT] osmnx yok - mevcut karakol kıyası atlanıyor.")
        return None
    try:
        from shapely.geometry import box
        b = bbox
        poly = box(b['lon_min'], b['lat_min'], b['lon_max'], b['lat_max'])
        print("[MEVCUT] OSM'den mevcut polis tesisleri indiriliyor (amenity=police)...")
        gdf = ox.features_from_polygon(poly, tags={'amenity': 'police'})
        pts = gdf.geometry.representative_point()
        lats, lons = pts.y.values, pts.x.values
        # ~200 m ızgarada tekilleştir (aynı tesisin node+way kopyaları)
        x, y = project_to_meters(lons, lats)
        key = set()
        keep = []
        for i in range(len(lats)):
            k = (int(x[i] // 200), int(y[i] // 200))
            if k not in key:
                key.add(k)
                keep.append(i)
        lats, lons = lats[keep], lons[keep]
        print(f"[MEVCUT] {len(gdf)} OSM öğesi -> {len(lats)} tekil tesis konumu")
        return np.column_stack([lats, lons])
    except Exception as e:
        print(f"[MEVCUT] OSM erişimi başarısız ({e}) - kıyas atlanıyor.")
        return None


# ---------------------------------------------------------------
# 6. ZAMANSAL DIŞ-ÖRNEKLEM DOĞRULAMA
# ---------------------------------------------------------------
def robustness_validation(df, cand_xy, p):
    """Dış-örneklem doğrulama. Veri birden çok ay içeriyorsa zamansal bölme
    (ilk aylar -> eğitim, son aylar -> test); tek ay içeriyorsa rastgele
    yarı-bölme ile örneklem sağlamlığı testi. Eğitim yarısında seçilen
    yerleşim, test yarısının talebiyle değerlendirilir ve test yarısının
    kendi optimumuyla kıyaslanır."""
    months = sorted(df['month'].dropna().unique())
    if len(months) >= 2:
        half = len(months) // 2
        df_a = df[df['month'].isin(months[:half])]
        df_b = df[df['month'].isin(months[half:])]
        mode = 'zamansal'
        label_a = f"{months[0]}..{months[half - 1]}"
        label_b = f"{months[half]}..{months[-1]}"
    else:
        rng = np.random.RandomState(RANDOM_SEED)
        mask = rng.rand(len(df)) < 0.5
        df_a, df_b = df[mask], df[~mask]
        mode = 'örneklem (rastgele yarı-bölme)'
        label_a, label_b = 'A yarısı (eğitim)', 'B yarısı (test)'
    print(f"[DOĞRULAMA] Kip: {mode} | eğitim: {label_a} ({len(df_a)} kayıt) | "
          f"test: {label_b} ({len(df_b)} kayıt)")

    g_train = DemandGrid(df_a)
    g_test = DemandGrid(df_b)

    s_train = PMedianSolver(g_train.xy, g_train.weights, cand_xy)
    sel_train, _ = s_train.teitz_bart(p, verbose=False)

    s_test = PMedianSolver(g_test.xy, g_test.weights, cand_xy)
    sel_test, _ = s_test.teitz_bart(p, verbose=False)

    m_train_on_test = evaluate_solution(cand_xy[sel_train], g_test.xy, g_test.weights)
    m_test_opt = evaluate_solution(cand_xy[sel_test], g_test.xy, g_test.weights)
    gap = 100.0 * (m_train_on_test['weighted_mean_km'] / m_test_opt['weighted_mean_km'] - 1.0)
    print(f"[DOĞRULAMA] Test yarısında: eğitim-yerleşimi={m_train_on_test['weighted_mean_km']:.4f} km, "
          f"test-optimumu={m_test_opt['weighted_mean_km']:.4f} km, dış-örneklem açığı={gap:.2f}%")
    return {'mode': mode, 'label_train': label_a, 'label_test': label_b,
            'train_on_test': m_train_on_test, 'test_opt': m_test_opt, 'gap_pct': gap}


# ---------------------------------------------------------------
# 7. DEVRİYE ROTASI (GERÇEK YOL AĞI + CHRISTOFIDES TSP)
# ---------------------------------------------------------------
def patrol_route(station_latlon):
    """Karakollar arası gerçek yol-ağı mesafeleriyle metrik tam çizge kurar;
    MST alt sınırını ve Christofides kapalı devriye turunu hesaplar.
    Dönen değer: tur sırası, uzunluklar ve harita için rota geometrileri."""
    n = len(station_latlon)
    result = {'network': False}
    if OSMNX_AVAILABLE:
        try:
            from shapely.geometry import MultiPoint
            print(f"[DEVRIYE] {n} karakolu kapsayan gerçek yol ağı indiriliyor...")
            hull = MultiPoint([(lo, la) for la, lo in station_latlon]).convex_hull.buffer(0.03)
            G = ox.graph_from_polygon(hull, network_type='drive', simplify=True)
            Gu = ox.convert.to_undirected(G)
            print(f"[DEVRIYE] Yol ağı: {len(Gu.nodes)} düğüm, {len(Gu.edges)} kenar")
            nodes = ox.distance.nearest_nodes(
                Gu, X=[lo for la, lo in station_latlon],
                Y=[la for la, lo in station_latlon])
            # ikili yol-ağı mesafeleri
            Dm = np.full((n, n), np.inf)
            for i, src in enumerate(nodes):
                lengths = nx.single_source_dijkstra_path_length(Gu, src, weight='length')
                for j, dst in enumerate(nodes):
                    if dst in lengths:
                        Dm[i, j] = lengths[dst]
            Dm = np.minimum(Dm, Dm.T)
            if np.isinf(Dm).any():
                raise RuntimeError('yol ağı bazı çiftler için bağlantısız')
            result.update(network=True, G=Gu, nodes=nodes, D=Dm)
        except Exception as e:
            print(f"[DEVRIYE] Yol ağı kullanılamadı ({e}) - kuş uçuşu mesafeye dönülüyor.")
    if not result['network']:
        lats = station_latlon[:, 0]; lons = station_latlon[:, 1]
        x, y = project_to_meters(lons, lats)
        result['D'] = dist_matrix(np.column_stack([x, y]), np.column_stack([x, y]))

    Dm = result['D']
    Gc = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gc.add_edge(i, j, weight=float(Dm[i, j]))

    mst_edges = list(nx.minimum_spanning_edges(Gc, data=True))
    mst_len = sum(d['weight'] for _, _, d in mst_edges)
    tour = nx.approximation.christofides(Gc, weight='weight')  # kapalı tur
    tour_len = sum(Dm[tour[k], tour[k + 1]] for k in range(len(tour) - 1))
    print(f"[DEVRIYE] MST alt sınırı={mst_len / 1000:.2f} km | "
          f"Christofides turu={tour_len / 1000:.2f} km "
          f"(oran={tour_len / mst_len:.3f}, garanti<=1.5{' , GERÇEK YOL' if result['network'] else ' , kuş uçuşu'})")

    # harita geometrileri: ardışık tur çiftleri için yol geometrisi
    geoms = []
    if result['network']:
        Gu, nodes = result['G'], result['nodes']
        for k in range(len(tour) - 1):
            try:
                path = nx.shortest_path(Gu, nodes[tour[k]], nodes[tour[k + 1]], weight='length')
                geoms.append([(Gu.nodes[nd]['y'], Gu.nodes[nd]['x']) for nd in path])
            except Exception:
                geoms.append([tuple(station_latlon[tour[k]]), tuple(station_latlon[tour[k + 1]])])
    else:
        for k in range(len(tour) - 1):
            geoms.append([tuple(station_latlon[tour[k]]), tuple(station_latlon[tour[k + 1]])])

    result.update(tour=tour, tour_len_m=tour_len, mst_len_m=mst_len,
                  geoms=geoms, mst_edges=[(u, v, d['weight']) for u, v, d in mst_edges])
    return result


# ---------------------------------------------------------------
# 8. GÖRSELLEŞTİRME
# ---------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(C_SURFACE)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def plot_sensitivity(sens_rows, existing_metrics, p_exist, out='p_duyarlilik_analizi.png'):
    ps = [r['p'] for r in sens_rows]
    wmd = [r['weighted_mean_km'] for r in sens_rows]
    cov = [r['coverage_3km_pct'] for r in sens_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=C_SURFACE)
    ax = axes[0]
    _style_axes(ax)
    ax.plot(ps, wmd, color=C_BLUE, linewidth=2, marker='o', markersize=5)
    if existing_metrics:
        ax.axhline(existing_metrics['weighted_mean_km'], color=C_GRAY,
                   linestyle='--', linewidth=1.5)
        ax.annotate(f"Mevcut karakollar (p={p_exist}): "
                    f"{existing_metrics['weighted_mean_km']:.2f} km",
                    xy=(ps[-1], existing_metrics['weighted_mean_km']),
                    xytext=(0, 6), textcoords='offset points',
                    ha='right', fontsize=8.5, color=C_INK2)
    ax.set_xlabel('Karakol sayısı (p)', color=C_INK2, fontsize=10)
    ax.set_ylabel('Talep-ağırlıklı ort. mesafe (km)', color=C_INK2, fontsize=10)
    ax.set_title('p-Median amaç değeri', color=C_INK, fontsize=11, loc='left')

    ax = axes[1]
    _style_axes(ax)
    ax.plot(ps, cov, color=C_AQUA, linewidth=2, marker='o', markersize=5)
    if existing_metrics:
        ax.axhline(existing_metrics['coverage_3km_pct'], color=C_GRAY,
                   linestyle='--', linewidth=1.5)
        ax.annotate(f"Mevcut karakollar (p={p_exist}): "
                    f"%{existing_metrics['coverage_3km_pct']:.1f}",
                    xy=(ps[-1], existing_metrics['coverage_3km_pct']),
                    xytext=(0, -12), textcoords='offset points',
                    ha='right', fontsize=8.5, color=C_INK2)
    ax.set_xlabel('Karakol sayısı (p)', color=C_INK2, fontsize=10)
    ax.set_ylabel('3 km kapsama (talep %)', color=C_INK2, fontsize=10)
    ax.set_title('Kapsama oranı', color=C_INK, fontsize=11, loc='left')

    fig.suptitle('Karakol sayısı duyarlılık analizi (Teitz-Bart p-median)',
                 color=C_INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"[GRAFIK] {out}")


def plot_comparison(metrics_by_method, out='kapsama_karsilastirmasi.png'):
    methods = list(metrics_by_method.keys())
    colors = {'Mevcut karakollar': C_GRAY, 'p-Median (önerilen)': C_BLUE,
              'MCLP (önerilen)': C_AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=C_SURFACE)

    ax = axes[0]
    _style_axes(ax)
    vals = [metrics_by_method[m]['weighted_mean_km'] for m in methods]
    bars = ax.bar(methods, vals, width=0.55,
                  color=[colors.get(m, C_YELLOW) for m in methods])
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.2f}', xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9, color=C_INK)
    ax.set_ylabel('Talep-ağırlıklı ort. mesafe (km)', color=C_INK2, fontsize=10)
    ax.set_title('Ortalama erişim mesafesi (düşük = iyi)', color=C_INK,
                 fontsize=11, loc='left')
    ax.tick_params(axis='x', labelrotation=8)

    ax = axes[1]
    _style_axes(ax)
    xpos = np.arange(len(COVERAGE_LEVELS_M))
    width = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = [metrics_by_method[m][f'coverage_{r // 1000}km_pct'] for r in COVERAGE_LEVELS_M]
        ax.bar(xpos + k * width, vals, width=width * 0.92,
               color=colors.get(m, C_YELLOW), label=m)
    ax.set_xticks(xpos + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f'{r // 1000} km' for r in COVERAGE_LEVELS_M])
    ax.set_ylabel('Kapsanan talep (%)', color=C_INK2, fontsize=10)
    ax.set_title('Yarıçapa göre kapsama (yüksek = iyi)', color=C_INK,
                 fontsize=11, loc='left')
    ax.legend(fontsize=8.5, frameon=False, labelcolor=C_INK2)

    fig.suptitle('Mevcut ve önerilen karakol yerleşimlerinin kıyası (eşit p)',
                 color=C_INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"[GRAFIK] {out}")


def plot_robustness(tv, out='dis_orneklem_dogrulama.png'):
    if tv is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=C_SURFACE)
    _style_axes(ax)
    labels = [f"Test-optimum yerleşim\n({tv['label_test']})",
              f"Eğitim yerleşimi\n({tv['label_train']})"]
    vals = [tv['test_opt']['weighted_mean_km'], tv['train_on_test']['weighted_mean_km']]
    bars = ax.bar(labels, vals, width=0.5, color=[C_BLUE, C_YELLOW])
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.3f} km', xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=9.5, color=C_INK)
    ax.set_ylabel('Test yarısı talep-ağırlıklı ort. mesafe (km)',
                  color=C_INK2, fontsize=10)
    ax.set_title(f"Dış-örneklem doğrulama ({tv['mode']}) — açık: %{tv['gap_pct']:.2f}",
                 color=C_INK, fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"[GRAFIK] {out}")


def build_map(df, pmed_latlon, mclp_latlon, existing_latlon, patrol,
              patrol_latlon, out='optimizasyon_haritasi.html', variants=None):
    m = folium.Map(location=[float(df['lat'].mean()), float(df['lon'].mean())],
                   zoom_start=10, tiles='cartodbpositron')

    # önem-ağırlıklı ısı haritası (performans için 30k örneklem)
    hs = df.sample(n=min(30000, len(df)), random_state=RANDOM_SEED)
    heat = [[r.lat, r.lon, r.severity] for r in hs.itertuples()]
    plugins.HeatMap(heat, radius=11, blur=13, min_opacity=0.25,
                    name='Suç yoğunluğu (önem ağırlıklı)').add_to(m)

    if existing_latlon is not None:
        fg = folium.FeatureGroup(name=f'Mevcut karakollar (OSM, n={len(existing_latlon)})')
        for la, lo in existing_latlon:
            folium.CircleMarker([la, lo], radius=4, color='#52514e',
                                fill=True, fill_opacity=0.8, weight=1,
                                tooltip='Mevcut polis tesisi (OSM)').add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name=f'Önerilen: p-Median (n={len(pmed_latlon)})')
    for i, (la, lo) in enumerate(pmed_latlon):
        folium.Marker([la, lo], tooltip=f'p-Median karakol #{i + 1}',
                      icon=folium.Icon(color='red', icon='shield-halved',
                                       prefix='fa')).add_to(fg)
        folium.Circle([la, lo], radius=COVERAGE_RADIUS_M, color='#e34948',
                      weight=1, fill=True, fill_opacity=0.05).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup(name=f'Önerilen: MCLP (n={len(mclp_latlon)})', show=False)
    for i, (la, lo) in enumerate(mclp_latlon):
        folium.Marker([la, lo], tooltip=f'MCLP karakol #{i + 1}',
                      icon=folium.Icon(color='blue', icon='shield-halved',
                                       prefix='fa')).add_to(fg)
    fg.add_to(m)

    if patrol is not None:
        tag = 'gerçek yol ağı' if patrol['network'] else 'kuş uçuşu'
        fg = folium.FeatureGroup(
            name=f"Devriye turu (Christofides, p={len(patrol_latlon)}, {tag}, "
                 f"{patrol['tour_len_m'] / 1000:.1f} km)")
        for k, geom in enumerate(patrol['geoms']):
            leg_km = patrol['D'][patrol['tour'][k], patrol['tour'][k + 1]] / 1000.0
            folium.PolyLine(geom, color='#4a3aa7', weight=3, opacity=0.85,
                            tooltip=f'Devriye bacağı {k + 1} — {leg_km:.1f} km'
                            ).add_to(fg)
        for i, (la, lo) in enumerate(patrol_latlon):
            folium.CircleMarker([la, lo], radius=5, color='#4a3aa7', fill=True,
                                fill_color='#ffffff', fill_opacity=1,
                                tooltip=f'Devriye istasyonu #{i + 1}').add_to(fg)
        fg.add_to(m)

    # Ağırlık duyarlılığı varyantları (saglamlik_analizi.py üretir):
    # alternatif ağırlıklamayla seçilen konumlar, kapalı gelen katmanlar
    if variants:
        vcolors = ['#1baf7a', '#eda100', '#c33d69']
        for k, (name, latlon) in enumerate(variants.items()):
            fg = folium.FeatureGroup(
                name=f'Ağırlık varyantı: {name} (n={len(latlon)})', show=False)
            for la, lo in latlon:
                folium.CircleMarker([la, lo], radius=5,
                                    color=vcolors[k % len(vcolors)],
                                    fill=True, fill_opacity=0.9, weight=1,
                                    tooltip=f'p-Median karakol — {name} ağırlıklaması'
                                    ).add_to(fg)
            fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out)
    print(f"[HARITA] {out}")


# ---------------------------------------------------------------
# 9. ANA AKIŞ
# ---------------------------------------------------------------
def main(city='london', map_only=False):
    """map_only=True: yalnızca haritayı yeniden üretir — MILP doğrulaması,
    p duyarlılığı, dış-örneklem doğrulaması ve grafik/CSV çıktıları atlanır."""
    cfg = CITY_CONFIGS[city]
    set_projection(cfg.get('epsg', 27700))
    outdir = os.path.join('sonuclar', city)
    os.makedirs(outdir, exist_ok=True)

    t_start = time.time()
    np.random.seed(RANDOM_SEED)
    print('=' * 70)
    print(f"POLİS KARAKOLU YERLEŞİM VE DEVRİYE OPTİMİZASYONU — {cfg['name'].upper()}")
    print('p-Median + MCLP + gerçek yol ağı devriye turu')
    print(f"Çıktı dizini: {outdir}")
    print('=' * 70)

    # 1) Veri
    df = CrimeDataLoader(cfg).load()

    # 2) Talep modeli
    # Talep: 500 m ızgara. Adaylar: tüm şehre yayılı 1 km ızgara hücreleri —
    # adayları yalnızca yoğun bölgelere kısıtlamak büyük p'de dış bölgeleri
    # hizmetsiz bırakır ve mevcut duruma karşı haksız bir kıyas üretir.
    grid = DemandGrid(df, GRID_SIZE_M)
    cand_grid = DemandGrid(df, CAND_GRID_M)
    cand_xy = cand_grid.xy
    print(f"[TALEP] {grid.n} talep hücresi ({GRID_SIZE_M} m ızgara), "
          f"{len(cand_xy)} aday tesis konumu ({CAND_GRID_M} m ızgara)")

    # 3) Mevcut karakollar (OSM)
    existing = fetch_existing_stations(cfg['bbox'])
    if existing is not None:
        ex_x, ex_y = project_to_meters(existing[:, 1], existing[:, 0])
        existing_xy = np.column_stack([ex_x, ex_y])
        existing_metrics = evaluate_solution(existing_xy, grid.xy, grid.weights)
        p_star = len(existing)
    else:
        existing_xy, existing_metrics = None, None
        p_star = 25
    print(f"[KURULUM] Ana senaryo: p* = {p_star} karakol (mevcut tesis sayısıyla eşit)")

    # 4) p-Median (ana senaryo, sezgisel)
    print('\n--- p-MEDIAN (Teitz-Bart) ---')
    pm = PMedianSolver(grid.xy, grid.weights, cand_xy)
    sel_pm, obj_pm = pm.teitz_bart(p_star)
    pmed_xy = cand_xy[sel_pm]
    pmed_metrics = evaluate_solution(pmed_xy, grid.xy, grid.weights)

    # 5) Kesin MILP doğrulaması (kaba ızgara örneklemi)
    print('\n--- p-MEDIAN KESİN MILP DOĞRULAMASI (kaba ızgara) ---')
    exact_report = None
    if map_only:
        print('   (sadece-harita kipi — atlandı)')
    elif PULP_AVAILABLE:
        coarse = DemandGrid(df, COARSE_GRID_M)
        cand_exact = cand_grid.top_candidates(N_CANDIDATES_EXACT)
        pm_c = PMedianSolver(coarse.xy, coarse.weights, cand_exact)
        sel_h, obj_h = pm_c.teitz_bart(p_star, verbose=False)
        res = pm_c.solve_exact(p_star)
        if res and res['objective']:
            gap = 100.0 * (obj_h / res['objective'] - 1.0)
            print(f"   [DOĞRULAMA] Sezgisel amaç={obj_h / coarse.weights.sum() / 1000:.4f} km, "
                  f"MILP amaç={res['objective'] / coarse.weights.sum() / 1000:.4f} km, "
                  f"optimalite açığı={gap:.3f}%")
            exact_report = {'heuristic_obj': obj_h, 'milp_obj': res['objective'],
                            'gap_pct': gap, 'status': res['status'],
                            'n_demand': coarse.n, 'n_cand': len(cand_exact)}
    else:
        print('   PuLP kurulu değil - atlandı.')

    # 6) MCLP
    print('\n--- MCLP (3 km kapsama maksimizasyonu) ---')
    mclp = MCLPSolver(grid.xy, grid.weights, cand_xy)
    sel_g, cov_g = mclp.greedy(p_star)
    print(f"   [MCLP açgözlü] kapsanan ağırlık %{100 * cov_g / grid.weights.sum():.2f}")
    res_mclp = mclp.solve_exact(p_star)
    if (res_mclp and res_mclp['selected'] and res_mclp['objective']
            and res_mclp['objective'] >= cov_g):
        sel_mclp = res_mclp['selected']
    else:
        sel_mclp = sel_g
    mclp_xy = cand_xy[sel_mclp]
    mclp_metrics = evaluate_solution(mclp_xy, grid.xy, grid.weights)

    # 7) p duyarlılık analizi
    sens_rows, tv = [], None
    if not map_only:
        print('\n--- p DUYARLILIK ANALİZİ ---')
        p_list = sorted(set(P_SENSITIVITY + [p_star]))
        for p in p_list:
            sel, _ = pm.teitz_bart(p, verbose=False)
            met = evaluate_solution(cand_xy[sel], grid.xy, grid.weights)
            met['p'] = p
            sens_rows.append(met)
            print(f"   p={p:3d}  ort={met['weighted_mean_km']:.3f} km  "
                  f"3km kapsama=%{met['coverage_3km_pct']:.1f}")

        # 8) Dış-örneklem doğrulama (zamansal veya örneklem sağlamlığı)
        print('\n--- DIŞ-ÖRNEKLEM DOĞRULAMA ---')
        tv = robustness_validation(df, cand_xy, p_star)

    # 9) Devriye rotası (operasyonel senaryo p=PATROL_P)
    print(f"\n--- DEVRİYE ROTASI (operasyonel senaryo p={PATROL_P}) ---")
    sel_patrol, _ = pm.teitz_bart(PATROL_P, verbose=False)
    patrol_xy = cand_xy[sel_patrol]
    plon, plat = unproject_to_wgs(patrol_xy[:, 0], patrol_xy[:, 1])
    patrol_latlon = np.column_stack([plat, plon])
    patrol = patrol_route(patrol_latlon)

    # 10) Çıktılar
    print('\n--- ÇIKTILAR ---')
    lon_pm, lat_pm = unproject_to_wgs(pmed_xy[:, 0], pmed_xy[:, 1])
    lon_mc, lat_mc = unproject_to_wgs(mclp_xy[:, 0], mclp_xy[:, 1])
    pmed_latlon = np.column_stack([lat_pm, lon_pm])
    mclp_latlon = np.column_stack([lat_mc, lon_mc])

    metrics_by_method = {}
    if existing_metrics:
        metrics_by_method['Mevcut karakollar'] = existing_metrics
    metrics_by_method['p-Median (önerilen)'] = pmed_metrics
    metrics_by_method['MCLP (önerilen)'] = mclp_metrics

    if not map_only:
        plot_sensitivity(sens_rows, existing_metrics, p_star,
                         out=os.path.join(outdir, 'p_duyarlilik_analizi.png'))
        plot_comparison(metrics_by_method,
                        out=os.path.join(outdir, 'kapsama_karsilastirmasi.png'))
        plot_robustness(tv, out=os.path.join(outdir, 'dis_orneklem_dogrulama.png'))

    # Ağırlık duyarlılığı varyantları (varsa) harita katmanı olarak eklenir
    variants = None
    vfile = os.path.join(outdir, 'saglamlik', 'agirlik_varyant_konumlari.csv')
    if os.path.exists(vfile):
        vdf = pd.read_csv(vfile)
        variants = {s: g[['lat', 'lon']].values
                    for s, g in vdf.groupby('scheme', sort=False)}
        print(f"[HARITA] {len(variants)} ağırlık varyantı katmanı ekleniyor: "
              f"{', '.join(variants)}")

    build_map(df, pmed_latlon, mclp_latlon, existing, patrol, patrol_latlon,
              out=os.path.join(outdir, 'optimizasyon_haritasi.html'),
              variants=variants)

    # CSV'ler
    if map_only:
        print(f"\n[TAMAM] Sadece-harita kipi: {outdir}/optimizasyon_haritasi.html "
              f"güncellendi ({time.time() - t_start:.0f} s)")
        return
    rows = []
    for i, (la, lo) in enumerate(pmed_latlon):
        rows.append({'method': 'p-median', 'id': i + 1, 'lat': la, 'lon': lo})
    for i, (la, lo) in enumerate(mclp_latlon):
        rows.append({'method': 'mclp', 'id': i + 1, 'lat': la, 'lon': lo})
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, 'optimum_karakol_konumlari.csv'), index=False)

    pd.DataFrame(sens_rows).to_csv(
        os.path.join(outdir, 'p_duyarlilik_sonuclari.csv'), index=False)

    summary = []
    for name, met in metrics_by_method.items():
        r = {'method': name}
        r.update(met)
        summary.append(r)
    pd.DataFrame(summary).to_csv(os.path.join(outdir, 'metrik_ozet.csv'), index=False)

    tour_rows = []
    cum = 0.0
    for k in range(len(patrol['tour']) - 1):
        i = patrol['tour'][k]
        seg = patrol['D'][patrol['tour'][k], patrol['tour'][k + 1]]
        tour_rows.append({'order': k + 1, 'station': int(i) + 1,
                          'lat': patrol_latlon[i, 0], 'lon': patrol_latlon[i, 1],
                          'leg_km': seg / 1000.0, 'cumulative_km': cum / 1000.0})
        cum += seg
    pd.DataFrame(tour_rows).to_csv(os.path.join(outdir, 'devriye_turu.csv'), index=False)
    print(f"[CSV] {outdir}/: optimum_karakol_konumlari.csv, metrik_ozet.csv, "
          'p_duyarlilik_sonuclari.csv, devriye_turu.csv')

    # ÖZET RAPOR
    print('\n' + '=' * 70)
    print('SONUÇ ÖZETİ')
    print('=' * 70)
    if existing_metrics:
        imp_d = 100 * (1 - pmed_metrics['weighted_mean_km'] / existing_metrics['weighted_mean_km'])
        imp_c = pmed_metrics['coverage_3km_pct'] - existing_metrics['coverage_3km_pct']
        imp_c_mclp = mclp_metrics['coverage_3km_pct'] - existing_metrics['coverage_3km_pct']
        print(f"Mevcut {p_star} tesise kıyasla önerilen yerleşimler (eşit p):")
        print(f"  * p-Median, talep-ağırlıklı ort. mesafe: "
              f"{existing_metrics['weighted_mean_km']:.3f} -> "
              f"{pmed_metrics['weighted_mean_km']:.3f} km  "
              f"({imp_d:+.1f}% ; pozitif = iyileşme)")
        print(f"  * p-Median, 3 km kapsama: %{existing_metrics['coverage_3km_pct']:.1f} -> "
              f"%{pmed_metrics['coverage_3km_pct']:.1f}  ({imp_c:+.1f} puan)")
        print(f"  * MCLP, 3 km kapsama: %{existing_metrics['coverage_3km_pct']:.1f} -> "
              f"%{mclp_metrics['coverage_3km_pct']:.1f}  ({imp_c_mclp:+.1f} puan)")
        # Eşdeğer hizmet analizi: mevcut hizmet seviyesine ulaşan en küçük p
        p_eq_d = next((r['p'] for r in sens_rows
                       if r['weighted_mean_km'] <= existing_metrics['weighted_mean_km']), None)
        p_eq_c = next((r['p'] for r in sens_rows
                       if r['coverage_3km_pct'] >= existing_metrics['coverage_3km_pct']), None)
        if p_eq_d:
            print(f"  * Mevcut ort. mesafe seviyesi ({existing_metrics['weighted_mean_km']:.3f} km), "
                  f"optimize yerleşimle p={p_eq_d} karakolda yakalanıyor "
                  f"(%{100 * (1 - p_eq_d / p_star):.0f} daha az tesis)")
        if p_eq_c:
            print(f"  * Mevcut 3 km kapsama seviyesi (%{existing_metrics['coverage_3km_pct']:.1f}), "
                  f"optimize yerleşimle p={p_eq_c} karakolda yakalanıyor "
                  f"(%{100 * (1 - p_eq_c / p_star):.0f} daha az tesis)")
    if exact_report:
        print(f"Sezgiselin MILP'e göre optimalite açığı (kaba örneklem): "
              f"%{exact_report['gap_pct']:.3f} ({exact_report['status']})")
    if tv:
        print(f"Zamansal dış-örneklem açığı: %{tv['gap_pct']:.2f} "
              f"(yerleşim zaman içinde kararlı)")
    print(f"Devriye turu (p={PATROL_P}): {patrol['tour_len_m'] / 1000:.1f} km "
          f"(MST alt sınırı {patrol['mst_len_m'] / 1000:.1f} km)")
    print(f"\nToplam süre: {time.time() - t_start:.0f} s")


if __name__ == '__main__':
    # Kullanım: python polis_optimizasyon.py [şehir ...] [--sadece-harita]
    #   şehir: 'london' (varsayılan), 'west-midlands', 'chicago' veya 'all'
    #   --sadece-harita: yalnızca haritayı yeniden üretir (hızlı)
    args = sys.argv[1:]
    map_only = '--sadece-harita' in args
    args = [a for a in args if a != '--sadece-harita'] or ['london']
    cities = list(CITY_CONFIGS) if args == ['all'] else args
    for c in cities:
        if c not in CITY_CONFIGS:
            print(f"Bilinmeyen şehir: {c} (geçerli: {', '.join(CITY_CONFIGS)})")
            continue
        main(c, map_only=map_only)
