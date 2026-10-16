from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import re, math, json
import numpy as np
from scipy.stats import linregress
ROOT = Path('C:\\Users\\77940\\Documents\\Codex\\2026-09-10\\5-8-5-1-8-x20-2')
SRC = ROOT / 'ch5_source' / '给福财' / '给福财' / '附件2：叶绿素和瞬时kd'
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def read_xlsx(path):
    with ZipFile(path) as z:
        ss = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            ss = [''.join((t.text or '' for t in si.iter('{%s}t' % NS['m']))) for si in root.findall('m:si', NS)]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in root.findall('.//m:sheetData/m:row', NS):
            vals = {}
            for c in row.findall('m:c', NS):
                v = c.find('m:v', NS)
                val = v.text if v is not None else ''
                if c.get('t') == 's' and val:
                    val = ss[int(val)]
                vals[c.get('r')] = val
            rows.append(vals)
    header = rows[0]
    cols = []
    for cell, name in header.items():
        m = re.match('([A-Z]+)', cell)
        cols.append((m.group(1), name))
    out = []
    for row in rows[1:]:
        d = {}
        for col, name in cols:
            v = row.get(next((c for c in row if c.startswith(col)), ''), '')
            try:
                d[name] = float(v)
            except:
                d[name] = v
        out.append(d)
    return out
files = sorted(SRC.glob('*.xlsx'))
rows = []
for f in files:
    rows.extend(read_xlsx(f))
wl = np.array([int(k[2:]) for k in rows[0] if str(k).startswith('Kd') and str(k)[2:].isdigit()])
wl.sort()
kdcols = [f'Kd{x}' for x in wl]
records = []
for r in rows:
    try:
        kd = np.array([float(r.get(c, np.nan)) for c in kdcols])
        k490 = float(r.get('Kd490', np.nan))
        chl = float(r.get('chl', np.nan))
        if np.isfinite(k490) and k490 > 0 and np.isfinite(chl) and (chl > 0) and np.all(np.isfinite(kd)) and np.all(kd > 0):
            slope = -linregress(np.log(wl / 490), np.log(kd / k490)).slope
            records.append((k490, chl, slope, *kd))
    except Exception:
        pass
A = np.array(records)
slopes = A[:, 2]
chl = A[:, 1]
X = np.column_stack([np.ones(len(chl)), np.log(chl)])
beta = np.linalg.lstsq(X, slopes, rcond=None)[0]
predS = X @ beta
pred = np.array([r[0] * (wl / 490.0) ** (-predS[i]) for i, r in enumerate(records)])
obs = A[:, 3:]
rel = (pred - obs) / obs
summary = []
for j, w in enumerate(wl):
    e = rel[:, j]
    summary.append({'wavelength_nm': int(w), 'n': int(len(e)), 'median_abs_pct': float(np.nanmedian(np.abs(e)) * 100), 'rmse_m-1': float(np.sqrt(np.nanmean((pred[:, j] - obs[:, j]) ** 2))), 'bias_m-1': float(np.nanmean(pred[:, j] - obs[:, j]))})
result = {'n_rows': int(len(records)), 'n_files': len(files), 'wavelengths_nm': wl.tolist(), 'slope_model': 'S=a+b*log(chl)', 'beta': beta.tolist(), 'slope_rmse': float(np.sqrt(np.mean((predS - slopes) ** 2))), 'summary': summary}
(OUT / 'spectral_model_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
import csv
with (OUT / 'spectral_model_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
    wr = csv.writer(f)
    wr.writerow(['Kd490', 'chl', 'S_observed', 'S_predicted'] + [f'obs_Kd{w}' for w in wl] + [f'pred_Kd{w}' for w in wl])
    for i, r in enumerate(records):
        wr.writerow([r[0], r[1], r[2], predS[i], *r[3:], *pred[i]])
