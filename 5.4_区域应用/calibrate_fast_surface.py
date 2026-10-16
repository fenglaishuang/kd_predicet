from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import re, numpy as np, pandas as pd, json
from sklearn.linear_model import LinearRegression
ROOT = Path('C:\\Users\\77940\\Documents\\Codex\\2026-09-10\\5-8-5-1-8-x20-2')
P = Path('C:\\Users\\77940\\OneDrive\\Desktop\\博士毕业\\外审\\外审意见1\\第五章加内容')
O = Path(__file__).resolve().parent / 'results'
SRC = ROOT / 'ch5_source' / '给福财' / '给福财' / '附件2：叶绿素和瞬时kd'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def xlsx(path):
    with ZipFile(path) as z:
        ss = []
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        ss = [''.join((t.text or '' for t in si.iter('{%s}t' % NS['m']))) for si in root.findall('m:si', NS)]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in root.findall('.//m:sheetData/m:row', NS):
            d = {}
            for c in row.findall('m:c', NS):
                v = c.find('m:v', NS)
                val = v.text if v is not None else ''
                if c.get('t') == 's' and val:
                    val = ss[int(val)]
                d[c.get('r')] = val
            rows.append(d)
    h = rows[0]
    cols = [(re.match('([A-Z]+)', k).group(1), v) for k, v in h.items()]
    out = []
    for row in rows[1:]:
        d = {}
        for col, n in cols:
            cell = next((k for k in row if k.startswith(col)), None)
            v = row.get(cell, '')
            try:
                d[n] = float(v)
            except:
                d[n] = v
        if str(d.get('depth', '')).strip():
            out.append(d)
    return out
files = sorted(SRC.glob('*.xlsx'))
prof = pd.read_csv(P / 'reconstruct_fast_3layers_transition' / 'all_profiles_reconstructed_fast.csv')
tops = prof.groupby('profile_id').first().reset_index()
rec = []
for i, f in enumerate(files):
    rows = xlsx(f)
    r = min(rows, key=lambda x: float(x['depth']))
    rec.append({'profile_id': i + 1, 'kd490_obs': float(r['Kd490']), 'chl_obs': float(r['chl'])})
cal = pd.DataFrame(rec).merge(tops[['profile_id', 'top_kd_fast']], on='profile_id')
cal = cal.replace([np.inf, -np.inf], np.nan).dropna()
cal = cal[(cal.kd490_obs > 0) & (cal.chl_obs > 0) & (cal.top_kd_fast > 0)].copy()
X = np.c_[np.log(cal.kd490_obs), np.log(cal.chl_obs)]
y = np.log(cal.top_kd_fast)
pr = np.zeros(len(cal))
for i in range(len(cal)):
    tr = np.arange(len(cal)) != i
    m = LinearRegression().fit(X[tr], y[tr])
    pr[i] = m.predict(X[i:i + 1])[0]
cal['pred_lopo'] = np.exp(pr)
rmse = float(np.sqrt(np.mean((cal.pred_lopo - cal.top_kd_fast) ** 2)))
corr = float(np.corrcoef(cal.pred_lopo, cal.top_kd_fast)[0, 1])
model = LinearRegression().fit(X, y)
sat = pd.read_csv(O / 'satellite_surface_inputs_202507.csv')
sat['kd_fast_calibrated'] = np.exp(model.predict(np.c_[np.log(sat.kd490), np.log(sat.chl)]))
lo, hi = (cal.top_kd_fast.quantile(0.02), cal.top_kd_fast.quantile(0.98))
sat['kd_fast_calibrated'] = sat.kd_fast_calibrated.clip(lo, hi)
sat.to_csv(O / 'satellite_surface_inputs_202507_calibrated.csv', index=False)
cal.to_csv(O / 'fast_surface_calibration_profiles.csv', index=False)
meta = {'formula': 'log(Fast_surface_Kd)=a+b log(Kd490)+c log(chl)', 'coef': model.coef_.tolist(), 'intercept': float(model.intercept_), 'LOPO_RMSE': rmse, 'LOPO_corr': corr, 'clip_range': [float(lo), float(hi)]}
(O / 'fast_surface_calibration_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
