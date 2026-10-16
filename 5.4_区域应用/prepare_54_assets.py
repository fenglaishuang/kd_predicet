from pathlib import Path
import pandas as pd
p = Path(__file__).resolve().parent / 'results' / 'satellite_vertical_kd_202507.csv'
d = pd.read_csv(p)
d.loc[d.component == 'slow', 'kd_reconstructed'] = d.loc[d.component == 'slow', 'kd_reconstructed'].clip(lower=0.001)
d.to_csv(p, index=False)
