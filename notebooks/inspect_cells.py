import json
from pathlib import Path

nb_path = Path('/home/sbialek/ONC/selfsupervision_anomalies_onc/notebooks/interactive_hydrophone_anomaly_workshop_local.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

cells = nb['cells']

print(f"Total cells: {len(cells)}")
for i, cell in enumerate(cells):
    source = ''.join(cell['source'])[:100].replace('\n', ' ')
    print(f"Cell {i}: Type={cell['cell_type']}, Source={source}...")
