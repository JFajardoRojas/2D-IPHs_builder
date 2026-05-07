# Run Standalone 2D Histogram

From project root:

```bash
cd standalone_2d_histogram
```

## 1) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 2) Run 2D-histogram generation on the 3 test MOFs

```bash
python3 batch_cif_2d.py --cif-dir test_mofs --no-plot --copy-to-npys npys
```

## 3) Build feature table from generated `.npy` files

```bash
python3 mof_2d_features.py --npy-folder npys --output MOF_2D_features_test3.csv --manifest MOF_2D_features_test3.manifest.json
```

## 4) Quick verification

```bash
python3 -c "import pandas as pd; df=pd.read_csv('MOF_2D_features_test3.csv'); print(df.shape); print(df['MOF_name'].tolist())"
```

Expected:
- 3 successful CIF runs from `batch_cif_2d.py`
- `npys/` contains 3 `.npy` files
- CSV shape is `(3, 4441)` (1 MOF id column + 4440 features)
