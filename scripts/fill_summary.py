import pandas as pd

sub = pd.read_csv('outputs/submission_with_fallback.csv', dtype=str)

label_cols = [c for c in sub.columns if c not in ('ID', 'PXD', 'Raw Data File')]

rows = []
for col in label_cols:
    non_na = (sub[col] != 'Not Applicable').sum()
    rows.append((col, non_na, round(non_na / len(sub) * 100, 1)))

rows.sort(key=lambda x: -x[1])
print(f"{'Column':<50} {'Filled':>7} {'%':>6}")
print('-' * 66)
for col, n, pct in rows:
    marker = '  <- zero signal' if n == 0 else ''
    print(f"{col:<50} {n:>7} {pct:>5}%{marker}")

print()
print(f'Total rows: {len(sub)}')
print(f'Columns with any fill: {sum(1 for _, n, _ in rows if n > 0)} / {len(rows)}')
