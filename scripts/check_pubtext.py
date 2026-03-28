import pandas as pd, re
test = pd.read_csv('outputs/test_preprocessed.csv', dtype=str)

for pxd in ['PXD061090', 'PXD061195', 'PXD062469', 'PXD016436', 'PXD050621']:
    rows = test[test['PXD'] == pxd]
    text = rows['pub_text'].iloc[0]
    sents = re.split(r'(?<=[.!?])\s+', text)
    hits = [s for s in sents if re.search(r'fraction|replicate|triplicate|duplicate', s, re.I)]
    print(f'=== {pxd} ({len(rows)} rows) ===')
    for h in hits[:5]:
        print(f'  {h[:220]}')
    print()
