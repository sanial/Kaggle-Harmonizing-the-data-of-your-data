"""
prefetch_caches.py
==================
Run this LOCALLY (internet required) before uploading to Kaggle.

Produces two files in ./outputs/kaggle_dataset/:
  pride_cache.json   — raw PRIDE API responses for all 15 test PXDs
  ols_cache.json     — OLS4 canonical strings for every term we'll encounter

Upload both files as a Kaggle dataset, then reference them in the notebook.

Usage:
  cd your-project-root
  python notebooks/prefetch_caches.py
"""

import re, json, time, requests
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / 'outputs' / 'kaggle_dataset'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRIDE_CACHE_PATH = OUT_DIR / 'pride_cache.json'
OLS_CACHE_PATH   = OUT_DIR / 'ols_cache.json'

# ── Test PXDs ─────────────────────────────────────────────────────────────────
TEST_PXDS = [
    'PXD004010','PXD016436','PXD019519','PXD025663','PXD040582',
    'PXD050621','PXD061009','PXD061090','PXD061136','PXD061195',
    'PXD061285','PXD062014','PXD062469','PXD062877','PXD064564',
]

# ── HTTP sessions ─────────────────────────────────────────────────────────────
pride_session = requests.Session()
pride_session.headers.update({'User-Agent': 'SDRF-Prefetch/1.0'})

ols_session = requests.Session()
ols_session.headers.update({'Accept': 'application/json', 'User-Agent': 'SDRF-OLS/1.0'})


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: PRIDE API cache
# Captures the full raw API response so the notebook can re-process it
# with any normalization logic without needing internet.
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pride_raw(pxd):
    """Fetch full raw PRIDE API response for a PXD."""
    try:
        r = pride_session.get(
            f'https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}',
            timeout=20
        )
        if r.status_code != 200:
            print(f'  PRIDE {pxd}: HTTP {r.status_code}')
            return None
        return r.json()
    except Exception as e:
        print(f'  PRIDE {pxd}: {e}')
        return None


def fetch_px_xml_raw(pxd):
    """Fetch ProteomeXchange XML as a string backup."""
    try:
        r = pride_session.get(
            f'https://proteomecentral.proteomexchange.org/cgi/GetDataset'
            f'?ID={pxd}&outputMode=XML&test=no',
            timeout=20
        )
        if r.status_code != 200:
            return None
        return r.text
    except Exception as e:
        print(f'  PX XML {pxd}: {e}')
        return None


def build_pride_cache():
    print('\n' + '='*60)
    print('PART 1: Fetching PRIDE API for all test PXDs')
    print('='*60)

    cache = {}
    for pxd in TEST_PXDS:
        print(f'\n  {pxd}...')
        raw = fetch_pride_raw(pxd)
        px_xml = fetch_px_xml_raw(pxd)

        if raw:
            # Extract the fields we care about for display
            orgs  = [o.get('name','') for o in raw.get('organisms', [])]
            parts = [p.get('name','') for p in (raw.get('organisms_part') or raw.get('tissues') or [])]
            insts = [i.get('name','') for i in raw.get('instruments', [])]
            dis   = [d.get('name','') for d in raw.get('diseases', [])]
            qms   = [q.get('name','') for q in raw.get('quantification_methods', [])]
            kws   = raw.get('keywords', [])
            title = raw.get('title', '')

            print(f'    organisms   : {orgs}')
            print(f'    tissues     : {parts}')
            print(f'    instruments : {insts}')
            print(f'    diseases    : {dis}')
            print(f'    labels      : {qms}')
            print(f'    title       : {title[:80]}')

        cache[pxd] = {
            'pride_api': raw,
            'px_xml': px_xml,
        }
        time.sleep(0.5)  # be polite

    with open(PRIDE_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    filled = sum(1 for v in cache.values() if v['pride_api'])
    print(f'\n  Saved pride_cache.json — {filled}/{len(TEST_PXDS)} PXDs fetched')
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: OLS cache
# For every unique raw string that needs ontology normalization,
# query OLS4 and store the canonical NT=;AC= result.
# ══════════════════════════════════════════════════════════════════════════════

def ols_query(term, ontology):
    """Query OLS4 API. Returns canonical string or None."""
    if not term or str(term).strip().lower() in ('not applicable','na','n/a',''):
        return None
    try:
        r = ols_session.get(
            'https://www.ebi.ac.uk/ols4/api/search',
            params={
                'q': str(term).strip(),
                'ontology': ontology,
                'rows': 1,
                'exact': 'false',
                'fieldList': 'label,obo_id,short_form',
            },
            timeout=8
        )
        if r.status_code != 200:
            return None
        docs = r.json().get('response', {}).get('docs', [])
        if not docs:
            return None
        doc = docs[0]
        label  = doc.get('label', '')
        obo_id = doc.get('obo_id', '') or doc.get('short_form', '')
        if label and obo_id:
            if ontology == 'ms' and obo_id.startswith('MS:'):
                return f'AC={obo_id};NT={label}'
            return f'NT={label};AC={obo_id}'
        return None
    except Exception as e:
        print(f'    OLS error [{ontology}] "{term}": {e}')
        return None


def collect_terms_to_resolve(pride_cache):
    """
    Walk the PRIDE cache and collect every unique raw string
    that needs OLS resolution, organized by ontology.
    """
    terms = {
        'uberon': set(),   # tissues / organism parts
        'ms': set(),       # instruments, fragmentation methods
        'efo': set(),      # diseases (EFO ontology)
        'ncbitaxon': set(),# organisms
    }

    for pxd, data in pride_cache.items():
        raw = data.get('pride_api')
        if not raw:
            continue

        # Organisms
        for o in raw.get('organisms', []):
            name = o.get('name', '').strip()
            if name and name.lower() not in ('not available', 'n/a', ''):
                terms['ncbitaxon'].add(name)

        # Tissues
        for op in (raw.get('organisms_part') or raw.get('tissues') or []):
            name = op.get('name', '').strip()
            if name and name.lower() not in ('not available', 'n/a', ''):
                terms['uberon'].add(name)

        # Instruments
        for inst in raw.get('instruments', []):
            name = inst.get('name', '').strip()
            if name:
                terms['ms'].add(name)

        # Diseases
        for dis in raw.get('diseases', []):
            name = dis.get('name', '').strip()
            if name and name.lower() not in ('not available', 'n/a', 'none', 'normal'):
                terms['efo'].add(name)

    # Also add all the tissue terms from our regex patterns
    # (these are the terms regex would find in paper text)
    TISSUE_TEXT_TERMS = [
        'blood plasma', 'blood serum', 'whole blood', 'peripheral blood',
        'cerebrospinal fluid', 'bronchoalveolar lavage', 'synovial fluid',
        'prefrontal cortex', 'frontal cortex', 'temporal cortex', 'cerebral cortex',
        'hippocampus', 'cerebellum', 'striatum', 'substantia nigra',
        'bone marrow', 'adipose tissue', 'skeletal muscle', 'lymph node',
        'prostate gland', 'extracellular vesicle',
        'plasma', 'serum', 'blood', 'urine', 'saliva', 'csf',
        'brain', 'cortex', 'liver', 'lung', 'heart', 'kidney', 'pancreas',
        'colon', 'prostate', 'breast', 'ovary', 'spleen', 'thymus',
        'adipose', 'muscle', 'skin', 'testis', 'retina', 'stomach',
        'pbmc', 'platelet', 'exosome',
    ]
    for t in TISSUE_TEXT_TERMS:
        terms['uberon'].add(t)

    # Instrument name variants from paper text
    INSTRUMENT_TEXT_TERMS = [
        'Q Exactive HF-X', 'Q Exactive HF', 'Q Exactive Plus', 'Q Exactive',
        'Orbitrap Astral', 'Orbitrap Fusion Lumos', 'Orbitrap Fusion',
        'Orbitrap Eclipse', 'Orbitrap Exploris 480', 'Exploris 480',
        'LTQ Orbitrap Velos', 'LTQ Orbitrap Elite', 'LTQ Orbitrap XL', 'LTQ Orbitrap',
        'timsTOF Pro 2', 'timsTOF Pro', 'timsTOF',
        'TripleTOF 6600', 'TripleTOF 5600',
        'Synapt G2-Si', 'impact II',
        'Orbitrap', 'ion trap', 'TOF',
    ]
    for inst in INSTRUMENT_TEXT_TERMS:
        terms['ms'].add(inst)

    return terms


def build_ols_cache(pride_cache):
    print('\n' + '='*60)
    print('PART 2: Building OLS cache')
    print('='*60)

    terms = collect_terms_to_resolve(pride_cache)

    print('\n  Terms to resolve:')
    for onto, tset in terms.items():
        print(f'    {onto}: {len(tset)} terms')

    ols_cache = {}
    total = sum(len(v) for v in terms.values())
    done = 0

    for ontology, term_set in terms.items():
        print(f'\n  Querying OLS [{ontology}] — {len(term_set)} terms...')
        for term in sorted(term_set):
            result = ols_query(term, ontology)
            key = f'{ontology}::{term.lower().strip()}'
            ols_cache[key] = result
            done += 1

            status = result if result else 'None'
            print(f'    [{done}/{total}] "{term}" → {status}')
            time.sleep(0.15)  # rate limiting

    with open(OLS_CACHE_PATH, 'w') as f:
        json.dump(ols_cache, f, indent=2)

    hits = sum(1 for v in ols_cache.values() if v)
    print(f'\n  Saved ols_cache.json — {hits}/{len(ols_cache)} terms resolved')
    return ols_cache


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Validation — print a summary so you can spot errors before uploading
# ══════════════════════════════════════════════════════════════════════════════

def validate_caches(pride_cache, ols_cache):
    print('\n' + '='*60)
    print('PART 3: Validation summary')
    print('='*60)

    print('\n  PRIDE cache — normalized values per PXD:')
    for pxd, data in pride_cache.items():
        raw = data.get('pride_api')
        if not raw:
            print(f'    {pxd}: MISSING')
            continue

        orgs  = [o.get('name','') for o in raw.get('organisms', [])]
        parts = [p.get('name','') for p in (raw.get('organisms_part') or raw.get('tissues') or [])]
        insts = [i.get('name','') for i in raw.get('instruments', [])]
        dis   = [d.get('name','') for d in raw.get('diseases', [])]

        print(f'\n    {pxd}:')
        print(f'      org    : {orgs}')
        print(f'      tissue : {parts}')
        print(f'      inst   : {insts}')
        print(f'      disease: {dis}')

    print('\n\n  OLS cache — key tissue terms:')
    key_tissues = ['blood serum', 'blood plasma', 'brain', 'bone marrow',
                   'kidney', 'liver', 'lung', 'adipose tissue']
    for tissue in key_tissues:
        key = f'uberon::{tissue}'
        val = ols_cache.get(key, 'NOT IN CACHE')
        print(f'    {tissue:30s} → {val}')

    print('\n  OLS cache — key instruments:')
    key_insts = ['Orbitrap Exploris 480', 'Q Exactive HF', 'timsTOF Pro', 'Orbitrap Astral']
    for inst in key_insts:
        key = f'ms::{inst.lower()}'
        val = ols_cache.get(key, 'NOT IN CACHE')
        print(f'    {inst:30s} → {val}')

    print(f'\n  Files saved to: {OUT_DIR}')
    print(f'    pride_cache.json : {PRIDE_CACHE_PATH.stat().st_size / 1024:.1f} KB')
    print(f'    ols_cache.json   : {OLS_CACHE_PATH.stat().st_size / 1024:.1f} KB')
    print('\n  Upload both files as a Kaggle dataset.')
    print('  Suggested dataset name: sdrf-api-caches')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('SDRF Cache Pre-fetcher')
    print('Requires internet access — run locally, not on Kaggle.')
    print(f'Output directory: {OUT_DIR}')

    # Check if caches already exist (allow partial re-runs)
    if PRIDE_CACHE_PATH.exists():
        print(f'\n  pride_cache.json already exists — loading...')
        with open(PRIDE_CACHE_PATH) as f:
            pride_cache = json.load(f)
        print(f'  Loaded {len(pride_cache)} PXDs from cache.')
    else:
        pride_cache = build_pride_cache()

    if OLS_CACHE_PATH.exists():
        print(f'\n  ols_cache.json already exists — loading...')
        with open(OLS_CACHE_PATH) as f:
            ols_cache = json.load(f)
        print(f'  Loaded {len(ols_cache)} OLS terms from cache.')
    else:
        ols_cache = build_ols_cache(pride_cache)

    validate_caches(pride_cache, ols_cache)
