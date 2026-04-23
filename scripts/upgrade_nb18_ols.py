#!/usr/bin/env python3
"""
Upgrade notebook 18 with OLS entity-linking infrastructure from notebook 16.

Changes:
  Cell 2  — add requests/lru_cache/difflib imports + OLS_TIMEOUT / PX_TIMEOUT constants
  Cell 4  — update _find_col with _strip_wrapper; append OLS functions
             (ols_lookup, ols_organism, ols_tissue, ols_instrument)
  Cell 6  — add col_vocab tracking to training SDRF loader
  Cell 20 — replace simple dict-based fetch_pride with OLS version; add fetch_px_xml
  Cell 26 — add fuzzy_snap + Layer 2 (PX XML) before KNN layer
"""
import json, sys, textwrap
from pathlib import Path

NB = Path('notebooks/18_semantic_similarity_pipeline.ipynb')
nb = json.load(open(NB, 'r', encoding='utf-8'))

def src(idx): return ''.join(nb['cells'][idx]['source'])

def set_src(idx, text):
    """Store text back into a cell as a list of line-strings (jpynb convention)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        pass  # last line has no trailing newline — that's fine
    nb['cells'][idx]['source'] = lines

# ────────────────────────────────────────────────────────────────────────────────
# Cell 2 — imports + OLS/PX timeouts
# ────────────────────────────────────────────────────────────────────────────────
c2 = src(2)
assert 'from tqdm import tqdm' in c2, "anchor not found in cell 2"
c2 = c2.replace(
    'from tqdm import tqdm',
    'from tqdm import tqdm\nimport difflib\nimport requests\nfrom functools import lru_cache'
)
assert 'PRIDE_TIMEOUT   = 15' in c2, "PRIDE_TIMEOUT anchor not found in cell 2"
c2 = c2.replace(
    'PRIDE_TIMEOUT   = 15',
    'PRIDE_TIMEOUT   = 15\nOLS_TIMEOUT     = 6\nPX_TIMEOUT      = 12'
)
set_src(2, c2)
print('✓ Cell 2: added difflib/requests/lru_cache imports + OLS_TIMEOUT/PX_TIMEOUT')

# ────────────────────────────────────────────────────────────────────────────────
# Cell 4 — update _find_col with _strip_wrapper; append OLS functions
# ────────────────────────────────────────────────────────────────────────────────
OLD_FIND_COL = """\
def _find_col(col: str, df_cols: set) -> str:
    \"\"\"Locate a possibly-renamed column in a training SDRF (handles case / spacing variation).\"\"\"
    if col in df_cols:
        return col
    col_l = col.lower().replace(' ', '')
    for c in df_cols:
        if c.lower().replace(' ', '') == col_l:
            return c
    return ''


print('Text helpers & ontologies loaded.')"""

NEW_FIND_COL_AND_OLS = """\
def _strip_wrapper(col):
    m = re.match(r'(?:characteristics|comment|factor\\s*value)\\[(.+?)\\]', col, re.I)
    return m.group(1) if m else col


def _find_col(col: str, df_cols: set) -> str:
    \"\"\"Locate a possibly-renamed column in a training SDRF.\"\"\"
    if col in df_cols: return col
    base = re.sub(r'\\.\\d+$', '', col)
    if base in df_cols: return base
    stripped = _strip_wrapper(base)
    if stripped in df_cols: return stripped
    return ''


# ── OLS4 entity linking ───────────────────────────────────────────────────────
_ols_session = requests.Session()
_ols_session.headers.update({'Accept': 'application/json', 'User-Agent': 'SDRF-OLS/1.0'})


@lru_cache(maxsize=2000)
def ols_lookup(term, ontology):
    \"\"\"Query OLS4 for a term in a specific ontology. Returns canonical NT=;AC= string or None.
    ontology: 'uberon', 'ms', 'unimod', 'pride'
    \"\"\"
    if not term or str(term).strip().lower() in ('not applicable', 'na', 'n/a', ''):
        return None
    try:
        r = _ols_session.get(
            'https://www.ebi.ac.uk/ols4/api/search',
            params={'q': str(term).strip(), 'ontology': ontology,
                    'rows': 1, 'exact': 'false', 'fieldList': 'label,obo_id,short_form'},
            timeout=OLS_TIMEOUT
        )
        if r.status_code != 200: return None
        docs = r.json().get('response', {}).get('docs', [])
        if not docs: return None
        doc = docs[0]
        label  = doc.get('label', '')
        obo_id = doc.get('obo_id', '') or doc.get('short_form', '')
        if label and obo_id:
            if ontology == 'ms' and obo_id.startswith('MS:'):
                return f'AC={obo_id};NT={label}'
            return f'NT={label};AC={obo_id}'
        return None
    except Exception:
        return None


def ols_organism(name):
    \"\"\"Resolve organism name to NCBI taxon ID format: '9606 (Homo sapiens)'.\"\"\"
    LOCAL = {
        'homo sapiens': '9606 (Homo sapiens)', 'human': '9606 (Homo sapiens)',
        'humans': '9606 (Homo sapiens)',
        'mus musculus': '10090 (Mus musculus)', 'mouse': '10090 (Mus musculus)',
        'mice': '10090 (Mus musculus)', 'murine': '10090 (Mus musculus)',
        'rattus norvegicus': '10116 (Rattus norvegicus)', 'rat': '10116 (Rattus norvegicus)',
        'saccharomyces cerevisiae': '4932 (Saccharomyces cerevisiae)',
        'yeast': '4932 (Saccharomyces cerevisiae)',
        'escherichia coli': '562 (Escherichia coli)', 'e. coli': '562 (Escherichia coli)',
        'e.coli': '562 (Escherichia coli)',
        'drosophila melanogaster': '7227 (Drosophila melanogaster)',
        'danio rerio': '7955 (Danio rerio)', 'zebrafish': '7955 (Danio rerio)',
        'arabidopsis thaliana': '3702 (Arabidopsis thaliana)',
        'sus scrofa': '9823 (Sus scrofa)', 'pig': '9823 (Sus scrofa)', 'porcine': '9823 (Sus scrofa)',
        'bos taurus': '9913 (Bos taurus)', 'bovine': '9913 (Bos taurus)',
        'gallus gallus': '9031 (Gallus gallus)', 'chicken': '9031 (Gallus gallus)',
        'caenorhabditis elegans': '6239 (Caenorhabditis elegans)',
        'c. elegans': '6239 (Caenorhabditis elegans)',
        'xenopus laevis': '8355 (Xenopus laevis)',
        'macaca mulatta': '9544 (Macaca mulatta)',
        'rabbit': '9986 (Oryctolagus cuniculus)',
        'oryctolagus cuniculus': '9986 (Oryctolagus cuniculus)',
        'dog': '9615 (Canis lupus familiaris)',
    }
    n = str(name).lower().strip()
    for key in sorted(LOCAL, key=len, reverse=True):
        if key in n: return LOCAL[key]
    return None


def ols_tissue(name):
    \"\"\"Resolve tissue/organ to UBERON canonical string; falls back to OLS4 API.\"\"\"
    TISSUE_FAST = {
        'blood plasma': 'NT=blood plasma;AC=UBERON:0001969',
        'plasma': 'NT=blood plasma;AC=UBERON:0001969',
        'blood serum': 'NT=blood serum;AC=UBERON:0001977',
        'serum': 'NT=blood serum;AC=UBERON:0001977',
        'whole blood': 'NT=blood;AC=UBERON:0000178',
        'blood': 'NT=blood;AC=UBERON:0000178',
        'peripheral blood': 'NT=blood;AC=UBERON:0000178',
        'urine': 'NT=urine;AC=UBERON:0001088',
        'cerebrospinal fluid': 'NT=cerebrospinal fluid;AC=UBERON:0001359',
        'csf': 'NT=cerebrospinal fluid;AC=UBERON:0001359',
        'saliva': 'NT=saliva;AC=UBERON:0001836',
        'brain': 'NT=brain;AC=UBERON:0000955',
        'prefrontal cortex': 'NT=prefrontal cortex;AC=UBERON:0000451',
        'frontal cortex': 'NT=frontal cortex;AC=UBERON:0001870',
        'cerebral cortex': 'NT=cerebral cortex;AC=UBERON:0000956',
        'hippocampus': 'NT=hippocampal formation;AC=UBERON:0002421',
        'cerebellum': 'NT=cerebellum;AC=UBERON:0002037',
        'liver': 'NT=liver;AC=UBERON:0002107',
        'lung': 'NT=lung;AC=UBERON:0002048',
        'heart': 'NT=heart;AC=UBERON:0000948',
        'kidney': 'NT=kidney;AC=UBERON:0002113',
        'pancreas': 'NT=pancreas;AC=UBERON:0001264',
        'colon': 'NT=colon;AC=UBERON:0001155',
        'prostate': 'NT=prostate gland;AC=UBERON:0002367',
        'prostate gland': 'NT=prostate gland;AC=UBERON:0002367',
        'breast': 'NT=breast;AC=UBERON:0000310',
        'ovary': 'NT=ovary;AC=UBERON:0000992',
        'spleen': 'NT=spleen;AC=UBERON:0002106',
        'bone marrow': 'NT=bone marrow;AC=UBERON:0002371',
        'adipose tissue': 'NT=adipose tissue;AC=UBERON:0001013',
        'adipose': 'NT=adipose tissue;AC=UBERON:0001013',
        'skeletal muscle': 'NT=skeletal muscle;AC=UBERON:0001134',
        'muscle': 'NT=skeletal muscle;AC=UBERON:0001134',
        'skin': 'NT=skin of body;AC=UBERON:0002097',
        'thymus': 'NT=thymus;AC=UBERON:0002370',
        'lymph node': 'NT=lymph node;AC=UBERON:0000029',
        'testis': 'NT=testis;AC=UBERON:0000473',
        'retina': 'NT=retina;AC=UBERON:0000966',
        'pbmc': 'NT=peripheral blood mononuclear cell;AC=CL:0000057',
        'peripheral blood mononuclear': 'NT=peripheral blood mononuclear cell;AC=CL:0000057',
        'platelet': 'NT=platelet;AC=CL:0000233',
        'extracellular vesicle': 'NT=extracellular vesicle;AC=GO:0061695',
        'exosome': 'NT=extracellular vesicle;AC=GO:0061695',
    }
    n = str(name).lower().strip()
    for key in sorted(TISSUE_FAST, key=len, reverse=True):
        if key in n: return TISSUE_FAST[key]
    return ols_lookup(name, 'uberon')


def ols_instrument(name):
    \"\"\"Resolve instrument to PSI-MS canonical AC=MS:XXXXXX;NT=Name; falls back to OLS4 API.\"\"\"
    INSTRUMENT_FAST = {
        'q exactive hf-x': 'AC=MS:1003027;NT=Q Exactive HF-X',
        'q exactive hf': 'AC=MS:1002523;NT=Q Exactive HF',
        'q exactive plus': 'AC=MS:1002634;NT=Q Exactive Plus',
        'q-exactive plus': 'AC=MS:1002634;NT=Q Exactive Plus',
        'q exactive': 'AC=MS:1001911;NT=Q Exactive',
        'qexactive': 'AC=MS:1001911;NT=Q Exactive',
        'orbitrap astral': 'AC=MS:1003378;NT=Orbitrap Astral',
        'orbitrap fusion lumos': 'AC=MS:1002732;NT=Orbitrap Fusion Lumos',
        'fusion lumos': 'AC=MS:1002732;NT=Orbitrap Fusion Lumos',
        'orbitrap fusion': 'AC=MS:1002416;NT=Orbitrap Fusion',
        'orbitrap eclipse': 'AC=MS:1003029;NT=Orbitrap Eclipse',
        'orbitrap exploris 480': 'AC=MS:1003094;NT=Orbitrap Exploris 480',
        'exploris 480': 'AC=MS:1003094;NT=Orbitrap Exploris 480',
        'ltq orbitrap velos': 'AC=MS:1001742;NT=LTQ Orbitrap Velos',
        'ltq orbitrap elite': 'AC=MS:1001910;NT=LTQ Orbitrap Elite',
        'ltq orbitrap xl': 'AC=MS:1000556;NT=LTQ Orbitrap XL',
        'ltq orbitrap': 'AC=MS:1000449;NT=LTQ Orbitrap',
        'timstof pro 2': 'AC=MS:1003474;NT=timsTOF Pro 2',
        'timstof pro': 'AC=MS:1003231;NT=timsTOF Pro',
        'timstof': 'AC=MS:1002817;NT=timsTOF',
        'triple tof 6600': 'AC=MS:1000931;NT=TripleTOF 6600',
        'triple tof 5600': 'AC=MS:1000931;NT=TripleTOF 5600',
        'triple tof': 'AC=MS:1000931;NT=TripleTOF 6600',
        'impact ii': 'AC=MS:1002817;NT=impact II',
        'synapt g2': 'AC=MS:1002726;NT=Synapt G2-Si',
        'velos pro': 'AC=MS:1001909;NT=LTQ Velos Pro',
    }
    n = str(name).lower().strip()
    ac = re.search(r'AC=(MS:\\d+)', name)
    nt_m = re.search(r'NT=([^;]+)', name)
    if ac and nt_m:
        return f'AC={ac.group(1).strip()};NT={nt_m.group(1).strip()}'
    for key in sorted(INSTRUMENT_FAST, key=len, reverse=True):
        if key in n: return INSTRUMENT_FAST[key]
    return ols_lookup(name, 'ms')


print('Text helpers, OLS functions & ontologies loaded.')"""

c4 = src(4)
assert OLD_FIND_COL in c4, f"OLD_FIND_COL anchor not found in cell 4. First 200 chars:\n{c4[:200]}"
c4 = c4.replace(OLD_FIND_COL, NEW_FIND_COL_AND_OLS)
set_src(4, c4)
print('✓ Cell 4: updated _find_col with _strip_wrapper; added OLS functions')

# ────────────────────────────────────────────────────────────────────────────────
# Cell 6 — add col_vocab to SDRF loader
# ────────────────────────────────────────────────────────────────────────────────
c6 = src(6)

# Add col_vocab init after col_counters init
OLD_COUNTERS = 'col_counters = {col: Counter() for col in target_cols}\ntrain_pxd_sdrf: dict[str, dict] = {}'
NEW_COUNTERS  = 'col_counters = {col: Counter() for col in target_cols}\ncol_vocab    = defaultdict(set)   # per-column training vocabulary for fuzzy_snap\ntrain_pxd_sdrf: dict[str, dict] = {}'
assert OLD_COUNTERS in c6, "col_counters anchor not found in cell 6"
c6 = c6.replace(OLD_COUNTERS, NEW_COUNTERS)

# Populate col_vocab inside the per-col loop (after col_counters.update)
OLD_UPDATE = '            col_counters[col].update(vals.tolist())\n            uniq = list(vals.unique())'
NEW_UPDATE  = '            col_counters[col].update(vals.tolist())\n            col_vocab[re.sub(r\'\\.\\d+$\', \'\', col)].update(vals.tolist())\n            uniq = list(vals.unique())'
assert OLD_UPDATE in c6, f"col_counters.update anchor not found in cell 6.\nCell:\n{c6}"
c6 = c6.replace(OLD_UPDATE, NEW_UPDATE)

set_src(6, c6)
print('✓ Cell 6: added col_vocab tracking to SDRF loader')

# ────────────────────────────────────────────────────────────────────────────────
# Cell 20 — replace simple dict-based fetch_pride with OLS version; add fetch_px_xml
# ────────────────────────────────────────────────────────────────────────────────
NEW_CELL20 = """\
http_session = requests.Session()
http_session.headers.update({'User-Agent': 'SDRF-OLS/1.0'})


def fetch_pride(pxd):
    \"\"\"Query EBI PRIDE API; normalise with OLS functions from cell 4.\"\"\"
    try:
        r = http_session.get(
            f'https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}',
            timeout=PRIDE_TIMEOUT
        )
        if r.status_code != 200: return {}
        d = r.json()
        out = defaultdict(list)

        for o in d.get('organisms', []):
            name = o.get('name', '')
            if name:
                norm = ols_organism(name)
                if norm: out['Characteristics[Organism]'].append(norm)

        for op in (d.get('organisms_part') or d.get('tissues') or []):
            name = op.get('name', ''); acc = op.get('accession', '')
            if name and name.lower() not in ('not available', 'n/a', ''):
                norm = ols_tissue(name)
                if norm:
                    out['Characteristics[OrganismPart]'].append(norm)
                elif acc:
                    out['Characteristics[OrganismPart]'].append(f'NT={name};AC={acc}')

        for dis in d.get('diseases', []):
            name = dis.get('name', '')
            if name and name.lower() not in ('not available', 'n/a', 'none', 'normal', ''):
                DISEASE_NORM = {
                    'lung cancer': 'lung carcinoma',
                    'breast cancer': 'breast carcinoma',
                    'prostate cancer': 'prostate carcinoma',
                    'prostate adenocarcinoma': 'prostate carcinoma',
                    'colorectal cancer': 'colorectal carcinoma',
                    'colon cancer': 'colorectal carcinoma',
                    'ovarian cancer': 'ovarian carcinoma',
                    'brain glioblastoma multiforme': 'glioblastoma',
                    "alzheimer's disease": 'Alzheimer disease',
                    "parkinson's disease": 'Parkinson disease',
                    'healthy': 'normal', 'healthy control': 'normal',
                }
                norm = DISEASE_NORM.get(name.lower().strip(), name)
                out['Characteristics[Disease]'].append(norm)

        for inst in d.get('instruments', []):
            name = inst.get('name', ''); acc = inst.get('accession', '')
            if name:
                norm = ols_instrument(name)
                if norm:
                    out['Comment[Instrument]'].append(norm)
                elif acc:
                    out['Comment[Instrument]'].append(f'AC={acc};NT={name}')

        for qm in d.get('quantification_methods', []):
            name = qm.get('name', '')
            if name:
                norm = fmt_label(name)
                out['Characteristics[Label]'].append(norm)

        return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}
    except Exception as e:
        print(f'  PRIDE error {pxd}: {e}')
        return {}


def fetch_px_xml(pxd):
    \"\"\"ProteomeXchange XML backup — instruments (MS cvParams) + organisms (NEWT cvParams).\"\"\"
    out = defaultdict(list)
    try:
        r = http_session.get(
            f'https://proteomecentral.proteomexchange.org/cgi/GetDataset'
            f'?ID={pxd}&outputMode=XML&test=no',
            timeout=PX_TIMEOUT
        )
        if r.status_code != 200: return {}
        xml = r.text
        for m in re.finditer(r'<cvParam[^>]+accession="(MS:\\d+)"[^>]+name="([^"]+)"', xml):
            if 'instrument' in m.group(2).lower():
                norm = ols_instrument(m.group(2))
                if norm: out['Comment[Instrument]'].append(norm)
        for m in re.finditer(r'<cvParam[^>]+accession="(NEWT:\\d+)"[^>]+name="([^"]+)"', xml):
            name = m.group(2)
            norm = ols_organism(name)
            if norm: out['Characteristics[Organism]'].append(norm)
    except Exception:
        pass
    return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}


print('PRIDE API fetchers (OLS-normalised) ready.')"""

set_src(20, NEW_CELL20)
print('✓ Cell 20: replaced fetch_pride (OLS) + added fetch_px_xml')

# ────────────────────────────────────────────────────────────────────────────────
# Cell 26 — add fuzzy_snap + Layer 2 (PX XML)
# ────────────────────────────────────────────────────────────────────────────────
c26 = src(26)

# 1. Add fuzzy_snap definition before the main loop
OLD_LOOP_START = 'for pxd, pxd_df in tqdm(final_sub.groupby(\'PXD\'), desc=\'PXDs\'):'
FUZZY_SNAP = """\
def fuzzy_snap(value, base_col, cutoff=0.82):
    \"\"\"Snap a predicted value to the nearest known training vocabulary label.\"\"\"
    if not value or base_col not in col_vocab: return value
    matches = difflib.get_close_matches(value, list(col_vocab[base_col]), n=1, cutoff=cutoff)
    return matches[0] if matches else value


"""
assert OLD_LOOP_START in c26, f"main loop anchor not found in cell 26.\nFirst 300:\n{c26[:300]}"
c26 = c26.replace(OLD_LOOP_START, FUZZY_SNAP + OLD_LOOP_START)

# 2. Add Layer 2 (PX XML) after Layer 1 (PRIDE), before Layer 3 (KNN)
DASH52 = '\u2500' * 52
DASH37 = '\u2500' * 37
DASH47 = '\u2500' * 47
DASH31 = '\u2500' * 31

OLD_PRIDE_BLOCK = (
    f"# \u2500\u2500 Layer 1: PRIDE API {DASH52}\n"
    "    pride_data = fetch_pride(pxd)\n"
    "    time.sleep(0.3)\n"
    "    for col, vals in pride_data.items():\n"
    "        for v in vals:\n"
    "            pxd_add(col, v)\n"
    "\n"
    f"    # \u2500\u2500 Layer 2: KNN semantic similarity {DASH37}"
)
# Build the PX XML line with the same total line length (75 chars) as original KNN header
DASH_PX = '\u2500' * 19  # to make "# ── Layer 2: PX XML backup (instruments + organisms) " + 19 = 76 chars
NEW_PRIDE_BLOCK = (
    f"# \u2500\u2500 Layer 1: PRIDE API {DASH52}\n"
    "    pride_data = fetch_pride(pxd)\n"
    "    time.sleep(0.3)\n"
    "    for col, vals in pride_data.items():\n"
    "        for v in vals:\n"
    "            pxd_add(col, v)\n"
    "\n"
    f"    # \u2500\u2500 Layer 2: PX XML backup (instruments + organisms) {DASH_PX}\n"
    "    for col, vals in fetch_px_xml(pxd).items():\n"
    "        for v in (vals or []): pxd_add(col, v)\n"
    "\n"
    f"    # \u2500\u2500 Layer 3: KNN semantic similarity {DASH37}"
)
assert OLD_PRIDE_BLOCK in c26, f"PRIDE block anchor not found in cell 26.\nContent:\n{c26[c26.find('Layer 1'):c26.find('Layer 1')+500]}"
c26 = c26.replace(OLD_PRIDE_BLOCK, NEW_PRIDE_BLOCK)

# 3. Renumber later layer comments (exact lines)
c26 = c26.replace(
    f"\u2500\u2500 Layer 3: Protocol regex {DASH47}",
    f"\u2500\u2500 Layer 4: Protocol regex {DASH47}"
)
c26 = c26.replace(
    f"\u2500\u2500 Layer 4: Conservative majority fallback {DASH31}",
    f"\u2500\u2500 Layer 5: Conservative majority fallback {DASH31}"
)

set_src(26, c26)
print('✓ Cell 26: added fuzzy_snap + PX XML Layer 2; renumbered later layers')

# ────────────────────────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────────────────────────
json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'\n✓ Saved {NB}')
print('All 5 upgrades complete.')
