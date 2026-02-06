# BioKB Module Documentation

## 🆕 NEW: Enhanced Field Detection & Batch-Aware Sampling

**Version 2.0 adds:**
- ✨ **Fuzzy Field Matching**: Automatically detects SMILES, dose, plate_id, etc. with 100+ naming variations
- 🎯 **Adaptive Sampling**: Intelligently scales from 50-300 SMILES based on dataset size
- 🧪 **Batch-Aware Sampling**: Prioritizes cross-batch SMILES for robust biological insights
- 📊 **Rich Metadata**: Tracks field matching, batch distribution, and sampling decisions

See [Enhanced Features](#enhanced-features-v20) for details.

---

## Overview

The BioKB (Biological Knowledge Base) module provides SMILES-to-biological-process mapping capabilities for the CellCausal project. It implements a complete semantic enrichment pipeline that:

1. Extracts SMILES strings from configuration or H5 files
2. Queries ChEMBL API for drug targets
3. Queries Reactome API for biological pathways
4. Maps pathways to biological processes using keyword heuristics
5. Generates structured semantic tables in JSON format
6. Converts semantic tables to evidence items with unique Evidence IDs (B1, B2, B3...)

## Architecture

The module follows a modular design inspired by [mosabutey/Biomni](https://github.com/mosabutey/Biomni) with these key patterns:

### Directory Structure

```
cellscientist/core/bio_kb/
├── __init__.py                  # Public API exports
├── config.py                    # BioKBConfig dataclass
├── utils.py                     # Timeout & fallback decorators
├── data_lake.py                 # Static pathway→process mappings
├── tool_schemas.py              # Tool metadata definitions
├── registry.py                  # KnowledgeSource & BioKBRegistry
├── smiles_resolver.py           # SMILES extraction & canonicalization
├── chembl_client.py             # ChEMBL API client with caching
├── reactome_client.py           # Reactome API client with caching
├── process_mapper.py            # Pathway→Process keyword matching
├── evidence_builder.py          # Semantic table & EvidenceItem generation
└── cache/                       # Local cache directory
    └── .gitkeep
```

### Design Patterns

#### 1. Modular Tool System
Each submodule has a single responsibility:
- **smiles_resolver**: SMILES parsing and canonicalization
- **chembl_client**: ChEMBL API interactions
- **reactome_client**: Reactome API interactions
- **process_mapper**: Pathway-to-process mapping logic
- **evidence_builder**: Orchestration and output generation

#### 2. Registry Pattern
The `BioKBRegistry` manages knowledge sources with metadata:
```python
@dataclass
class KnowledgeSource:
    name: str
    description: str
    category: str  # "target" | "pathway" | "process"
    query_fn: Callable
    required_params: List[str]
    optional_params: List[str]
    cache_enabled: bool
    timeout_seconds: int
```

#### 3. Data Lake Concept
Pre-loaded static mappings in `data_lake.py`:
```python
PATHWAY_TO_PROCESS_KEYWORDS = {
    "proliferation": ["cell cycle", "mitosis", "cyclin", ...],
    "apoptosis": ["apoptosis", "caspase", "BCL2", ...],
    "emt": ["epithelial-mesenchymal", "E-cadherin", ...],
    ...
}
```

#### 4. Declarative Tool Schemas
Tool capabilities defined in `tool_schemas.py`:
```python
CHEMBL_TARGET_SCHEMA = ToolSchema(
    name="chembl_target_query",
    required_params=["smiles"],
    optional_params=["inchikey", "max_targets"],
    timeout_seconds=30,
    cacheable=True
)
```

#### 5. Timeout Protection
Decorators for resilience:
```python
@with_timeout(30, fallback=[])
def query_api(url):
    return requests.get(url).json()

@graceful_fallback([])
def query_database():
    return db.query()
```

#### 6. Dataclass Configuration
```python
@dataclass
class BioKBConfig:
    enabled: bool = False
    smiles_list: List[str] = field(default_factory=list)
    chembl_timeout: int = 30
    cache_enabled: bool = True
    
    @classmethod
    def from_cfg(cls, cfg: Dict) -> "BioKBConfig":
        return cls.from_dict(cfg["literature"]["bio_kb"])
```

## Public API

### Core Functions

#### generate_biokb_semantic_table
```python
def generate_biokb_semantic_table(
    cfg: Dict[str, Any],
    stage: str,
    log: Callable[[str], None]
) -> Dict[str, Any]:
    """Generate semantic table with molecule→target→pathway→process mappings.
    
    Args:
        cfg: Configuration dictionary (expects cfg["literature"]["bio_kb"])
        stage: Pipeline stage ("design" or "review")
        log: Logging function
        
    Returns:
        Semantic table dictionary
    """
```

**Output Format:**
```json
{
  "stage": "design",
  "generated_at": "2026-02-06T14:30:21Z",
  "smiles_source": "config",
  "molecules": [
    {
      "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
      "targets": [
        {
          "target_name": "Prostaglandin G/H synthase 1",
          "target_id": "CHEMBL221",
          "gene_symbol": "PTGS1",
          "mechanism": "Cyclooxygenase inhibitor"
        }
      ],
      "pathways": [
        {
          "pathway_id": "R-HSA-2162123",
          "pathway_name": "Synthesis of Prostaglandins",
          "source": "reactome"
        }
      ],
      "inferred_processes": [
        {
          "process": "apoptosis",
          "confidence": 0.7,
          "matched_keywords": ["prostaglandin", "COX"]
        }
      ]
    }
  ],
  "summary": {
    "total_molecules": 1,
    "total_targets": 1,
    "total_pathways": 1,
    "process_distribution": {"apoptosis": 1}
  }
}
```

#### persist_biokb_semantic_table
```python
def persist_biokb_semantic_table(
    semantic_table: Dict[str, Any],
    workspace_dir: str,
    log: Callable[[str], None]
) -> None:
    """Save semantic table to workspace_dir/external_knowledge/biokb_semantic_table_{stage}.json"""
```

#### biokb_table_to_evidence_items
```python
def biokb_table_to_evidence_items(
    semantic_table: Dict[str, Any]
) -> List[Dict]:
    """Convert semantic table to list of evidence item dicts with 'eid' field.
    
    Returns:
        List of evidence items compatible with EvidenceItem dataclass
    """
```

**Evidence Item Format:**
```python
{
    "eid": "B1",
    "title": "BioKB: CC(=O)OC1=CC=CC=C1C(=O)O → PTGS1",
    "url": "",
    "snippet": "BioKB semantic mapping for molecular perturbation...",
    "source": "biokb",
    "published": "2026-02-06T14:30:21Z",
    "scraped_excerpt": "**SMILES**: ...\n**Targets**: ..."
}
```

## Enhanced Features (v2.0)

### Fuzzy Field Matching

The new `FieldMatcher` class automatically detects field names with intelligent matching:

```python
from cellscientist.core.bio_kb.field_matcher import FieldMatcher

matcher = FieldMatcher()

# Automatically handles these variations:
# SMILES: "smiles", "SMILES", "compound_smiles", "smiles_string", etc.
# Dose: "dose", "concentration", "conc", "treatment_dose", etc.
# Plate: "plate_id", "batch", "plate_number", "assay_plate", etc.

result = matcher.find_field(
    available_fields=["SMILES", "Dose", "PlateID"],
    target="smiles"
)
# Returns: ("SMILES", 1.0, "exact_case_insensitive")
```

**Match types:**
- `exact`: Perfect match
- `exact_case_insensitive`: Case-insensitive match
- `synonym`: Matched via synonym list (95% confidence)
- `fuzzy`: Fuzzy string matching (≥60% confidence)

### Adaptive Sampling

Intelligently determines how many SMILES to sample based on dataset size:

```python
from cellscientist.core.bio_kb import BioKBConfig

config = BioKBConfig(
    sampling_strategy="adaptive",
    adaptive_min=50,      # Minimum SMILES to sample
    adaptive_max=300,     # Maximum SMILES to sample
    adaptive_ratio=0.2    # Sample 20% of unique SMILES
)

# Behavior:
# - ≤50 unique: Use all
# - 50-1500 unique: Use 20% (min 50, max 300)
# - >1500 unique: Cap at 300
```

### Batch-Aware Sampling

Prioritizes SMILES that appear in multiple experimental batches:

```python
config = BioKBConfig(
    sampling_method="cross_batch_first",  # Default
    max_smiles_per_batch=20
)

# Methods:
# - "cross_batch_first": Prioritize cross-batch SMILES (robust)
# - "diverse": Sample evenly across all batches
# - "frequent": Select most common SMILES
```

### Robust H5 Extraction

New `extract_smiles_from_h5_robust()` function returns detailed metadata:

```python
from cellscientist.core.bio_kb import extract_smiles_from_h5_robust

smiles_list, metadata = extract_smiles_from_h5_robust(
    h5_path="./data/experiment.h5",
    config=config,
    log=print
)

# metadata includes:
# - source: "h5_pandas" | "h5_h5py" | "none"
# - total_samples: Total rows in dataset
# - unique_smiles: Number of unique SMILES
# - matched_fields: Field matching results with confidence
# - batch_info: Cross-batch vs single-batch distribution
# - sampling_info: Exact sampling statistics
```

**Example output:**
```
[BIOKB] 📂 Loaded H5 group: combined (10000 samples)
[BIOKB] 🔍 Available fields: ['smiles', 'dose', 'plate_id', 'split_id']
[BIOKB] ✅ Matched 'smiles' → 'smiles' (conf=1.00, type=exact)
[BIOKB] ✅ Matched 'plate_id' → 'plate_id' (conf=1.00, type=exact)
[BIOKB] 🧪 Detected 20 batches/plates
[BIOKB] 📊 Cross-batch SMILES: 180 (robust)
[BIOKB] 📊 Single-batch SMILES: 320
[BIOKB] 🎯 Adaptive limit: 125/500 SMILES (25% of unique)
[BIOKB] ✅ Sampled 125 SMILES:
[BIOKB]    - Cross-batch: 110
[BIOKB]    - Single-batch: 15
```

## Configuration

### Configuration Schema

BioKB config is nested under `cfg["literature"]["bio_kb"]`:

```python
{
  "literature": {
    "bio_kb": {
      "enabled": true,
      
      # SMILES sources (priority: smiles_list > h5_path)
      "smiles_list": ["CCO", "CC(C)O"],
      "h5_path": "/path/to/stage1.h5",
      
      # NEW: Sampling strategy
      "sampling_strategy": "adaptive",  # "adaptive" | "fixed"
      "sampling_method": "cross_batch_first",  # "cross_batch_first" | "diverse" | "frequent"
      "max_smiles_per_batch": 15,
      "max_total_smiles": null,  # null = use adaptive logic
      
      # NEW: Adaptive sampling parameters
      "adaptive_min": 50,      # Minimum SMILES to sample
      "adaptive_max": 300,     # Maximum SMILES to sample
      "adaptive_ratio": 0.2,   # Sample 20% of unique SMILES
      
      # API configuration
      "chembl_enabled": true,
      "chembl_base_url": "https://www.ebi.ac.uk/chembl/api/data",
      "chembl_timeout": 30,
      
      "reactome_enabled": true,
      "reactome_base_url": "https://reactome.org/ContentService",
      "reactome_timeout": 30,
      
      # Caching
      "cache_enabled": true,
      "cache_dir": "",  # Auto-set if empty
      "cache_ttl_days": 30,
      
      # Performance
      "parallel_queries": 3,  # NEW
      
      # Output limits
      "max_smiles": 10,  # Backward compatibility (use adaptive_max instead)
      "max_targets_per_smiles": 10,
      "max_pathways_per_target": 20,
      "inject_max_items": 5,
      "include_batch_info": true,  # NEW
      
      # Logging
      "log_to_console": true
    }
  }
}
```

### Environment Variables

- `BIOKB_ENABLED`: Override `enabled` flag ("true", "1", "yes")
- `STAGE1_H5_PATH`: Path to H5 file for SMILES extraction

## Integration

### With External Knowledge System

The module integrates with `external_knowledge_mirothink.py` at:
- **Lines 68-77**: Import statements
- **Lines 586-620**: BioKB pack building

Example integration:
```python
from cellscientist.core.external_knowledge_mirothink import retrieve_external_knowledge

# BioKB is automatically included if enabled in config
knowledge_pack = retrieve_external_knowledge(
    cfg=config,
    context_text="cancer cell proliferation experiment",
    stage="design",
    workspace_dir="./workspace"
)

# Access BioKB items
biokb_items = [item for item in knowledge_pack.items if item.source == "biokb"]
```

## SMILES Resolution

### Priority Order
1. **Config**: `cfg["literature"]["bio_kb"]["smiles_list"]`
2. **H5 File**: `cfg["paths"]["stage1_h5"]` or `$STAGE1_H5_PATH`
3. **Placeholder**: Returns `["SMILES_NOT_FOUND"]` (visible failure)

### Canonicalization

Uses RDKit if available:
```python
from rdkit import Chem

mol = Chem.MolFromSmiles(smiles)
canonical = Chem.MolToSmiles(mol, canonical=True)
inchikey = Chem.MolToInchiKey(mol)
```

Graceful fallback if RDKit unavailable:
```python
# Returns original SMILES, empty InChIKey
return {"canonical_smiles": smiles, "inchikey": None}
```

## API Clients

### ChEMBL Client

**Endpoints:**
- `/molecule.json?molecule_structures__standard_inchi_key={inchikey}`
- `/mechanism.json?molecule_chembl_id={molecule_id}`
- `/target/{target_id}.json`

**Caching:**
- Cache key: `sha1("chembl_targets:{smiles}:{inchikey}")`
- Cache path: `cache/chembl/{hash}.json`
- Cache format: `{"targets": [...]}`

**Rate Limiting:**
- Max mechanisms per molecule: 5
- Configurable timeout: 30s default

### Reactome Client

**Endpoints:**
- `/data/query/{gene_symbol}/pathways`
- `/data/pathway/{pathway_id}`

**Caching:**
- Cache key: `sha1("reactome_pathways:{gene_symbol}")`
- Cache path: `cache/reactome/{hash}.json`
- Cache format: `{"pathways": [...]}`

**Rate Limiting:**
- Max targets queried: 3
- Max pathways per target: 5
- Configurable timeout: 30s default

## Process Mapping

### Keyword Heuristics

Process classification uses keyword matching against pathway names:

```python
PATHWAY_TO_PROCESS_KEYWORDS = {
    "proliferation": [
        "cell cycle", "mitosis", "G1/S", "cyclin", "CDK",
        "MAPK", "ERK", "PI3K", "AKT", "mTOR", ...
    ],
    "apoptosis": [
        "apoptosis", "caspase", "BCL2", "p53", "TP53",
        "death receptor", "FAS", "cytochrome c", ...
    ],
    "emt": [
        "epithelial-mesenchymal", "EMT", "E-cadherin",
        "vimentin", "SNAIL", "TGF-beta", ...
    ],
    ...
}
```

### Confidence Scoring

```python
confidence = matched_keywords / total_keywords_for_process
```

Example:
- Pathway: "Synthesis of Prostaglandins (PG)"
- Matched keywords: ["prostaglandin", "synthesis"]
- Total keywords for apoptosis: 18
- Confidence: 2/18 ≈ 0.11

## Error Handling

### Strategy

1. **Timeout Protection**: All API calls use `@with_timeout` decorator
2. **Graceful Degradation**: Return empty lists on failure, never crash
3. **Visible Failures**: Use placeholders like `"SMILES_NOT_FOUND"` instead of silent fails
4. **Error Evidence Items**: On total failure, return error evidence item with EID "B0"

### Example Error Handling

```python
try:
    semantic_table = generate_biokb_semantic_table(cfg, "design", log)
except Exception as e:
    log(f"[BIOKB][ERROR] {e}")
    # Return minimal valid table
    semantic_table = {
        "stage": "design",
        "smiles_source": "error",
        "molecules": [],
        "summary": {"error": str(e)}
    }
```

## Testing

### Validation Commands

```bash
# Syntax check
python -m py_compile cellscientist/core/bio_kb/*.py

# Import check
python -c "from cellscientist.core.bio_kb import generate_biokb_semantic_table, persist_biokb_semantic_table, biokb_table_to_evidence_items"

# Config check
python -c "from cellscientist.core.bio_kb.config import BioKBConfig; cfg = BioKBConfig(enabled=True, smiles_list=['CCO']); print(f'Config OK: {cfg}')"
```

### Example Test

```python
from cellscientist.core.bio_kb import (
    generate_biokb_semantic_table,
    biokb_table_to_evidence_items
)

cfg = {
    'literature': {
        'bio_kb': {
            'enabled': True,
            'smiles_list': ['CCO'],  # Ethanol
            'chembl_enabled': True
        }
    }
}

# Generate semantic table
table = generate_biokb_semantic_table(cfg, "design", print)
print(f"Molecules: {len(table['molecules'])}")

# Convert to evidence items
items = biokb_table_to_evidence_items(table)
print(f"Evidence items: {len(items)}")
print(f"First item EID: {items[0]['eid']}")
```

## Backward Compatibility

### Guarantees

1. **Optional by Default**: BioKB disabled if `enabled: false` or config missing
2. **No Breaking Changes**: Existing `EvidenceItem` and `KnowledgePack` unchanged
3. **Independent**: Web literature still works if BioKB fails
4. **API Key Safety**: No keys logged to console

### Migration from Old bio_kb.py

The old monolithic `bio_kb.py` is replaced with the new modular structure:

**Old (deprecated):**
```python
from cellscientist.core.bio_kb import generate_biokb_semantic_table
```

**New (same import path, different implementation):**
```python
from cellscientist.core.bio_kb import generate_biokb_semantic_table
```

No code changes required for consumers!

## Performance Considerations

### Optimization Strategies

1. **Caching**: All API responses cached locally
2. **Rate Limiting**: Max queries per molecule configured
3. **Timeout Protection**: 30s default timeout prevents hanging
4. **Batch Processing**: Multiple SMILES processed in single run

### Typical Performance

- SMILES canonicalization: <1ms (with RDKit)
- ChEMBL target query: 2-5s (first call), <1ms (cached)
- Reactome pathway query: 1-3s (first call), <1ms (cached)
- Process mapping: <1ms (keyword matching)

**Total per molecule**: ~5-10s first run, ~1-2s cached

## Troubleshooting

### Common Issues

#### 1. RDKit Not Available
**Symptom**: Warnings about missing RDKit
**Solution**: Module works without RDKit (graceful fallback)
```bash
pip install rdkit  # Optional, improves SMILES canonicalization
```

#### 2. API Timeouts
**Symptom**: `[BIOKB][WARN] ChEMBL query failed`
**Solution**: Increase timeout in config
```python
"chembl_timeout": 60  # Increase from 30s
```

#### 3. No SMILES Found
**Symptom**: `"SMILES_NOT_FOUND"` in output
**Solution**: Add SMILES to config or set H5 path
```python
"smiles_list": ["CCO", "CC(C)O"]
```

#### 4. Cache Directory Issues
**Symptom**: Cache writes failing
**Solution**: Ensure writable cache directory
```python
"cache_dir": "/path/to/writable/directory"
```

## References

- [Biomni Architecture](https://github.com/mosabutey/Biomni)
- [ChEMBL API Docs](https://chembl.gitbook.io/chembl-interface-documentation/web-services)
- [Reactome API Docs](https://reactome.org/ContentService/)
- [CellCausal External Knowledge System](../external_knowledge_mirothink.py)
