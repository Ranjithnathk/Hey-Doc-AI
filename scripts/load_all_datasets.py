from datasets import load_from_disk
from pathlib import Path

# Paths (convert to file URIs for OS compatibility)
pubmed_path = Path("datasets/pubmed_qa").resolve()
meddialog_path = Path("datasets/meddialog").resolve()
mts_path = Path("datasets/MTS_Dialogue-Clinical_Note").resolve()

# Load datasets
print("Loading pubmed_qa...")
pubmed_qa = load_from_disk(pubmed_path)

print("Loading meddialog...")
meddialog = load_from_disk(meddialog_path)

print("Loading MTS Dialogue Clinical Note...")
mts = load_from_disk(mts_path)

# Show sample
print("\n Sample pubmed_qa:", pubmed_qa['train'][0])
print("\n Sample meddialog:", meddialog['train'][0])
print("\n Sample mts:", mts['train'][0])
