import json
import sys

def extract_code_from_ipynb(ipynb_path, out_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if source.strip():
                code_cells.append(source)
                
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n\n# --- CELL ---\n\n'.join(code_cells))
        
if __name__ == '__main__':
    extract_code_from_ipynb('01. Data Extraction.ipynb', '01_extracted.py')
    extract_code_from_ipynb('02. DatasetCreation.ipynb', '02_extracted.py')
    print("Extraction complete.")
