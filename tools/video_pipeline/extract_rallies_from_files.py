"""
Script per estrarre informazioni sui rally dai file video esportati
e generare un JSON nel formato richiesto da eval_rallies_gt.py

I file hanno nomi tipo: rally_01_right_1.mp4 con timestamp nel nome o metadati
"""

import json
import re
from pathlib import Path
from typing import List, Dict


def parse_rally_filename(filename: str) -> Dict:
    """
    Estrae informazioni dal nome del file.
    Formati possibili:
    - rally_01_right_1.mp4
    - rally_01_right_010.00-1017.7.mp4
    - rally_01_right_1_010.00-1017.7.mp4
    """
    # Rimuovi estensione
    name = Path(filename).stem
    
    # Pattern per estrarre numero, side, e timestamp
    # Esempio: rally_01_right_1 o rally_01_right_010.00-1017.7
    pattern1 = r'rally_(\d+)_(left|right)_(\d+)(?:_([\d.]+)-([\d.]+))?'
    match = re.match(pattern1, name)
    
    if match:
        idx = int(match.group(1))
        side = match.group(2)
        suffix = match.group(3)
        start_str = match.group(4)
        end_str = match.group(5)
        
        if start_str and end_str:
            return {
                'idx': idx,
                'side': side,
                'start': float(start_str),
                'end': float(end_str)
            }
    
    # Pattern alternativo: timestamp direttamente nel nome
    # Esempio: rally_01_right_010.00-1017.7
    pattern2 = r'rally_(\d+)_(left|right)_([\d.]+)-([\d.]+)'
    match = re.match(pattern2, name)
    
    if match:
        idx = int(match.group(1))
        side = match.group(2)
        start = float(match.group(3))
        end = float(match.group(4))
        return {
            'idx': idx,
            'side': side,
            'start': start,
            'end': end
        }
    
    return None


def extract_rallies_from_directory(directory: Path) -> List[Dict]:
    """
    Legge tutti i file rally dalla directory e estrae start/end.
    """
    rallies = []
    
    # Cerca tutti i file che iniziano con "rally_"
    for file_path in sorted(directory.glob("rally_*")):
        if file_path.is_file():
            info = parse_rally_filename(file_path.name)
            if info:
                rallies.append({
                    'start': info['start'],
                    'end': info['end'],
                    'side': info['side']
                })
            else:
                print(f"Impossibile parsare: {file_path.name}")
    
    # Ordina per start time
    rallies.sort(key=lambda x: x['start'])
    
    return rallies


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Estrae informazioni sui rally dai file video e genera JSON"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory contenente i file rally (es. Test1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path del file JSON di output"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Directory non trovata: {input_dir}")
        return
    
    print(f"Leggo file da: {input_dir}")
    rallies = extract_rallies_from_directory(input_dir)
    
    if not rallies:
        print("Nessun rally trovato!")
        return
    
    print(f"Trovati {len(rallies)} rally")
    
    # Salva JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rallies, f, indent=2, ensure_ascii=False)
    
    print(f"JSON salvato in: {output_path}")
    print(f"\nPrimi 3 rally:")
    for i, r in enumerate(rallies[:3], 1):
        print(f"  {i}. start={r['start']:.2f}s, end={r['end']:.2f}s, side={r['side']}")


if __name__ == "__main__":
    main()

