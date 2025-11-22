"""
Script per generare Ground Truth da un file JSON di rally esportato.

Uso:
    python -m tools.video_pipeline.make_gt_from_rallies \
        --input results/rallies_1010_1600.json \
        --output tools/video_pipeline/ground_truth/gt_new.json \
        [--keep-ids keep_ids.txt]

Se --keep-ids non è specificato, include tutti i rally.
Se --keep-ids è specificato, include solo i rally con id elencati nel file (uno per riga).
"""

import json
import argparse
from pathlib import Path
from typing import List, Set, Optional


def load_rallies_json(input_path: str) -> List[dict]:
    """Carica i rally da un file JSON esportato."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON deve essere una lista di rally, trovato: {type(data)}")
    
    return data


def load_keep_ids(keep_ids_path: Optional[str]) -> Optional[Set[int]]:
    """
    Carica gli ID dei rally da mantenere da un file di testo.
    Formato: un numero per riga (1-based, come appaiono nella lista).
    """
    if keep_ids_path is None:
        return None
    
    keep_ids_path = Path(keep_ids_path)
    if not keep_ids_path.exists():
        print(f"File keep_ids.txt non trovato: {keep_ids_path}")
        print("   Includerò tutti i rally.")
        return None
    
    keep_ids = set()
    with open(keep_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    # Gli ID nel file sono 1-based (come appaiono nella lista)
                    keep_ids.add(int(line))
                except ValueError:
                    print(f"Riga ignorata (non è un numero): {line}")
    
    return keep_ids


def generate_gt(
    rallies: List[dict],
    keep_ids: Optional[Set[int]] = None
) -> List[dict]:
    """
    Genera una Ground Truth nel formato standard.
    
    Args:
        rallies: Lista di rally dal JSON esportato
        keep_ids: Set di ID da mantenere (1-based). Se None, mantiene tutti.
    
    Returns:
        Lista di dict con formato: {"id": int, "start": float, "end": float, "side": str}
    """
    gt_rallies = []
    gt_id = 1
    
    for idx, rally in enumerate(rallies, start=1):
        # Se keep_ids è specificato, salta i rally non elencati
        if keep_ids is not None and idx not in keep_ids:
            continue
        
        # Estrai i dati necessari
        start = float(rally.get("start", 0.0))
        end = float(rally.get("end", 0.0))
        side = rally.get("side", "unknown")
        
        # Validazione base
        if end <= start:
            print(f"Rally #{idx} ignorato: end ({end}) <= start ({start})")
            continue
        
        gt_rallies.append({
            "id": gt_id,
            "start": start,
            "end": end,
            "side": side
        })
        gt_id += 1
    
    return gt_rallies


def main():
    parser = argparse.ArgumentParser(
        description="Genera Ground Truth da un file JSON di rally esportato"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path del file JSON di rally esportato (es. results/rallies_1010_1600.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path del file GT di output (es. tools/video_pipeline/ground_truth/gt_new.json)"
    )
    parser.add_argument(
        "--keep-ids",
        type=str,
        default=None,
        help="Path opzionale di un file keep_ids.txt con gli ID dei rally da mantenere (uno per riga, 1-based)"
    )
    
    args = parser.parse_args()
    
    # Carica i rally
    print(f"Carico rally da: {args.input}")
    try:
        rallies = load_rallies_json(args.input)
        print(f"   Trovati {len(rallies)} rally")
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        return
    
    # Carica keep_ids se specificato
    keep_ids = None
    if args.keep_ids:
        print(f"Carico keep_ids da: {args.keep_ids}")
        keep_ids = load_keep_ids(args.keep_ids)
        if keep_ids:
            print(f"   Mantenuti {len(keep_ids)} ID: {sorted(keep_ids)}")
    
    # Genera GT
    print(f"\nGenero Ground Truth...")
    gt_rallies = generate_gt(rallies, keep_ids)
    
    if not gt_rallies:
        print("Nessun rally valido per la GT!")
        return
    
    print(f"   Generati {len(gt_rallies)} rally per la GT")
    
    # Salva GT
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gt_rallies, f, indent=2, ensure_ascii=False)
    
    print(f"\nGround Truth salvata in: {output_path}")
    print(f"\nRiepilogo:")
    print(f"   Rally totali nel JSON: {len(rallies)}")
    print(f"   Rally nella GT: {len(gt_rallies)}")
    
    if gt_rallies:
        print(f"\n   Primi 3 rally GT:")
        for r in gt_rallies[:3]:
            dur = r["end"] - r["start"]
            print(f"     #{r['id']}: {r['start']:.2f}s -> {r['end']:.2f}s ({dur:.2f}s, side={r['side']})")


if __name__ == "__main__":
    main()

