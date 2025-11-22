import json
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------- UTILITIES ----------

def mmss_to_seconds(mmss: str) -> float:
    """Converte 'MM:SS' in secondi (float)."""
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


@dataclass
class GTRally:
    id: int
    serve_time: float
    point_time: float
    serve_team: str
    winner_team: str
    score_after_str: Optional[str]


@dataclass
class PredRally:
    idx: int
    start: float
    end: float


def load_gt(path: str) -> List[GTRally]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_rallies: List[GTRally] = []
    for item in data:
        # Supporta due formati:
        # 1. Vecchio formato: serve_time_str/point_time_str (MM:SS)
        # 2. Nuovo formato: start/end (secondi)
        if "start" in item and "end" in item:
            # Nuovo formato verificato
            serve_time = float(item["start"])
            point_time = float(item["end"])
            serve_team = item.get("side", "unknown")  # side può essere left/right
            winner_team = item.get("side", "unknown")
        else:
            # Vecchio formato con stringhe MM:SS
            serve_time = mmss_to_seconds(item["serve_time_str"])
            point_time = mmss_to_seconds(item["point_time_str"])
            serve_team = item["serve_team"]
            winner_team = item["winner_team"]
        
        gt_rallies.append(
            GTRally(
                id=item["id"],
                serve_time=serve_time,
                point_time=point_time,
                serve_team=serve_team,
                winner_team=winner_team,
                score_after_str=item.get("score_after_str"),
            )
        )
    return gt_rallies


def load_pred(path: str) -> List[PredRally]:
    """
    Si aspetta un JSON tipo:
    [
      {"start": 1010.0, "end": 1017.7, "side": "right", ...},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred: List[PredRally] = []
    for idx, item in enumerate(data):
        pred.append(
            PredRally(
                idx=idx,
                start=float(item["start"]),
                end=float(item["end"]),
            )
        )
    return pred


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    inter = max(0.0, inter_end - inter_start)
    if inter <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    gt: List[GTRally],
    pred: List[PredRally],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[GTRally, PredRally, float]], List[GTRally], List[PredRally]]:
    """
    Match greedy basato su IoU degli intervalli [serve_time, point_time] vs [start, end].
    """
    matches: List[Tuple[GTRally, PredRally, float]] = []
    unmatched_gt = set(range(len(gt)))
    unmatched_pred = set(range(len(pred)))

    # lista di tutte le coppie con IoU > 0
    candidates = []
    for gi, g in enumerate(gt):
        g_start, g_end = g.serve_time, g.point_time
        for pi, p in enumerate(pred):
            p_start, p_end = p.start, p.end
            iou = interval_iou(g_start, g_end, p_start, p_end)
            if iou > 0.0:
                candidates.append((iou, gi, pi))

    # ordina per IoU decrescente
    candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gi, pi in candidates:
        if iou < iou_threshold:
            break
        if gi in unmatched_gt and pi in unmatched_pred:
            matches.append((gt[gi], pred[pi], iou))
            unmatched_gt.remove(gi)
            unmatched_pred.remove(pi)

    return (
        matches,
        [gt[i] for i in unmatched_gt],
        [pred[i] for i in unmatched_pred],
    )


# ---------- METRICHE ----------

def eval_rallies(
    gt_path: str,
    pred_path: str,
    iou_threshold: float = 0.3,
) -> None:
    gt = load_gt(gt_path)
    pred = load_pred(pred_path)

    matches, miss_gt, miss_pred = greedy_match(gt, pred, iou_threshold=iou_threshold)

    tp = len(matches)
    fn = len(miss_gt)
    fp = len(miss_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # offset medio start/end
    start_offsets = []
    end_offsets = []
    for g, p, _iou in matches:
        start_offsets.append(p.start - g.serve_time)
        end_offsets.append(p.end - g.point_time)

    mean_start_offset = sum(start_offsets) / len(start_offsets) if start_offsets else 0.0
    mean_end_offset = sum(end_offsets) / len(end_offsets) if end_offsets else 0.0

    print("===== VALUTAZIONE RALLY =====")
    print(f"Ground truth rally: {len(gt)}")
    print(f"Predetti:           {len(pred)}")
    print(f"Match (TP):         {tp}")
    print(f"False Negative:     {fn}")
    print(f"False Positive:     {fp}")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"Offset medio start: {mean_start_offset:.2f} s (pred - GT)")
    print(f"Offset medio end:   {mean_end_offset:.2f} s (pred - GT)")
    print("\nDettaglio match:")
    for g, p, iou in matches:
        print(
            f"- GT #{g.id} [{g.serve_time:.1f}-{g.point_time:.1f}] "
            f"<-> Pred #{p.idx} [{p.start:.1f}-{p.end:.1f}]  IoU={iou:.2f}"
        )

    if miss_gt:
        print("\nRally GT NON trovati:")
        for g in miss_gt:
            print(f"- GT #{g.id} [{g.serve_time:.1f}-{g.point_time:.1f}]  winner={g.winner_team}")

    if miss_pred:
        print("\nRally predetti senza match (FP):")
        for p in miss_pred:
            print(f"- Pred #{p.idx} [{p.start:.1f}-{p.end:.1f}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Valuta i rally rispetto alla ground truth.")
    parser.add_argument("--gt", required=True, help="Path JSON ground truth")
    parser.add_argument("--pred", required=True, help="Path JSON rally predetti")
    parser.add_argument("--iou", type=float, default=0.3, help="Soglia IoU per considerare un match")
    args = parser.parse_args()

    eval_rallies(args.gt, args.pred, iou_threshold=args.iou)

