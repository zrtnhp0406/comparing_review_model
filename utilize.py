from collections import defaultdict

# NHÃN ĐÃ MAP: [-1, 0, 1] → [0, 1, 2]
# Khi mô hình dự đoán, nó chỉ dự đoán các nhãn [0, 1, 2]
# - 0: Hòa (draw) - tương ứng với nhãn gốc -1
# - 1: Bia 1 thắng (beer 1 wins) - tương ứng với nhãn gốc 0  
# - 2: Bia 2 thắng (beer 2 wins) - tương ứng với nhãn gốc 1
labels = [0, 1, 2]
positive_labels = [0, 1, 2]
negative_label = None  # Không còn negative label nữa, tất cả đều là positive

def compute_metrics(true_labels, pred_labels):

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(true_labels, pred_labels):

        if true == pred:
            if true in positive_labels:
                tp[true] += 1

        else:

            # Nếu dự đoán sai
            if true in positive_labels and pred in positive_labels:
                fn[true] += 1
                fp[pred] += 1

    return tp, fp, fn


def calculate_class_metrics(tp, fp, fn):

    class_metrics = {}

    for label in positive_labels:

        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label]
        }

    return class_metrics


def calculate_summary_metrics(tp, fp, fn, class_metrics):

    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(positive_labels)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(positive_labels)
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(positive_labels)

    total_tp = sum(tp[label] for label in positive_labels)
    total_fp = sum(fp[label] for label in positive_labels)
    total_fn = sum(fn[label] for label in positive_labels)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def evaluate_overall(eval_dataset, aspect_predictions):

    aspects = ["appearance", "aroma", "palate", "taste"]

    all_preds = []
    all_labels = []

    for i in range(len(eval_dataset)):
        for aspect in aspects:

            all_preds.append(aspect_predictions[aspect][i])
            all_labels.append(eval_dataset[i][aspect])

    tp, fp, fn = compute_metrics(all_labels, all_preds)

    class_metrics = calculate_class_metrics(tp, fp, fn)

    summary_metrics = calculate_summary_metrics(tp, fp, fn, class_metrics)

    return {
        "summary": summary_metrics,
        "class_metrics": class_metrics
    }

def evaluate_aspect_wise(eval_dataset, aspect_predictions):

    aspects = ["appearance", "aroma", "palate", "taste"]

    results = {}

    for aspect in aspects:

        preds = aspect_predictions[aspect]
        labels = [sample[aspect] for sample in eval_dataset]

        tp, fp, fn = compute_metrics(labels, preds)

        class_metrics = calculate_class_metrics(tp, fp, fn)

        summary_metrics = calculate_summary_metrics(tp, fp, fn, class_metrics)

        results[aspect] = {
            "summary": summary_metrics,
            "class_metrics": class_metrics
        }

    return results

def print_class_report(class_metrics):
    print("\nClass Metrics")
    print("Label | Precision | Recall | F1 | TP | FP | FN")
    print("-"*50)

    # Mapping ngược: [0, 1, 2] → [-1, 0, 1]
    decode_map = {0: -1, 1: 0, 2: 1}
    
    for label, m in class_metrics.items():
        original_label = decode_map.get(label, label)
        print(
            f"{original_label:5} | "
            f"{m['precision']:.4f} | "
            f"{m['recall']:.4f} | "
            f"{m['f1']:.4f} | "
            f"{m['tp']:3} | "
            f"{m['fp']:3} | "
            f"{m['fn']:3}"
        )