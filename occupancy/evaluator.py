
from dataclasses import dataclass


@dataclass
class EvaluationResult:

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    accuracy: float
    precision: float
    recall: float
    f1_score: float


class OccupancyEvaluator:
    """
    Evaluates workstation occupancy predictions.
    """

    def evaluate(
        self,
        actual: list[str],
        predicted: list[str]
    ) -> EvaluationResult:

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted lists must have equal lengths."
            )

        valid_states = {"OCCUPIED", "VACANT"}

        if any(state not in valid_states for state in actual):
            raise ValueError("Invalid actual state.")

        if any(state not in valid_states for state in predicted):
            raise ValueError("Invalid predicted state.")

        tp = tn = fp = fn = 0

        for true_state, predicted_state in zip(actual, predicted):

            if true_state == "OCCUPIED" and predicted_state == "OCCUPIED":
                tp += 1

            elif true_state == "VACANT" and predicted_state == "VACANT":
                tn += 1

            elif true_state == "VACANT" and predicted_state == "OCCUPIED":
                fp += 1

            elif true_state == "OCCUPIED" and predicted_state == "VACANT":
                fn += 1

        total = len(actual)

        accuracy = (tp + tn) / total if total else 0.0

        precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        return EvaluationResult(
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
