
from occupancy.evaluator import OccupancyEvaluator


def test_evaluation():

    evaluator = OccupancyEvaluator()

    actual = [
        "OCCUPIED",
        "VACANT",
        "OCCUPIED",
        "VACANT"
    ]

    predicted = [
        "OCCUPIED",
        "VACANT",
        "VACANT",
        "OCCUPIED"
    ]

    result = evaluator.evaluate(actual, predicted)

    assert result.true_positive == 1
    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1

    assert result.accuracy == 0.5
