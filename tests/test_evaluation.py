from diag_opt.evaluation import (
    compare_baseline_vs_optimized,
    evaluate_on_test,
    fit_serving_model,
)


def test_evaluate_on_test_metricas_validas(dataset):
    res = evaluate_on_test("SVM", {"C": 1.0, "gamma": "scale", "kernel": "rbf"}, dataset)
    for k in ("accuracy", "recall_maligno", "precision_maligno", "f1_maligno", "roc_auc"):
        assert 0.0 <= res.metrics[k] <= 1.0
    # SVM decente no test set
    assert res.metrics["accuracy"] > 0.9
    assert res.false_negatives + res.false_positives < len(dataset.y_test)


def test_compare_retorna_baseline_e_otimizado(dataset):
    comp = compare_baseline_vs_optimized("SVM", {"C": 10, "gamma": 0.05, "kernel": "rbf"}, dataset)
    assert set(comp) == {"baseline", "optimized"}
    assert comp["baseline"].params != comp["optimized"].params


def test_fit_serving_model_tem_predict_proba(dataset):
    pipe = fit_serving_model("SVM", {"C": 1.0, "gamma": "scale", "kernel": "rbf"}, dataset)
    proba = pipe.predict_proba(dataset.X_test.iloc[:3])
    assert proba.shape == (3, 2)
