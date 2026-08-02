import numpy as np

from models.ai_detectors.calibration import fit_temperature, negative_log_likelihood


def test_temperature_of_one_is_a_no_op():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert negative_log_likelihood(1.0, logits, np.array([0, 1])) < negative_log_likelihood(5.0, logits, np.array([0, 1]))


def test_fit_temperature_reduces_overconfidence_on_noisy_labels():
    """Simulate an overconfident model: very large-magnitude logits that are
    only right 70% of the time. A well-fit temperature should push T > 1
    (softening the distribution), since blind overconfidence is penalized
    more heavily by NLL than a softer, better-calibrated distribution.
    """
    rng = np.random.default_rng(0)
    n = 200
    true_indices = rng.integers(0, 2, size=n)
    correct_mask = rng.random(n) < 0.7
    predicted_indices = np.where(correct_mask, true_indices, 1 - true_indices)
    logits = np.zeros((n, 2))
    for i, predicted in enumerate(predicted_indices):
        logits[i, predicted] = 8.0  # very confident, whether right or wrong
        logits[i, 1 - predicted] = -8.0

    fitted_temperature = fit_temperature(logits, true_indices)
    assert fitted_temperature > 1.0

    nll_uncalibrated = negative_log_likelihood(1.0, logits, true_indices)
    nll_calibrated = negative_log_likelihood(fitted_temperature, logits, true_indices)
    assert nll_calibrated < nll_uncalibrated


def test_fit_temperature_stays_near_one_for_already_well_calibrated_logits():
    rng = np.random.default_rng(1)
    n = 500
    true_indices = rng.integers(0, 2, size=n)
    # Logits whose softmax matches the true 80% accuracy already:
    # sigmoid(2 * 0.693) = 0.8, so +/-0.693 is genuinely well-calibrated here.
    logit_magnitude = 0.5 * np.log(0.8 / 0.2)
    logits = np.zeros((n, 2))
    for i, true_index in enumerate(true_indices):
        correct = rng.random() < 0.8
        predicted = true_index if correct else 1 - true_index
        logits[i, predicted] = logit_magnitude
        logits[i, 1 - predicted] = -logit_magnitude

    fitted_temperature = fit_temperature(logits, true_indices)
    assert 0.7 < fitted_temperature < 1.4
