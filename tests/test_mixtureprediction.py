import pytest
from unittest.mock import patch
from ruptura import Components, MixturePrediction
import numpy as np

@pytest.fixture
def basic_components():
    return Components([{
            "MoleculeName": "nC7",
            "GasPhaseMolFraction": 0.5,
            "isotherms": [["Langmuir", 1.09984, 6.55857e-5], ["Langmuir", 0.19466, 8.90731e-07]]
        }, {
            "MoleculeName": "C6m2",
            "GasPhaseMolFraction": 0.5,
            "isotherms": [["Langmuir", 1.22228, 3.90895e-05], ["Langmuir", 0.481726, 9.64046e-08]]
        }])

@pytest.fixture
def mixture_prediction(basic_components):
    return MixturePrediction(components=basic_components)

def test_initialization(mixture_prediction):
    assert mixture_prediction.shape == (100, 2, 6)
    assert mixture_prediction.DisplayName == "Column"
    assert mixture_prediction._MixturePrediction is not None
    assert mixture_prediction.data is None

def test_compute(mixture_prediction):
    with patch('_ruptura.MixturePrediction.compute') as mock_compute:
        mock_data = np.random.rand(100, 2, 6)  # Mocking the computation result
        mock_compute.return_value = mock_data

        result = mixture_prediction.compute()
        np.testing.assert_array_equal(result, mock_data)
        mock_compute.assert_called_once()


def test_compute_generates_profiles_without_patch(basic_components):
    prediction = MixturePrediction(
        components=basic_components,
        PressureStart=1.0,
        PressureEnd=10.0,
        NumberOfPressurePoints=5,
        PressureScale="linear",
    )

    data = prediction.compute()
    assert data.shape == (5, 2, 6)
    expected_pressures = np.linspace(1.0, 10.0, 5)
    expected_grid = np.broadcast_to(expected_pressures[:, None], data[:, :, 0].shape)
    np.testing.assert_allclose(data[:, :, 0], expected_grid)
    assert np.all(data[:, :, 1] > 0)
    assert not np.allclose(data[:, :, 2], 0)


def test_batch_compute_multiple_mixtures(basic_components):
    prediction = MixturePrediction(
        components=basic_components,
        PressureStart=5.0,
        PressureEnd=50.0,
        NumberOfPressurePoints=4,
        PressureScale="linear",
    )

    fractions = np.array([[0.8, 0.2], [0.2, 0.8]])
    batch = prediction.batch_compute(fractions)

    assert batch.shape == (2, 4, 2, 6)
    expected_pressures = np.linspace(5.0, 50.0, 4)
    expected_grid = np.broadcast_to(expected_pressures[None, :, None], batch[:, :, :, 0].shape)
    np.testing.assert_allclose(batch[:, :, :, 0], expected_grid)
    assert not np.allclose(batch[0], batch[1])


def test_batch_compute_with_parameter_overrides(basic_components):
    prediction = MixturePrediction(
        components=basic_components,
        PressureStart=1.0,
        PressureEnd=10.0,
        NumberOfPressurePoints=3,
        PressureScale="linear",
    )

    fractions = np.array([[0.6, 0.4], [0.4, 0.6]])
    overrides = [
        {"PressureEnd": 20.0, "NumberOfPressurePoints": 4},
        {"PressureEnd": 40.0, "NumberOfPressurePoints": 4},
    ]

    batch = prediction.batch_compute(fractions, parameter_overrides=overrides)

    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2, 4, 2, 6)
    np.testing.assert_allclose(batch[0, :, 0, 0], np.linspace(1.0, 20.0, 4))
    np.testing.assert_allclose(batch[1, :, 0, 0], np.linspace(1.0, 40.0, 4))