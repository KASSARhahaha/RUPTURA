import pytest
from unittest.mock import patch
from ruptura import Components, Breakthrough
import numpy as np

@pytest.fixture
def basic_components():
    return Components([{
            "MoleculeName": "Helium",
            "CarrierGas": True,
            "GasPhaseMolFraction": 0.9,
            "isotherms": []
        }, {
            "MoleculeName": "nC7",
            "GasPhaseMolFraction": 0.05,
            "isotherms": [["Langmuir", 1.09984, 6.55857e-5], ["Langmuir", 0.19466, 8.90731e-07]]
        }, {
            "MoleculeName": "C6m2",
            "GasPhaseMolFraction": 0.05,
            "isotherms": [["Langmuir", 1.22228, 3.90895e-05], ["Langmuir", 0.481726, 9.64046e-08]]
        }])

@pytest.fixture
def breakthrough_instance(basic_components):
    components = Components([
        {'MoleculeName': 'Component1', 'GasPhaseMolFraction': 0.5},
        {'MoleculeName': 'Component2', 'GasPhaseMolFraction': 0.5}
    ])
    return Breakthrough(components=basic_components)

def test_compute(breakthrough_instance):
    expected_output = np.random.rand(100, 20)  # Simulate an output shape for the breakthrough data

    with patch('_ruptura.Breakthrough.compute', return_value=expected_output) as mock_compute:
        result = breakthrough_instance.compute()
        np.testing.assert_array_equal(result, expected_output)
        mock_compute.assert_called_once()


def test_compute_generates_profiles_without_patch(basic_components):
    breakthrough = Breakthrough(
        components=basic_components,
        NumberOfTimeSteps=12,
        NumberOfGridPoints=6,
        TimeStep=0.05,
        ColumnLength=0.6,
    )

    data = breakthrough.compute()
    expected_columns = 8 + len(basic_components._components) * 6
    assert data.shape == (12, 6, expected_columns)
    dimensionless_time = data[:, -1, 0]
    assert np.all(np.diff(dimensionless_time) >= 0)
    assert np.all(data[:, :, 2] == pytest.approx(433.0))


def test_batch_compute_multiple_profiles(basic_components):
    breakthrough = Breakthrough(
        components=basic_components,
        NumberOfTimeSteps=10,
        NumberOfGridPoints=5,
        TimeStep=0.1,
        ColumnLength=0.5,
    )

    fractions = np.array([
        [0.85, 0.1, 0.05],
        [0.7, 0.2, 0.1],
    ])

    batch = breakthrough.batch_compute(fractions)
    expected_columns = 8 + len(basic_components._components) * 6
    assert batch.shape == (2, 10, 5, expected_columns)
    assert not np.allclose(batch[0], batch[1])


def test_batch_compute_with_parameter_overrides(basic_components):
    breakthrough = Breakthrough(
        components=basic_components,
        NumberOfTimeSteps=10,
        NumberOfGridPoints=5,
    )

    fractions = np.array([
        [0.8, 0.15, 0.05],
        [0.6, 0.25, 0.15],
    ])

    overrides = [
        {"NumberOfTimeSteps": 6, "NumberOfGridPoints": 4},
        {"NumberOfTimeSteps": 6, "NumberOfGridPoints": 4},
    ]

    batch = breakthrough.batch_compute(fractions, parameter_overrides=overrides)
    expected_columns = 8 + len(basic_components._components) * 6
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2, 6, 4, expected_columns)