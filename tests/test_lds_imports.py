import importlib
import pickle
from pathlib import Path

import pytest
import torch

from ctd.data_modeling.models import SAE
from ctd.data_modeling.models.SAE.lds import LDSSAE


def test_sae_module_filenames_do_not_collide_by_case():
    names = [path.name.casefold() for path in Path(SAE.__file__).parent.glob("*.py")]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", ["LDSSAE", "LinearCell", "RNN"])
def test_historical_uppercase_module_resolves_to_canonical_classes(name):
    canonical = importlib.import_module("ctd.data_modeling.models.SAE.lds")
    historical = importlib.import_module("ctd.data_modeling.models.SAE.LDS")
    assert historical is canonical
    assert SAE.LDS is canonical
    assert getattr(historical, name) is getattr(canonical, name)


def test_legacy_lds_pickle_loads_and_preserves_predictions():
    model = LDSSAE(
        dataset="NBFF",
        encoder_size=3,
        encoder_window=2,
        heldin_size=4,
        heldout_size=6,
        latent_size=2,
        lr=0.001,
        weight_decay=0.0,
        dropout=0.0,
        input_size=1,
    ).eval()
    spikes = torch.ones(2, 5, 4)
    inputs = torch.ones(2, 5, 1)
    # Protocol 2 stores GLOBAL module names as newline-delimited strings.
    # Recreate the historical class paths for the model and its decoder.
    payload = pickle.dumps(model, protocol=2)
    canonical = b"ctd.data_modeling.models.SAE.lds\n"
    historical = b"ctd.data_modeling.models.SAE.LDS\n"
    assert canonical in payload
    legacy_payload = payload.replace(canonical, historical)
    restored = pickle.loads(legacy_payload)
    assert type(restored) is LDSSAE
    assert restored.state_dict().keys() == model.state_dict().keys()
    with torch.no_grad():
        for actual, expected in zip(restored(spikes, inputs), model(spikes, inputs)):
            torch.testing.assert_close(actual, expected)
