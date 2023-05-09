"""Load data from json, as an input."""
from typing import Dict
import json


def load_texts(data_hparams: Dict) -> Dict[str, str]:
    """Load json.

     It should be in the format:
    {
        'id_key': text,
        ...
    }

    """
    f = open(data_hparams['kwargs']['path_to_data'])
    data = json.load(f)
    f.close()
    return data
