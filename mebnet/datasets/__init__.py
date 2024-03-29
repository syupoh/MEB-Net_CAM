from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhk03 import CUHK03

from .personX import PersonX
# from .personX_spgan import PersonX_SPGAN
# from .personX_sda import PersonX_SDA
from .target_training import Target_Training
from .target_validation import Target_Validation
# from .target_test import Target_Test


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17': MSMT17,
    'cuhk03': CUHK03,
    'personx': PersonX,
    # 'personx_spgan': PersonX_SPGAN,
    # 'personx_sda': PersonX_SDA,
    'target_train': Target_Training,
    'target_val': Target_Validation,
    # 'target_test': Target_Test,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
