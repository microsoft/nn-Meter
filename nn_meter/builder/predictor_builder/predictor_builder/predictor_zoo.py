# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from sklearn.ensemble import RandomForestRegressor


__MODEL_ZOO__ = {
    "convbnrelu": {
        "cpu": {
            "max_depth": 70,
            "n_estimators": 320,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 6,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 80,
            "n_estimators": 550,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 100,
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        }
    },
    "dwconvbnrelu": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 240,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 6,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 40,
            "n_estimators": 240,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 7,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 100,
            "n_estimators": 650,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        }
    },
    "fc": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 70,
            "n_estimators": 330,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 4,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 70,
            "n_estimators": 330,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 4,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        }
    },
    "channelshuffle": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "se": {
        "cpu": {
            "max_depth": 20,
            "n_estimators": 290,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 110,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "maxpool": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 210,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "globalavgpool": {
        "cpu": {
            "max_depth": 70,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "hswish": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 110,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "avgpool": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 390,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "bnrelu": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "relu": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "bn": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 390,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "concat": {
        "cpu": {
            "max_depth": 100,
            "n_estimators": 690,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 100,
            "n_estimators": 690,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "addrelu": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "gpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "vpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "split": {
        "cpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    }
}


def get_model(block_type, hardware):
    try:
        model_param = __MODEL_ZOO__[block_type][hardware]
        model = RandomForestRegressor(**model_param)
    except:
        model = RandomForestRegressor(
            max_depth = 50,
            n_estimators = 370,
            min_samples_leaf = 1,
            min_samples_split = 2,
            max_features = 2,
            oob_score = True,
            random_state = 10,
        )
    return model
