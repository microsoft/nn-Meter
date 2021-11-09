# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from sklearn.ensemble import RandomForestRegressor


def get_model(hardware, kernel):
    model = None
    if kernel == "convbnrelu":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=70,
                n_estimators=320,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=6,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=80,
                n_estimators=550,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                n_jobs=32,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=100,
                n_estimators=500,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                n_jobs=32,
                random_state=10,
            )
    if kernel == "dwconvbnrelu":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=240,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=6,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=40,
                n_estimators=240,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=7,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=100,
                n_estimators=650,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                n_jobs=32,
                random_state=10,
            )
    if kernel == "fc":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=70,
                n_estimators=330,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=4,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=70,
                n_estimators=330,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=4,
                oob_score=True,
                n_jobs=32,
                random_state=10,
            )
    if kernel == "channelshuffle":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "se":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=20,
                n_estimators=290,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=110,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "maxpool":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=210,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
    if kernel == "globalavgpool":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=70,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "hswish":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=110,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )

    if kernel == "avgpool":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=390,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
    if kernel == "bnrelu":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=570,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "relu":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "bn":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=370,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=390,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )
    if kernel == "concat":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=100,
                n_estimators=690,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )
        if hardware == "gpu":
            model = RandomForestRegressor(
                max_depth=100,
                n_estimators=690,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=5,
                oob_score=True,
                random_state=10,
            )

    if kernel == "addrelu":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=570,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=3,
                oob_score=True,
                random_state=10,
            )
        if hardware == "addrelu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=570,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=3,
                oob_score=True,
                random_state=10,
            )
        if hardware == "vpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=570,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=3,
                oob_score=True,
                random_state=10,
            )

    if kernel == "split":
        if hardware == "cpu":
            model = RandomForestRegressor(
                max_depth=50,
                n_estimators=190,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=2,
                oob_score=True,
                random_state=10,
            )

    return model
