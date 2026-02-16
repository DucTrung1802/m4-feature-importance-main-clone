import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import List


# =========================
# ENUM DEFINITIONS
# =========================


class ModelEnum(str, Enum):
    KERAS_NET = "M4KerasNetRegressionModel"
    KNN = "M4KNeighborsRegressor"
    KNN_COSINE = "M4KNeighborsCosineRegressor"
    DUMMY_MEAN = "M4DummyMeanRegression"
    XGB = "M4XGBRegressor"
    RF = "M4RandomForestRegressor"
    RF_SINGLE = "M4RandomForestSingleOutputRegressor"


class FeatureGroupEnum(str, Enum):
    TSFRESH = "TSFresh"
    CATCH22 = "Catch22"
    TSFRESH_ALL = "TSFresh+TSFreshDiff+TSFreshLog"
    CATCH22_ALL = "Catch22+Catch22Diff+Catch22Log"
    TSFRESH_CATCH22 = "TSFresh+Catch22"


# =========================
# CONFIG STRUCTURE
# =========================


@dataclass
class ExperimentConfig:
    run: int
    model: ModelEnum
    feature_group: FeatureGroupEnum
    correlation: float
    save_model: int = 0


# =========================
# EDIT YOUR CONFIG HERE
# =========================

CONFIG = ExperimentConfig(
    run=1,
    model=ModelEnum.KERAS_NET,
    feature_group=FeatureGroupEnum.TSFRESH,
    correlation=0.5,
    save_model=1,
)


# =========================
# FEATURE RESOLVER
# =========================


def resolve_features(group: FeatureGroupEnum) -> List[str]:
    mapping = {
        FeatureGroupEnum.TSFRESH: ["TSFresh"],
        FeatureGroupEnum.CATCH22: ["Catch22"],
        FeatureGroupEnum.TSFRESH_ALL: ["TSFresh", "TSFreshDiff", "TSFreshLog"],
        FeatureGroupEnum.CATCH22_ALL: ["Catch22", "Catch22Diff", "Catch22Log"],
        FeatureGroupEnum.TSFRESH_CATCH22: ["TSFresh", "Catch22"],
    }
    return mapping[group]


# =========================
# RUN FUNCTION
# =========================


def run_experiment(config: ExperimentConfig):
    features = resolve_features(config.feature_group)
    feature_name = ",".join(features)

    output = (
        f"trained_model/feature_compare/"
        f"m_{config.model.value}"
        f"_f_{feature_name}"
        f"_r_{config.run}"
        f"_c_{config.correlation}"
    )

    venv_python = r"mt_env\Scripts\python.exe"

    command = [
        venv_python,
        "train_model.py",
        "--model",
        config.model.value,
        "--run",
        str(config.run),
        "--predictions",
        "prediction_errors/mape.csv",
        "--feature_correlation_threshold",
        str(config.correlation),
        "--features",
    ]

    command.extend([f"features/{f}.csv" for f in features])

    if config.save_model == 0:
        command.extend(["--save_model", "0"])

    command.extend(["--output_save", output])

    print("Running command:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print("Done")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_experiment(CONFIG)
