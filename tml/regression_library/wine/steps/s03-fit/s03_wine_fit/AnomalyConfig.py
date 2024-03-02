import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from joblib.memory import Memory
from pycaret.anomaly.oop import AnomalyExperiment
from pycaret.loggers.base_logger import BaseLogger
from pycaret.utils.constants import DATAFRAME_LIKE, SEQUENCE_LIKE

_EXPERIMENT_CLASS = AnomalyExperiment
_CURRENT_EXPERIMENT: Optional[AnomalyExperiment] = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


@dataclass
class AnomalyConfig():
    data: Optional[DATAFRAME_LIKE] = None,
    data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
    index: Union[bool, int, str, SEQUENCE_LIKE] = True,
    ordinal_features: Optional[Dict[str, list]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    keep_features: Optional[List[str]] = None,
    preprocess: bool = True,
    create_date_columns: List[str] = ["day", "month", "year"],
    imputation_type: Optional[str] = "simple",
    numeric_imputation: str = "mean",
    categorical_imputation: str = "mode",
    text_features_method: str = "tf-idf",
    max_encoding_ohe: int = -1,
    encoding_method: Optional[Any] = None,
    rare_to_value: Optional[float] = None,
    rare_value: str = "rare",
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    low_variance_threshold: Optional[float] = None,
    group_features: Optional[list] = None,
    group_names: Optional[Union[str, list]] = None,
    drop_groups: bool = False,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    bin_numeric_features: Optional[List[str]] = None,
    remove_outliers: bool = False,
    outliers_method: str = "iforest",
    outliers_threshold: float = 0.05,
    transformation: bool = False,
    transformation_method: str = "yeo-johnson",
    normalize: bool = False,
    normalize_method: str = "zscore",
    pca: bool = False,
    pca_method: str = "linear",
    pca_components: Optional[Union[int, float, str]] = None,
    custom_pipeline: Optional[Any] = None,
    custom_pipeline_position: int = -1,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, str, logging.Logger] = True,
    log_experiment: Union[bool, str, BaseLogger, List[Union[str, BaseLogger]]] = False,
    experiment_name: Optional[str] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    verbose: bool = True,
    memory: Union[bool, str, Memory] = True,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
