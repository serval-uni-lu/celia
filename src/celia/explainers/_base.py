from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any
from celia.model import BaseModel
from celia.data import BaseData, PublicData
from celia.counterfactuals import Counterfactual
import pandas as pd
import numpy as np
from celia._errors import ConfigurationError, InstancesAreWithinRange


class BaseExplainer(ABC):
    """
    Abstract base class for explainers in Celia.
    All explainer classes should inherit from this class and implement the required methods.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args, **kwargs):
        """
        Initialize the BaseExplainer with a model instance.

        Parameters
        ----------
        model : BaseModel
            The model instance to be explained.
        data : BaseData
            The data instance associated with the model.

        """
        self._validate_init_params(data)
        self._model = model
        self._data = data
        self._explainer = self._create_explainer(self.model, self.data, *args, **kwargs)

    @property
    def model(self) -> BaseModel:
        """
        Return the model associated with this explainer.

        Returns
        -------
        BaseModel
            The model instance used by the explainer.
        """
        return self._model

    @property
    def data(self) -> BaseData:
        """
        Return the data associated with this explainer.

        Returns
        -------
        BaseData
            The data instance used by the explainer.
        """
        return self._data

    @property
    def explainer(self) -> Any:
        """
        Return the explainer instance created by the subclass.

        Returns
        -------
        Any
            The explainer instance initialized by the subclass.
        """
        return self._explainer

    @abstractmethod
    def _create_explainer(self, model: BaseModel, data:BaseData, *args, **kwargs) -> Any:
        """
        Create the explainer instance.

        This method should be implemented by subclasses to initialize the explainer
        with the necessary parameters and configurations.
        """
        raise NotImplementedError("Subclasses must implement _create_explainer method.")

    @abstractmethod
    def generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> List[Counterfactual] | Counterfactual:
        """
        Generate counterfactual explanations for the given data.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The data to generate explanations for.

        Returns
        -------
        List[Counterfactual] | Counterfactual
            A List containing Counterfactual instances corresponding to the input or a single Counterfactual object.
        """
        raise NotImplementedError("Subclasses must implement generate_counterfactuals method.")

    @staticmethod
    def _validate_init_params(data: BaseData) -> None:
        """
        Validate the initialization parameters for the explainer.

        Raises
        ------
        ConfigurationError
        """
        if not isinstance(data, BaseData):
            raise ConfigurationError(
                message="Explainer requires data to be an instance of BaseData.",
                param="data",
                hint="Please provide a BaseData object with appropriate metadata.",
                config={"data_type": type(data).__name__}
            )

    def validate_sample(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> None:
        """
        Validate the input sample for generating counterfactuals.

        This method checks that the sample is of the correct type and contains the required columns.
        It then calls the subclass-specific `_validate_sample` method to enforce any additional constraints.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The data to validate.

        Raises
        ------
        ConfigurationError
            If the sample does not match the expected format or features.
        TypeError
            If the sample is not a pandas DataFrame or Series.
        """
        self.validate_sample_dtypes(sample)
        if isinstance(sample, (pd.DataFrame, pd.Series)):
            self._validate_sample_columns_presence(sample)
        self._validate_sample(sample, *args, **kwargs)

    @abstractmethod
    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> None:
        """
        Subclass-specific validation of the input sample for counterfactual generation.

        This method is called after generic checks (type validation and column presence)
         have already been performed in `validate_sample`.

        Subclasses should implement this method to enforce any additional constraints
        specific to the counterfactual generation logic (e.g., task-specific assumptions,
        preprocessing requirements, or format constraints).

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The input data instance(s) to validate. Guaranteed to be of valid type and
            contain all required features.

        Raises
        ------
        ConfigurationError
            If the sample violates subclass-specific requirements.
        """
        pass

    @staticmethod
    def validate_sample_dtypes(sample: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Validate the type of the input sample.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The data to validate.

        Raises
        ------
        TypeError
            If the sample is not a pandas DataFrame or Series.
        """
        if not isinstance(sample, (pd.DataFrame, pd.Series)):
            raise ConfigurationError(
                message="Sample must be a pandas DataFrame or Series.",
                param="sample",
                hint="Please provide input data as a pandas DataFrame or Series.",
                config={"sample_type": type(sample).__name__}
            )

    def _validate_input_types(self, *args, **kwargs) -> None:
        """
        Subclass-specific validation of the input types.

        This method is called after generic type validation in `validate_input_types`.
        Subclasses should implement this method to enforce any additional type constraints
        specific to the counterfactual generation logic.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The input data instance(s) to validate. Guaranteed to be of valid type.

        Raises
        ------
        ConfigurationError
            If the sample violates subclass-specific type requirements.
        """
        pass

    def _validate_sample_columns_presence(self, sample: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Check if the sample contains the required columns.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The data to check.

        Raises
        ------
        ConfigurationError
            If the sample does not contain the columns defined in self.data.
        """

        if isinstance(self.data, PublicData):
            required_columns = self.data.column_names
        else:
            raise ConfigurationError(f"Data must be an instance of PublicData, got {type(self.data)}")

        if isinstance(sample, pd.Series):
            sample_columns = list(sample.index)
        else:
            sample_columns = list(sample.columns)

        #Lowercase everything for case-insensitive matching
        sample_lower = [c.lower() for c in sample_columns]
        required_lower = [c.lower() for c in required_columns]

        # Duplicate check of sample columns
        if len(sample_lower) != len(set(sample_lower)):
            dupes = {c for c in sample_lower if sample_lower.count(c) > 1}
            raise ConfigurationError(f"Duplicate column names in sample: {sorted(dupes)}")

        sample_columns_set = set(sample_lower)
        required_columns_set = set(required_lower)
        if sample_columns_set != required_columns_set:
            missing = required_columns_set - sample_columns_set
            extra = sample_columns_set - required_columns_set
            msg = []
            if missing:
                msg.append(f"Missing columns: {sorted(missing)}")
            if extra:
                msg.append(f"Unexpected columns: {sorted(extra)}")
            raise ConfigurationError("Sample column mismatch. " + " ".join(msg))

class RegressorExplainer(BaseExplainer):
    """
    Base class for explainers that work with regression models.
    Inherits from BaseExplainer and implements additional validation for regression-specific requirements.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args, **kwargs):
        super().__init__(model, data, *args, **kwargs)

    def generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                 target_range: Union[List[float], Tuple[float, float]] = None,
                                 *args, **kwargs) -> List[Counterfactual] | Counterfactual:
        """
        Generate counterfactual explanations for the given input sample.

        This is the public interface for counterfactual generation in regression explainers.
        It performs input validation (e.g., validate target ranges), and delegates the actual generation logic
        to the subclass-specific `_generate_counterfactuals` method.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            A single instance or multiple instances for which counterfactual explanations are to be generated.

        target_range : Union[List[float], Tuple[float, float]], optional
            A desired output range (min, max) that counterfactual predictions should aim to fall within.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing one or more counterfactual samples.
        """
        super().validate_sample(sample)
        self._validate_target_range(target_range)
        filtered_samples = self._filter_samples_within_target_range(sample, target_range)
        if filtered_samples.empty:
            raise InstancesAreWithinRange()
        return self._generate_counterfactuals(sample=filtered_samples,
                                              target_range=target_range,
                                              *args,
                                              **kwargs)

    @staticmethod
    def _validate_target_range(target_range: Union[List[float], Tuple[float, float]]) -> None:
        """
        Validate the target range for regression counterfactuals.

        Parameters
        ----------
        target_range : Union[List[float], Tuple[float, float]]
            The desired output range for counterfactuals.

        Raises
        ------
        ConfigurationError
            If the target range is invalid (e.g., min >= max).
        """
        if target_range is None:
            raise ConfigurationError("target_range must be provided for regression counterfactual generation.")
        if not isinstance(target_range, (list, tuple)) or len(target_range) != 2:
            raise ConfigurationError("target_range must be a list or tuple of two elements [min, max].")
        if target_range[0] >= target_range[1]:
            raise ConfigurationError(f"Invalid target range: {target_range}. Min must be less than max.")

    def _filter_samples_within_target_range(self, sample: Union[pd.DataFrame, pd.Series],
                                            target_range: Union[List[float], Tuple[float, float]]) -> pd.DataFrame:
        """
        Filter out samples whose predictions already fall within the desired target range.

        This method is used in regression-based counterfactual generation to exclude instances
        that already satisfy the desired output condition. A message is printed for excluded
        samples, including their indices.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            The input sample(s) for which counterfactuals are to be generated.
            Can be a single-row Series or a multi-row DataFrame.

        target_range : Union[List[float], Tuple[float, float]]
            A (min, max) tuple indicating the desired prediction output range.

        Returns
        -------
        pd.DataFrame
            A filtered DataFrame containing only those samples whose predictions fall
            outside the target range. The returned DataFrame may be empty if all samples
            are within the target range.
        """
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        preds = np.array(self.model.predict(sample))
        min_val, max_val = target_range

        out_of_range_mask = (preds < min_val) | (preds > max_val)
        filtered_sample = sample[out_of_range_mask]

        if filtered_sample.shape[0] < sample.shape[0]:
            excluded_indices = sample[~out_of_range_mask].index.tolist()
            print(
                f"[Warning] Excluded {len(excluded_indices)} sample(s) already within target range: {excluded_indices}")

        return filtered_sample

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series],
                         target_range: Union[List[float], Tuple[float, float]] = None, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  target_range: Union[List[float], Tuple[float, float]],
                                  *args, **kwargs) -> List[Counterfactual]:
        """
        Abstract method to be implemented by subclasses to generate counterfactuals.

        Assumes that `sample` and `target_range` have already been validated by the public method.
        Subclasses must implement this method to define their specific counterfactual generation logic.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            A validated single instance to generate counterfactuals for.

        target_range : Union[List[float], Tuple[float, float]]
            A validated (min, max) target range for the regression output.

        Returns
        -------
        List[Counterfactual] | Counterfactual
            A List containing Counterfactual instances corresponding to the input or a single Counterfactual object.
        """

class ClassifierExplainer(BaseExplainer):
    """
    Base class for explainers that work with classification models.
    Inherits from BaseExplainer and implements additional validation for classification-specific requirements.
    """

    def __init__(self, model: BaseModel, data: BaseData, *args, **kwargs):
        super().__init__(model, data, *args, **kwargs)

    def generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                 *args, **kwargs) -> List[Counterfactual] | Counterfactual:
        """
        Generate counterfactual explanations for the given input sample.

        This is the public interface for counterfactual generation in classification explainers.
        It performs input validation and delegates the actual generation logic
        to the subclass-specific `_generate_counterfactuals` method.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            A single instance or multiple instances for which counterfactual explanations are to be generated.
        *args : Any
            Additional positional arguments passed to the explainer's generation method.
        **kwargs : Any
            Additional keyword arguments passed to the explainer's generation method.

        Returns
        -------
        List[Counterfactual] | Counterfactual
            A List containing Counterfactual instances corresponding to the input or a single Counterfactual object.
        """

        super().validate_sample(sample)
        return self._generate_counterfactuals(sample, *args, **kwargs)

    def _validate_sample(self, sample: Union[pd.DataFrame, pd.Series], *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _generate_counterfactuals(self, sample: Union[pd.DataFrame, pd.Series],
                                  *args, **kwargs) -> List[Counterfactual] | Counterfactual:
        """
        Abstract method to be implemented by subclasses to generate counterfactuals.

        Assumes that `sample` has already been validated by the public method.
        Subclasses must implement this method to define their specific counterfactual generation logic.

        Parameters
        ----------
        sample : Union[pd.DataFrame, pd.Series]
            A validated single instance to generate counterfactuals for.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing counterfactual instances corresponding to the input.
        """
        pass