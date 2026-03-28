import warnings

import pytest

from celia.counterfactuals import Counterfactual
from celia.data import PublicData
from celia.errors import ConfigurationError, NoCounterfactualsFoundError
from celia.explainers import NNCEClassifierExplainer, NNCERegressorExplainer
from celia.model import SklearnModel


class TestNNCERegressorExplainer:

    def test_nnce_creation_with_valid_init(
            self,
            model_trained_without_encoded_data,
            dummy_regression_dataframe
    ):
        data = dummy_regression_dataframe
        X = data.drop(columns=['target'])
        y = data['target']
        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.select_dtypes(include=['float', 'int']).columns.tolist(),
            categorical_column_names=X.select_dtypes(include=['object', 'bool', 'str']).columns.tolist(),
            immutable_column_names=['feature1'],
            feasible_values={}
        )

        explainer = NNCERegressorExplainer(
            model=model_trained_without_encoded_data,
            data=public_data,
        )

        assert isinstance(explainer, NNCERegressorExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)
        assert explainer.data.data.equals(X)

    def test_nnce_creation_without_public_data(self, model_trained_with_encoded_data):
        """Test that NNCERegressorExplainer raises ConfigurationError when data is not PublicData"""
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            NNCERegressorExplainer(model=model_trained_with_encoded_data,
                                       data=invalid_data)
            err = exc_info.value
            assert "NNCERegressorExplainer requires data to be an instance of PublicData." in err.message
            assert err.param == "data"

    def test_nnce_run_without_target_range(self,  model_trained_without_encoded_data,
                                                  celia_public_data_without_encoded_data,
                                               dummy_test_regression_dataframe_encoded):
        """Test that NNCERegressorExplainer raises ConfigurationError when target_range is not provided"""

        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                               data=celia_public_data_without_encoded_data)

        sample = dummy_test_regression_dataframe_encoded
        sample = sample.drop(columns=['target'])
        with pytest.raises(ConfigurationError) as exc_info:
            explainer.generate_counterfactuals(sample)
            msg = str(exc_info.value)
            assert "target_range must be provided for regression counterfactual generation." in msg

    def test_nnce_run_with_multiple_instances(self, dummy_regression_dataframe_encoded):
        """Test that NNCERegressorExplainer accepts multiple instances and returns a list of Counterfactual."""
        from sklearn.tree import DecisionTreeRegressor

        X = dummy_regression_dataframe_encoded.drop(columns=["target"])
        y = dummy_regression_dataframe_encoded["target"]

        tree = DecisionTreeRegressor(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCERegressorExplainer(model=model, data=public_data)

        sample = X.iloc[:3]
        current_preds = tree.predict(sample)
        # Target range that excludes at least some current predictions
        target_range = [current_preds.max() + 0.01, current_preds.max() + 1.0]

        results = explainer.generate_counterfactuals(sample, target_range=target_range)

        if isinstance(results, Counterfactual):
            results = [results]

        assert len(results) >= 1
        for cf in results:
            assert isinstance(cf, Counterfactual)

    def test_nnce_no_counterfactuals_found(self, model_trained_without_encoded_data,
                                              celia_public_data_without_encoded_data,
                                              dummy_regression_dataframe):
        """Test that NNCERegressorExplainer raises NoCounterfactualsFoundError when no CFs found."""
        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                           data=celia_public_data_without_encoded_data,
                                           )

        sample = dummy_regression_dataframe.iloc[[0]]
        sample = sample.drop(columns=['target'])
        target_range = [10.0, 20.0]

        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample, target_range=target_range)

    def test_nnce_no_counterfactuals_found_emits_warning(self, model_trained_without_encoded_data,
                                                          celia_public_data_without_encoded_data,
                                                          dummy_regression_dataframe):
        """Test that a warning is emitted when no counterfactual is found for an instance."""
        explainer = NNCERegressorExplainer(model=model_trained_without_encoded_data,
                                           data=celia_public_data_without_encoded_data,
                                           )

        sample = dummy_regression_dataframe.iloc[[0]]
        sample = sample.drop(columns=['target'])
        target_range = [10.0, 20.0]

        with pytest.raises(NoCounterfactualsFoundError), pytest.warns(UserWarning):
            explainer.generate_counterfactuals(sample, target_range=target_range)

    def test_nnce_regressor_counterfactuals_exclude_target_column(
        self,
        dummy_regression_dataframe_encoded,
    ):
        """Counterfactual.original_instance and .counterfactuals must not contain the target column."""
        from sklearn.tree import DecisionTreeRegressor

        X = dummy_regression_dataframe_encoded.drop(columns=["target"])
        y = dummy_regression_dataframe_encoded["target"]

        tree = DecisionTreeRegressor(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCERegressorExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        current_pred = tree.predict(sample)[0]
        # Target range that excludes the current prediction
        target_range = [current_pred + 0.05, current_pred + 0.5]

        results = explainer.generate_counterfactuals(sample, target_range=target_range)
        target_name = public_data.target_name
        feature_columns = set(public_data.column_names)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            assert target_name not in cf.original_instance.columns, (
                f"original_instance should not contain target column '{target_name}', "
                f"got columns: {list(cf.original_instance.columns)}"
            )
            assert target_name not in cf.counterfactuals.columns, (
                f"counterfactuals should not contain target column '{target_name}', "
                f"got columns: {list(cf.counterfactuals.columns)}"
            )
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns

    def test_nnce_regressor_single_instance_returns_single_counterfactual(
        self,
        dummy_regression_dataframe_encoded,
    ):
        """A single instance should return a single Counterfactual, not a list."""
        from sklearn.tree import DecisionTreeRegressor

        X = dummy_regression_dataframe_encoded.drop(columns=["target"])
        y = dummy_regression_dataframe_encoded["target"]

        tree = DecisionTreeRegressor(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCERegressorExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        current_pred = tree.predict(sample)[0]
        target_range = [current_pred + 0.05, current_pred + 0.5]

        result = explainer.generate_counterfactuals(sample, target_range=target_range)

        assert isinstance(result, Counterfactual)


class TestNNCEClassifierExplainer:

    def test_nnce_classifier_creation_with_valid_init(
            self,
            model_trained_classifier_stratified,
            dummy_classification_dataframe,
    ):
        data = dummy_classification_dataframe
        X = data.drop(columns=['target'])
        y = data['target']
        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.select_dtypes(include=['float', 'int']).columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=['feature1'],
            feasible_values={},
        )

        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier_stratified,
            data=public_data,
        )

        assert isinstance(explainer, NNCEClassifierExplainer)
        assert isinstance(explainer.data, PublicData)
        assert isinstance(explainer.model, SklearnModel)
        assert explainer.data.data.equals(X)

    def test_nnce_classifier_creation_without_public_data(self, model_trained_classifier):
        """Test that NNCEClassifierExplainer raises ConfigurationError when data is not PublicData."""
        class InvalidData:
            pass

        invalid_data = InvalidData()
        with pytest.raises(ConfigurationError) as exc_info:
            NNCEClassifierExplainer(model=model_trained_classifier, data=invalid_data)

        err = exc_info.value
        assert "`data` must be an instance of PublicData." in err.message
        assert err.param == "data"

    def test_nnce_classifier_run_with_multiple_instances(
            self,
            dummy_classification_dataframe,
    ):
        """Test that NNCEClassifierExplainer accepts multiple instances and returns a list."""
        from sklearn.tree import DecisionTreeClassifier

        X = dummy_classification_dataframe.drop(columns=['target'])
        y = dummy_classification_dataframe['target']

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        public_data = PublicData(
            data=X,
            targets=y,
            target_name='target',
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCEClassifierExplainer(model=model, data=public_data)

        sample = X.iloc[:3]
        results = explainer.generate_counterfactuals(sample)

        if isinstance(results, Counterfactual):
            results = [results]

        assert len(results) >= 1
        for cf in results:
            assert isinstance(cf, Counterfactual)

    def test_nnce_classifier_no_counterfactuals_found(
            self,
            model_trained_classifier,
            celia_public_data_classification,
            dummy_classification_dataframe,
    ):
        """Test that NNCEClassifierExplainer raises NoCounterfactualsFoundError when no CFs are found.

        DummyClassifier(most_frequent) always predicts the same class, so no neighbors
        will have a different class prediction.
        """
        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier,
            data=celia_public_data_classification,
        )

        sample = dummy_classification_dataframe.drop(columns=['target']).iloc[[0]]
        with pytest.raises(NoCounterfactualsFoundError):
            explainer.generate_counterfactuals(sample)

    def test_nnce_classifier_no_counterfactuals_found_emits_warning(
            self,
            model_trained_classifier,
            celia_public_data_classification,
            dummy_classification_dataframe,
    ):
        """Test that a warning is emitted when no counterfactual is found for an instance."""
        explainer = NNCEClassifierExplainer(
            model=model_trained_classifier,
            data=celia_public_data_classification,
        )

        sample = dummy_classification_dataframe.drop(columns=['target']).iloc[[0]]
        with pytest.raises(NoCounterfactualsFoundError), pytest.warns(UserWarning):
            explainer.generate_counterfactuals(sample)

    def test_nnce_classifier_counterfactuals_exclude_target_column(
        self,
        dummy_classification_dataframe,
    ):
        """Counterfactual.original_instance and .counterfactuals must not contain the target column."""
        from sklearn.tree import DecisionTreeClassifier

        X = dummy_classification_dataframe.drop(columns=["target"])
        y = dummy_classification_dataframe["target"]

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        # No immutable features so NNCE can find neighbours with a different class
        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCEClassifierExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        target_name = public_data.target_name
        feature_columns = set(public_data.column_names)

        results = explainer.generate_counterfactuals(sample)

        if isinstance(results, Counterfactual):
            results = [results]

        for cf in results:
            assert target_name not in cf.original_instance.columns, (
                f"original_instance should not contain target column '{target_name}', "
                f"got columns: {list(cf.original_instance.columns)}"
            )
            assert target_name not in cf.counterfactuals.columns, (
                f"counterfactuals should not contain target column '{target_name}', "
                f"got columns: {list(cf.counterfactuals.columns)}"
            )
            assert set(cf.original_instance.columns) == feature_columns
            assert set(cf.counterfactuals.columns) == feature_columns

    def test_nnce_classifier_single_instance_returns_single_counterfactual(
        self,
        dummy_classification_dataframe,
    ):
        """A single instance should return a single Counterfactual, not a list."""
        from sklearn.tree import DecisionTreeClassifier

        X = dummy_classification_dataframe.drop(columns=["target"])
        y = dummy_classification_dataframe["target"]

        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        model = SklearnModel(tree)

        public_data = PublicData(
            data=X,
            targets=y,
            target_name="target",
            column_names=X.columns.tolist(),
            continuous_column_names=X.columns.tolist(),
            categorical_column_names=[],
            immutable_column_names=[],
            feasible_values={},
        )

        explainer = NNCEClassifierExplainer(model=model, data=public_data)

        sample = X.iloc[[0]]
        result = explainer.generate_counterfactuals(sample)

        assert isinstance(result, Counterfactual)
