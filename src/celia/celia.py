import pandas as pd
import numpy as np
from .explainers.certifai import CERTIFAI
from .explainers.dice import Data, Model, Dice
from .explainers.nnce import NearestNeighborCE


class CELIA:
    def __init__(self, model, data, target_name, task_type, continuous_features,
                 immutable_features, mutable_features, feasible_ranges, backend='sklearn', verbose=True):
        self.model = model
        self.data = data
        self.target_name = target_name
        self.task_type = task_type
        self.continuous_features = continuous_features
        self.immutable_features = immutable_features
        self.mutable_features = mutable_features
        self.feasible_ranges = feasible_ranges
        self.backend = backend
        self.verbose = verbose
        self.certifai, self.dice_random, self.dice_genetic, self.nnce = self.create_explainers()


    def create_explainers(self):
        certifai = self.__create_certifai()
        dice_random, dice_genetic = self.__create_dice()
        nnce = self.__create_nnce(self.task_type)
        return certifai, dice_random, dice_genetic, nnce

    def generate_counterfactuals(self, instance, target, task_type='regression'):
        """
        Generate counterfactuals using different methods.
        """
        # Generate counterfactuals using NNCE
        if self.verbose:
            print('Searching counterfactuals with NNCE...')
        nnce_ces = self.__generate_counterfactual_with_nnce(instance, target, task_type=task_type)

        #Generate counterfactuals using DiCE
        if self.verbose:
            print('Searching counterfactuals with DiCE...')
        dice_random_ces = self.__generate_counterfactual_with_dice_random(instance, target_lower, target_upper)
        dice_genetic_ces = self.__generate_counterfactual_with_dice_genetic(instance, target_lower, target_upper)

        #Generate counterfactuals using Certifai
        if self.verbose:
            print('Searching counterfactuals with Certifai...')
        certifai_ces = self.__generate_counterfactual_with_certifai(instance, target, task_type=task_type)


        # List of tuples containing method names and their results
        methods_ces = [
            ('NNCE', nnce_ces),
            ('DiCE_Random', dice_random_ces),
            ('DiCE_Genetic', dice_genetic_ces),
            ('Certifai', certifai_ces)
        ]

        dfs = []
        for method_name, ces in methods_ces:
            if ces is not None:
                if self.verbose:
                    print(f'Method {method_name} found counterfactuals.')
                ces = ces.copy()  # avoid modifying original
                ces['method'] = method_name
                dfs.append(ces)

        # Concatenate all DataFrames vertically
        results_df = pd.concat(dfs, axis=0, ignore_index=True) if dfs else None

        return results_df

    def __create_certifai(self):
        certifai = CERTIFAI(Pm=0.1, Pc=0.1)
        certifai.set_constraints(x=self.data.drop(columns=[self.target_name]), fixed=self.immutable_features)
        return certifai

    def __generate_counterfactual_with_certifai(self, instance, target, task_type='regression'):
        if task_type=='regression':
            self.certifai.fit(
                model=self.model,
                x=instance,
                generations=100,
                model_type=self.backend,
                classification=False,
                trained_with_columns=True,
                distance="L1",
                target_lower=np.atleast_1d(target[0]),
                target_upper=np.atleast_1d(target[1]),
                verbose=False)
        else:
            self.certifai.fit(
                model=self.model,
                x=instance,
                generations=100,
                model_type=self.backend,
                classification=True,
                trained_with_columns=True,
                distance="L1",
                verbose=False)
        counterfactuals_list = self.certifai.results[0][1]
        columns = list(instance.columns) + [self.target_name]
        counterfactuals_df = pd.DataFrame(counterfactuals_list, columns=columns)
        return counterfactuals_df

    def __create_dice(self):
        d = Data(dataframe=self.data, continuous_features=self.continuous_features, permitted_range=self.feasible_ranges,
                 outcome_name=self.target_name)
        m = Model(model=self.model, backend=self.backend, model_type="regressor")
        dice_random = Dice(d, m, method="random")
        dice_genetic = Dice(d, m, method="genetic")
        return dice_random, dice_genetic

    def __generate_counterfactual_with_dice_random(self, instance, target_lower, target_upper ):
        results = self.dice_random.generate_counterfactuals(
            instance,
            total_CFs=3,
            desired_range=[target_lower, target_upper],
            features_to_vary=self.mutable_features,
            permitted_range=self.feasible_ranges
        )
        counterfactual_list = results.cf_examples_list[0].final_cfs_df
        return counterfactual_list

    def __generate_counterfactual_with_dice_genetic(self, instance, target_lower, target_upper):
        results = self.dice_genetic.generate_counterfactuals(
            instance,
            total_CFs=3,
            desired_range=[target_lower, target_upper],
            features_to_vary=self.mutable_features,
            permitted_range=self.feasible_ranges,
            maxiterations=600,
            initialization="random",
            verbose=False,
        )
        counterfactual_list = results.cf_examples_list[0].final_cfs_df
        return counterfactual_list

    def __create_nnce(self, task_type='regression'):
        nnce = NearestNeighborCE(train_data=self.data.drop(columns=[self.target_name]),
                                 model=self.model,
                                 target_name=self.target_name,
                                 task_type=task_type,
                                 verbose=self.verbose)
        return nnce

    def __generate_counterfactual_with_nnce(self, instance, target, task_type='regression'):
        if task_type == 'regression':
            desired_output = [target[0], target[1]]  # lower, upper
        else:
            desired_output = target  # e.g., class index or label
        results = self.nnce.generate_counterfactuals(
            instance=instance,
            desired_output=desired_output,
            n_counterfactuals=3,
            mutable_features=self.mutable_features)
        return results







