from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from transformers_inventory import (
    DropColumnTransformer,
    SmoothCurve,
    LgbmTargetsCreator,
    CreateStatsDataframe,
    AggregateModelScores,
    )
from lgbm_model import LgbmModel
import copy


class LgbmDataPrep:
    def __init__(self,target_column:str = 'anomaly',look_back:int=10,num_splits:int=5) -> None:
        self.target_column = target_column
        self.look_back = look_back 
        self.num_splits = num_splits

    def get_model_pipeline(self):
        
        single_features_pipeline =Pipeline([
                ("drop_changepoint", DropColumnTransformer(["changepoint"])),
                ("smooth", SmoothCurve()),
                ("scaler", StandardScaler()),
                ("df_creator", CreateStatsDataframe(look_back=self.look_back)),
            ])
        
        transformers_list = []
        for i in range(self.num_splits):
            curr_pipe = copy.deepcopy(single_features_pipeline)
            predict_step = ('classifier',LgbmModel(partition_num=i ,num_splits=self.num_splits))
            curr_pipe.steps.append(predict_step)
            transformers_list.append((f'part_{i}',curr_pipe))        
        
        union_pipeline =FeatureUnion(transformer_list=transformers_list)
        
        features_pipeline = Pipeline([
            ('transformations', union_pipeline),
            ('processing_results',AggregateModelScores())
        ])
        
        
        return features_pipeline

    def get_targets_pipeline(self):
        return Pipeline([("targets_creation", LgbmTargetsCreator(look_back=self.look_back))])

    def run(self):
        model_pipeline = self.get_model_pipeline()
        targets_pipeline = self.get_targets_pipeline()

        return model_pipeline, targets_pipeline
