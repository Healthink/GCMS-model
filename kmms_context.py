import os
import sys
import numpy as np
import pandas as pd


import sys 
#sys.path.append("..")
from sysc.kmms import PARAM_EMPTY_ERROR, UN_IMPLEMENTED_PART, KmmSmartModel

from loguru import logger
from typing import Optional, overload, Literal, List, Dict, Tuple
# from fastapi import FastAPI


class KmmSmartContext(object):
    
    def __init__(self, appid,
                rate:float=0.2, 
                level:str='INFO',
                random:bool=False,
                outcome:str='outcome',
                log_path:str='tmp_file.log',
                hash_col:Optional[str]=None,
                best_para:Optional[dict]=None,
                para_grid_CV:Optional[str]=None, 
                dataset_path:Optional[str]=None,
                undersample:Optional[Dict[str, float]]=None,
                singleton:Optional[str]=None,
                zipped_dataset_path:Optional[str]=None,
                forced_dataset_path:Optional[str]=None,
                uppersample:Optional[List[int]]=None,
                silence:bool=False,
                random_selection_state:int=None):
        
        # init kmms system
        """You may use this system to train your own model to predict MS 
        related diseases and corresponding risk probabilities.
        Other options:
        You may also train with pre-set parameters,
        train with CV grid searching, load pre-trained model, .


        Args:
            appid ([type]): [application ID]
            log_path (str, optional): [log file path]. Defaults to 'file.log'.
            best_para ([type], optional): [predefined parameters for random forest model].
                                            Defaults to None.
            para_grid_CV ([type], optional): [grid searching training parameters]. 
                                            Defaults to None.
            dataset_path ([type], optional): [dataset file path for training]. 
                                            Defaults to None.
            zipped_dataset_path ([type], optional): [zipped dataset file path for training]. 
                                            Defaults to None.
            rate (float, optional): [training testing dataset spliting rate]. 
                                            Defaults to 0.2.
            outcome (str, optional): [outcome field name in the dataset]. 
                                            Defaults to 'outcome'.
            level (str, optional): [log outputing level]. Defaults to 'INFO'.
        """
        # application id
        self._appid= appid
        
        self._rate = rate
        
        # get workspace path
        self._workspace_dir = os.getcwd().split('imd')[0]
        self._project_train_dir = 'imd/data/train/'
        self._project_result_dir = 'imd/data/outputs/'
        
        
        # set logging system
        log_path = self._workspace_dir + self._project_result_dir + log_path
        # set log style and level
        if silence:
            logger.remove()
        logger.add(log_path, format="{time} {level} {message}", filter="", level=level)
        logger.info(f' starting point: [appid: {self._appid}]')
        logger.info(f' local path: {self._workspace_dir}')        
        
        self.model = KmmSmartModel(random=random, hash_col=hash_col, 
                                    random_selection_state=random_selection_state)
        
        # set file path 
        self._dataset_path, self._zipped_dataset_path = dataset_path, zipped_dataset_path
        self._forced_dataset_path = forced_dataset_path
        
        if dataset_path is not None or zipped_dataset_path is not None:
            self.load_dataset(dataset_path=dataset_path, zipped_dataset_path=zipped_dataset_path,
                                rate=rate, outcome=outcome, undersample=undersample, singleton=singleton,
                                forced_dataset_path=forced_dataset_path, uppersample=uppersample)
        else: logger.info(' no dataset file path provided in construction!')
        
        self._best_para = best_para
        self._para_grid_CV = para_grid_CV
        
        self._undersmaple = undersample
        
    def load_model(self, 
                    trained_model_path:str='multi_model.pbz2', 
                    rate:float=0.2, 
                    outcome:str='outcome') -> None:
        # load pre_trained training model
        # padding project root path
        # TODO fix multiple places of settings rate and outcome parameters
        _model_path = self._workspace_dir + self._project_result_dir + trained_model_path
        self.model.load_model(_model_path, rate, outcome)
        logger.info(' trainning model loaded.')
    
    @property
    def data_dir(self) -> str:
        # return data file directory
        return self._workspace_dir + self._project_train_dir

    @property
    def result_dir(self) -> str:
        # return output file directory
        return self._workspace_dir + self._project_result_dir 
    
    def load_dataset(self, 
                    rate:float=0.2, 
                    outcome:str='outcome',
                    dataset_path:Optional[str]=None, 
                    singleton:Optional[str] = None,
                    zipped_dataset_path:Optional[str]=None,
                    undersample:Optional[Dict[str, float]]=None,
                    uppersample:Optional[List[int]]=None,
                    forced_dataset_path:Optional[str] = None) -> None:
        # load dataset with ether pre-provided path in init function or here.
        
        ## Notice: pading input file path only in context object not model object
        # check path availability
        # if file path getting from here, 
        
        if dataset_path is not None:
            logger.debug(f'*** {self._workspace_dir}, {self._project_train_dir}, {dataset_path}')
            self._dataset_path = self._workspace_dir + self._project_train_dir + \
                dataset_path
        # elif self._dataset_path is not None:
        #    self._dataset_path = self._workspace_dir + self._project_train_dir + self._dataset_path
        
        if zipped_dataset_path is not None:
            self._zipped_dataset_path = self._workspace_dir + self._project_train_dir + \
                zipped_dataset_path
        #elif self._zipped_dataset_path is not None:
        #    self._zipped_dataset_path = self._workspace_dir + self._project_train_dir + self._zipped_dataset_path
        
        if forced_dataset_path is not None:
            self._forced_dataset_path = self._workspace_dir + self._project_train_dir + \
                forced_dataset_path
        
        logger.debug(' context_load_dataset: ')
        logger.debug(f' self._dataset_path: {self._dataset_path}')
        logger.debug(f' self._zipped_dataset_path: {self._zipped_dataset_path}')
        logger.debug(f' self._forced_dataset_path: {self._forced_dataset_path}')
        
        # loading
        self.model.load_dataset(dataset_path=self._dataset_path, singleton=singleton,
                                zipped_dataset_path=self._zipped_dataset_path, 
                                rate=rate, outcome=outcome, undersample=undersample,
                                uppersample=uppersample,
                                forced_dataset_path=self._forced_dataset_path)
    
    def get_train_ds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get training dataset
        return self.model.get_train_ds()
    
    def get_test_ds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get testing dataset
        return self.model.get_test_ds()
    
    def get_resampled_indices(self):
        return self.model.dataset._resampled_indices
    
    def save_model(self, 
                    trained_model_path:Optional[str]=None) -> None:
        # save trainded model to file
        
        # serized model to file
        self.model.save_model(self._workspace_dir + self._project_result_dir 
                                + trained_model_path)
        logger.info(' trainning model saved.')

    def train_CV(self, 
                para_CV:Optional[dict]=None, 
                rate:float=0.2, 
                outcome:Optional[str]='outcome',
                undersample=False) -> None:
        # grid search model train
        self.model._CV_para_dict = para_CV
        logger.info(' ***')
        self.model.train_CV(rate=rate, outcome=outcome, undersample=undersample)
    
    '''
    def set_dataset_path(self, dataset_path) -> None:
        # set dataset path manually
        self.model.set_dataset_path(dataset_path)
        
        
    def set_zipped_dataset_path(self, zipped_dataset_path) -> None:
        # set zipped dataset path manually
        logger.debug(f'zipped_dataset_path: {zipped_dataset_path}')
        _path_padding = self._workspace_dir + self._project_train_dir
        self.model.set_zipped_dataset_path(_path_padding+zipped_dataset_path)
    '''
    
    def train(self,
              best_para:Optional[dict]=None, 
              rate:Optional[float]=0.2,
              outcome:Optional[str]='outcome',
              undersample:Optional[Dict[str, float]]=False,
              uppersample:Optional[List[int]]=None,
              n_jobs:int=1) -> None:
        # train rf model with certain parameters
        if not self._dataset_path and not self._zipped_dataset_path:
            logger.error(' no dataset path provied.')
            sys.exit(PARAM_EMPTY_ERROR)
        
        # training
        self.model.train(best_para, rate, outcome, n_jobs=n_jobs, undersample=undersample, uppersample=uppersample)

    def analyze_dataset(self) -> str:
        # analyze the dataset after training, currently provide only basic 
        # information like dimension, etc. others will be added in the future.
        return self.model.analyze_dataset()
    
    @overload
    def analyze_model(self, result:Literal['cm'], force=False, cv=10, n_jobs=1,
                        method='cm', on:str='test') -> List[List[float]]: ...
    
    @overload
    def analyze_model(self, result:Literal['prediction'], force=False, cv=10, n_jobs=1,
                        method='cm', on:str='test') -> pd.DataFrame: ...
    
    def analyze_model(self, force=False, cv=10, n_jobs=1, method='cm',
                        result:str='cm', on:str='test') -> pd.DataFrame:
        # analyze the trained model, currently provide only cross validation
        # on training dataset, because the testing has no enough records for 
        # for all disease classes. other methods will be added in the future.
        return self.model.analyze_model(force, cv=cv, n_jobs=n_jobs, method=method, 
                                        result=result, on=on)

    def diagnoze_pat(self, X) -> str:
        # provide prediction on single patient, result will be single class.
        return self.model.diagnoze_pat(X)

    def diagnoze_pats(self, X) -> list:
        # provide predictions on a group of patients, currently only support 
        # dataframe type. other ndarray like types will be added in the future.
        return self.model.diagnoze_pats(X)

    def analyze_pat(self, X) -> np.float:
        # provide prediction *grading* on a group patients, currently only 
        # support dataframe type. other ndarray like types will be added in 
        # the future. currently only support mean probabilities, AUC will 
        # be supported in the future.
        return self.model.analyze_pat(X)

    def save_model_analysis(self, analysis_path) -> None:
        if (self.model.cm==None):
            logger.warning(f' internal random forest model has not been analyzed! \
                            please call analyze_model method first.')
        else: return self.model.cm

    def analyze_pats(self, X) -> list:
        # provide prediction *grading* on a single patient, currently only 
        # support dataframe type. other ndarray like types will be added in 
        # the future. currently only support mean probabilities, AUC will 
        # be supported in the future.
        return self.model.analyze_pats(X)

    def get_model(self):
        # get build-in trained model
        return self.model.get_model()

    def get_dataset(self):
        # get build-in dataset
        return self.model.get_dataset()

    def export_report(self, format='csv'):
        # generate report on a group of patients as input, currently is not 
        # implemented 
        if format == 'csv':
            pass
            # return self.model.get_report(format)
        else:
            logger.error(' unimplemented part reached. no other exporting format supported.')
            sys.exit(UN_IMPLEMENTED_PART)
            
    def get_importance(self, top=10) -> list:
        # get feature importance from trained model
        
        return self.model.get_feature_importance(top=top)
    
    def get_columns(self) -> List[str]:
        # get all columns for prediction
        return self.get_dataset().columns.to_list()[:-1]


    # prediction by rules
    def diagnoze_by_rules() -> List[str]:
        # diagnoze patients by rules
        
        ## define rules
        ### headers: [rule_level, rule_type, rule_threshold_L, rule_threshold_H, rule_threshold_quan]

        return None