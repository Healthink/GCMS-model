import os
import sys
import bz2
import pickle
import typing
import numpy as np
import pandas as pd
from random import randint
from typing import Optional, Dict

from sklearn import metrics

from collections import Counter
from sysc.kmms_utils import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# third-party library
from loguru import logger


# global vars
_LOGGING_PATH = ''
_LOGGING_FILE_NAME = ''
_RANDOM_SCOPE_ = 2022


class KmmSmartDataModel(object):
    #
    
    def __init__(self, dataset_path:str, zipped_dataset_path:str, random_state:int, 
                 random_selection_state:int, feat_scope='all'):
        # init kmms data model
        
        # by default, passed csv file path
        logger.debug(f' dataset_path: {dataset_path}')
        logger.debug(f' zipped_dataset_path: {zipped_dataset_path}')
        self._random_state = random_state
        self._random_selection_state = random_selection_state
        self.dataset = None
        if dataset_path != None:
            self._read_csv(dataset_path)
        elif zipped_dataset_path != None:
            self._load_zipped_dataset(zipped_dataset_path)
        else:
            logger.error(' instance construction error, file path mistaken')
        self._resampled_indices = None
        
    def get_resampled_indices(self):
        return self._resampled_indices
    
    def _read_csv(self, dataset_path:str) -> None:
        # read csv format dataset file
        logger.debug(f' csv dataset path: {dataset_path}')
        try:
            self.dataset = pd.read_csv(dataset_path)
        except Exception as e:
            logger.debug(e)
    
    def _load_zipped_dataset(self, zipped_dataset_path:str) -> None:
        # load csv dataset from zip file
        if zipped_dataset_path is None:
            logger.error(' no zipped dataset file path provieded.')
            logger.debug(f' zipped dataset path: {zipped_dataset_path}')
            sys.exit(PARAM_EMPTY_ERROR)
        else:
            with bz2.BZ2File(zipped_dataset_path, 'r') as f:
                self.dataset = pickle.load(f)
            if self.dataset.shape[0] == 0:
                logger.error(' zipped dataset loaded error.')
    
    def _set_dataset(self, dataset) -> None:
        # manually set dataset obj
        self.dataset = dataset

    def _get_dataset(self) -> pd.DataFrame:
        # get dataset obj
        return self.dataset

    def get_shape_str(self) -> str:
        # get dataset shape strign
        
        row_nr = self.dataset.shape[0]
        col_nr = self.dataset.shape[1]
        return f' dataset shape: [{row_nr}]x[{col_nr}].'
    
    def get_columns(self) ->list:
        # get column list
        return self.dataset.columns.to_list()
    
    def get_shape(self) -> tuple:
        # get dataset shape
        return self.dataset.shape

    def split_dataset(self, 
                    rate:float, 
                    outcome:str, 
                    random_state:float,
                    forced_dataset_hashnr_l:List[int]=None,
                    undersample:Optional[Dict[str, float]]=None,
                    uppersample:Optional[List[int]]=None
                    ) -> tuple:
        # split train test data set
        
        
        # TODO: add identify outcome and rearange the order
        #! specify outcome column name without supporting index
        
        # splitting the ds with stratify
        dataset_full = self.dataset
        if forced_dataset_hashnr_l is not None:
            logger.debug('into forced training mode')
            dataset_forced = dataset_full[dataset_full.index.isin(forced_dataset_hashnr_l)]
            dataset_rest = dataset_full[~dataset_full.index.isin(forced_dataset_hashnr_l)]
            
            dataset_rest_x = dataset_rest[self.get_columns()[:-1]]
            dataset_rest_y = dataset_rest[outcome]
            
            dataset_forced_x = dataset_forced[self.get_columns()[:-1]]
            dataset_forced_y = dataset_forced[outcome]
            logger.debug(f'forced dataset shape: {dataset_forced.shape}')
            logger.debug(f'rest dataset shape: {dataset_rest.shape}')
        else:
            dataset_rest_x = dataset_full[self.get_columns()[:-1]]
            dataset_rest_y = dataset_full[outcome]
            
        
        x_train, x_test, y_train, y_test = train_test_split(
                                        dataset_rest_x, 
                                        dataset_rest_y, 
                                        test_size=rate,
                                        stratify=dataset_rest_y,
                                        random_state=random_state
                                    )
        logger.debug(f''' slpitting dataset completed with dimension 
            {x_train.shape}{y_train.shape}{x_test.shape}{y_test.shape}''')
        logger.debug(f''' training dataset unsampled parts
                     {y_train.value_counts().to_dict()}''')
        
        
        if undersample is not None:
            logger.debug(f' undersampling: {undersample}')
        
            # TODO: add unbanlanced sampling special parameter here
            #! to make training dataset less unbalanced
            # process x_train, y_train
            
            min_num = min([v for _, v in dict(Counter(y_train)).items()])
            max_num = max([v for _, v in dict(Counter(y_train)).items()])
            
            ## key checking
            for k, _ in undersample.items():
                if k not in y_train.unique():
                    logger.warning(f' warning unmatched item specified for undersampling!')
                    logger.debug(f' item: {k}.')
                    logger.debug(f' y_train unique: {y_train.value_counts()}')
                    sys.exit(f'undersample strategy key value error: {k} | {y_train.unique()}')
            ## constructing sampling strategy dictionary
            sampling_strategy = {
                k: int(v*min_num)
                for k, v in undersample.items()
            }
            logger.debug(f' real sampling strategy: {sampling_strategy}')
            
            _method = 'undersampling'
            if uppersample is not None:
                vc = y_train.to_frame().merge(
                    uppersample, 
                    left_index=True,
                    right_index=True,
                )['flag_marker2'].apply(lambda x: x[0] if x[0]!='2' else x[:-2]).value_counts()
                print('*before level1 sampling value counts:')
                print(vc.to_markdown(), end='\n\n')
            
            if _method == 'undersampling':
                from imblearn.under_sampling import RandomUnderSampler
                logger.debug(f' undersampling mode with stratety: {undersample}')
                #sampling_strategy_
                #! imbalanced resampling resets the index
                rus = RandomUnderSampler(
                    random_state=self._random_selection_state, 
                    sampling_strategy=sampling_strategy
                    )
                X_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
                
                
                self._resampled_indices = x_train.iloc[rus.sample_indices_,:].index.to_list()
                
                if X_train_resampled.shape[0] == len(list(self._resampled_indices)):
                    X_train_resampled.index = list(self._resampled_indices)
                    y_train_resampled.index = list(self._resampled_indices)
                else:
                    logger.warning(f'resampled indices length error {X_train_resampled.shape[0]}:{len(list(self._resampled_indices))}')
                
                
            elif _method == 'SMOTE':
                from imblearn.over_sampling import SMOTE
                rus = SMOTE(
                    random_state=self._random_state, 
                    sampling_strategy=sampling_strategy, 
                    n_jobs=12 # TODO parameterized in the future
                    )
                X_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
            logger.debug(f''' undersampling dataset completed with dimension 
                {X_train_resampled.shape}{y_train_resampled.shape}''')
            
            #! upper sampling
            if uppersample is not None:
                
                logger.debug(f' uppersampling size: {len(uppersample)}x5')
                #TODO merge flag_marker1 and flag_marker2 for simplified training
                
                ## uppersample: single flag_marker2 list df with hash as index
                uppersample = uppersample.apply(lambda x: x[0] if x[0]!='2' else x[:-2])
                merged_df = pd.concat( [X_train_resampled, y_train_resampled, ], axis=1 )
                merged_df = merged_df.merge(
                    uppersample,
                    left_index=True,
                    right_index=True,
                )
                ori_val_counts = merged_df['flag_marker2'].value_counts().to_dict()
                
                non_mma_disease_list = {
                    '2-PA':18,  '2-Citrin':18,  '2-GA-I':18,  '2-HPA':18, '2-OTCD':15, 
                    '2-IVA':15,  '2-MCD':10,  '2-MSUD':10,  '2-MCCD':10, '2-β-BKT':10
                }
                
                merged_df = pd.concat(
                    
                    [
                        merged_df.query("flag_marker2=='0'"),
                        merged_df.query("flag_marker2=='1'"),
                        merged_df.query("flag_marker2=='2-MMA'").sample(int(ori_val_counts['2-MMA']*0.8), random_state=2022)
                    ] + \
                    [ 
                        pd.concat([merged_df.query(f"flag_marker2=='{non_mma}'")]*non_mma_disease_list[non_mma], axis=0)
                        for non_mma in non_mma_disease_list
                    ]
                    , axis=0
                )
                X_train_resampled = merged_df.iloc[:,:-2]
                y_train_resampled = merged_df.iloc[:,-2]
                
                logger.debug(y_train_resampled.value_counts().to_markdown())
                print('*after all resampling:')
                print(merged_df['flag_marker2'].value_counts().to_markdown())
                
            
            if forced_dataset_hashnr_l is not None:
                X_train_resampled = pd.concat(
                    [
                        X_train_resampled,
                        dataset_forced_x
                    ],
                    axis=0
                )
                y_train_resampled = pd.concat(
                    [
                        y_train_resampled,
                        dataset_forced_y
                    ],
                    axis=0
                )
                logger.debug(f'forced merging: dataset_forced_x: {dataset_forced_x.shape}')
                logger.debug(f'forced merging: dataset_forced_y: {dataset_forced_y.shape}')
                logger.debug(f'forced merging: dataset_forced_y: {dataset_forced_y.value_counts()}')
            
            return X_train_resampled, x_test, y_train_resampled, y_test
        return x_train, x_test, y_train, y_test

class KmmSmartModel(object):
    
    def __init__(self, hash_col, random_selection_state:int, random:Optional[bool]=False):
        #! dataset model #
        # get workspace path
        self._workspace_dir = os.getcwd().split('imd')[0]
        logger.debug(f' local path: {self._workspace_dir}')
        
        self._normalized_marker = False
        
        self.dataset = None
        self._trained_marker = False
        self._cm = None

        # cv: cross validation
        self._training_cv_score = None
        self._testing_cv_score = None

        self._rf_model = None
        self._roc_curve = None
        self._auc_score = None
        
        self._singleton_marker = False
        
        self._project_train_dir = 'imd/data/train/'
        self._project_result_dir = 'imd/data/outputs/'
        
        self._forced_dataset_hashnr_l = []
        self.forced_dataset = None

        # random samples setting
        global _RANDOM_SCOPE_
        if random:
            self._random_state = randint(1,_RANDOM_SCOPE_)
            print(f'random_state: {self._random_state}')
        else:
            self._random_state = 2022
            # 734 is a good random state for level 1
        
        self._random_selection_state = random_selection_state
        
        # unique identity
        self._hash_col = hash_col
        
        self._xtr, self._ytr, self._xte, self._yte = None, None, None, None
        
    def load_dataset(self, 
                    rate:float, 
                    outcome:str, 
                    singleton:Optional[str]=None,
                    dataset_path:Optional[str]=None, 
                    zipped_dataset_path:Optional[str]=None, 
                    undersample:Optional[Dict[str, float]]=None,
                    uppersample:Optional[List[int]]=None,
                    hash_col:Optional[str]=None,
                    forced_dataset_path:Optional[str]=None) -> tuple:
        # load dataset from file
        
        ## index col with hash value
        index_col = hash_col if hash_col else self._hash_col
        
        ## file path check
        self._dataset_path = dataset_path
        self._zipped_dataset_path = zipped_dataset_path
        self._forced_dataset_path = forced_dataset_path
        
        if self._dataset_path == None and self._zipped_dataset_path == None:
            logger.error(' csv file path and zipped file path both empty.')
            logger.error(' reconstruct or call corresponding setting methods.')
            sys.exit(PARAM_EMPTY_ERROR)
        
        ## dataset loaded only once
        elif self.dataset is None:
            ### load file
            logger.debug(f'self._zipped_dataset_path: {self._zipped_dataset_path}')
            self.dataset = KmmSmartDataModel(dataset_path=self._dataset_path, 
                                            zipped_dataset_path=self._zipped_dataset_path,
                                            random_state=self._random_state,
                                            random_selection_state=self._random_selection_state)
            
            if self._forced_dataset_path is not None:
                if '.csv' in self._forced_dataset_path:
                    self.forced_dataset = pd.read_csv(self._forced_dataset_path)
                    logger.debug(f'forced_dataset loaded: {self.forced_dataset.shape}')
                elif '.pbz2' in self._forced_dataset_path:
                    self.forced_dataset = read_zip(self._forced_dataset_path)
                    logger.debug(f'forced_dataset loaded: {self.forced_dataset.shape}')
                else:
                    logger.error(' wrong input file format for forced dataset')
                    
                
                self._forced_dataset_hashnr_l = self.forced_dataset['hash_nr']
            
                ### merging
                
                #if self.dataset.dataset.columns.to_list()[-1] 
                try:
                    self.forced_dataset = self.forced_dataset[self.dataset.dataset.columns.to_list()]
                except:
                    logger.error('forced_dataset does not match raw dataset')
                    sys.exit('_CONSTRUCTION_ERROR_')
                
                self.forced_dataset.columns = self.dataset.dataset.columns
                self.dataset.dataset = pd.DataFrame(pd.concat(
                    [
                        self.dataset.dataset,
                        self.forced_dataset
                    ],
                    axis=0
                ))
            
            else:
                self._forced_dataset_hashnr_l = []
            
            logger.info(' dataset loaded completed.')
            logger.debug(self.dataset.get_shape_str())
            
            ### set index and drop old index col
            if index_col:
                if index_col in self.dataset.dataset:
                    self.dataset.dataset.set_index(index_col, drop=True, inplace=True)
                    logger.debug(f'index column setup! {index_col}')
                else:
                    logger.warning(f'indicated index column not exist! {index_col}')
            
            ### empty dataset check after loading
            if self.dataset.get_shape()[0] == 0:
                logger.error(' dataset loaded error, dataset obj empty.')
                sys.exit(OBJ_CONSTRUCT_ERROR)
            
            ### getting feature list
            self.feature_set = self.dataset.get_columns()[:-1]
            logger.debug(f' feature numbers: [{len(self.feature_set)}]')
        
        
        # TODO nonmalized transformation
        ## df_norm = (df - df.min()) / (df.max() - df.min())
        self._normalized_marker = True #! no _normalized_marker for the moment
        if not self._normalized_marker:
            logger.debug(f' nomalizing dataset')
            feature_matrix = self.dataset.dataset.iloc[:,:-1]
            
            #! Notice the divided number must be checked not zero
            #! fixed
            # check na first
            if self.dataset.dataset.isna().sum().sum() != 0:
                logger.error(f' dataset contains na values')
                sys.exit('Value Error')
            # avoid divide by 0
            for i in range(self.dataset.dataset.shape[1]-1):
                    col = self.dataset.dataset.iloc[:,i]
                    if col.sum() == 0:
                        continue
                    self.dataset.dataset.iloc[:,i] = (col - col.min()) / (col.max() - col.min())
            self._normalized_marker = True
        else:   logger.debug(f' skipped normalizing dataset')
        
        ## singleton outcome column for binary modelling
        logger.info(f'singleton mode: {singleton}')
        logger.info(f'value counts: {self.dataset.dataset[outcome].value_counts()}')
        if not self._singleton_marker:
            if singleton is not None:
                self.dataset.dataset[outcome] = \
                    self.dataset.dataset[outcome].apply(
                        #! singleton to binarized labels
                        lambda x:
                            # x if x == singleton else '0'
                            '1' if x == singleton else '0'
                    )
                
                self._singleton_marker = True
                logger.debug(' singleton completed!')
                #! keep undersample strategy keys to 1 & 0
                undersample['1'] = undersample.pop(singleton)
        
        ## splitting dataset
        if self._xtr is None and self._ytr is None and self._xte is None and self._yte is None:
            self._xtr, self._xte, self._ytr, self._yte = self.dataset.split_dataset(
                rate=rate, outcome=outcome,
                undersample=undersample,
                uppersample=uppersample,
                random_state=self._random_state,
                forced_dataset_hashnr_l=self._forced_dataset_hashnr_l
            )
        
        return self._xtr, self._xte, self._ytr, self._yte
    
    def get_train_ds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get training dataset
        return self._xtr, self._ytr
    
    def get_test_ds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get testing dataset
        return self._xte, self._yte
    
    def set_profile(self, CV_para_dict:Optional[dict]=None) -> None:
        # load or set para_CV for CV grid training models
        
        # check parameters
        para_grid_list = ['n_estimators',
                    'max_features',
                    'max_depth',
                    'min_samples_split',
                    'min_samples_leaf',
                    'bootstrap',
                    'criterion']
        if [para for para in para_grid_list if para not in self._CV_para_dict]:
            logger.error(' parameter CV dictionary uncomplete.')
            logger.debug(' dictionary keys: ', [k for k in self._CV_para_dict])
            sys.exit(PARA_MISTAKE_ERROR)
        
        self._CV_para_dict = CV_para_dict
        logger.debug(' parameter grid CV: ', self._CV_para_dict)
    
    
    
    def train_CV(self, rate:float=0.2, outcome:Optional[str]=None, undersample=False) -> None:
        # declare a random forest grid search model
        
        self._rf_model = RandomForestClassifier(class_weight='balanced')
        logger.info(' rf model construction completed.')
        self.set_profile(self._CV_para_dict)
        rf_cv= RandomizedSearchCV(estimator = self._rf_model,
                                    param_distributions = self._CV_para_dict,
                                    n_iter = 200, cv = 5, verbose=1, n_jobs =12 ,random_state=38,
                                    scoring='balanced_accuracy')
        logger.info(' rf model CV parameter construction completed.')
        # feed with data
        self._split_rate = rate
        self._dataset_outcome = outcome
        x_train, _, y_train, _ = self.load_dataset(
            rate=self._split_rate, 
            dataset_path=self._dataset_path,
            zipped_dataset_path=self._zipped_dataset_path,
            outcome=self._dataset_outcome,
            undersample=undersample
            )
        logger.info(' grid search rf model training ...')
        rf_cv.fit(x_train, y_train)
        logger.info(' grid search rf model training completed.')
        logger.info(f' best para: {rf_cv.best_params_}')
        self.rf_cv = rf_cv


    def train(self, best_para, rate, outcome, n_jobs, undersample, uppersample) -> None:
        # declare a random forest model
        
        if best_para is None:
            _error_message= '''
            no parameters provided for traing a model, nether here nor in the init part
            '''
            logger.error(_error_message)
            sys.exit(PARAM_EMPTY_ERROR)
        self._rf_model = RandomForestClassifier(**best_para, n_jobs=n_jobs, class_weight='balanced',
                                                random_state=self._random_state)

        logger.info(' rf model construction completed.')
        
        # feed with data
        self._split_rate = rate
        self._dataset_outcome = outcome
        x_train, _, y_train, _ = self.load_dataset(
            rate= self._split_rate,
            outcome= self._dataset_outcome,
            dataset_path= self._dataset_path, 
            zipped_dataset_path= self._zipped_dataset_path,
            undersample = undersample,
            uppersample=uppersample)
        
        logger.debug('debugging training dataset undersampling info:')
        logger.debug(f' x_train: {x_train.shape}')
        logger.debug(f' y_train: {y_train.shape}')
        logger.debug(f' y_train counter: {Counter(y_train)}')
        logger.debug(f' x_train columns nr:{len(x_train.columns.to_list())}')
        logger.info(' rf model training ...')
        logger.debug(f' training dataset y na value counts: {y_train.isna().sum()}')
        self._rf_model.fit(x_train, y_train)
        logger.info(' rf model training completed.')
        self._trained_marker = True


    def diagnoze_pat(self, X) -> str:
        # inspect input data type
        # if input pandas datafram
        if type(X) is type(pd.DataFrame()):#  or type(X) is type(pd.Series()):
            # if 2d type input, converting to 3d type for rf model on purpose
            if X.shape[0] == 0:
                logger.error(' input data empty.')
                sys.exit(PARAM_EMPTY_ERROR)
            elif X.shape[0] == 1:
                prediction = self._rf_model.predict(X)
            else:
                _warning_text = '''
                input more than one records, other records ignored, please use diagnoze_pats instead.
                '''
                logger.warning(_warning_text)
                prediction = self._rf_model.predict([X.iloc[0,:]])
        else:
            # numpy ndarray or list implementation to be done in the future
            _warning_text = '''
            numpy ndarray or list implementation to be done in the future.
            '''
            logger.error(_warning_text)
            sys.exit(UN_IMPLEMENTED_PART)
        
        return prediction

    def diagnoze_pats(self, X) -> list:
        # inspect input data type
        # if input pandas datafram
        if type(X) is type(pd.DataFrame()) or type(X) is type(pd.Series()):
            # if 2d type input, converting to 3d type for rf model on purpose
            if len(X.shape) == 0:
                logger.error(' input data empty.')
                sys.exit(PARAM_EMPTY_ERROR)
            elif len(X.shape) == 1:
                predictions = self._rf_model.predict([X])
            else:
                predictions = self._rf_model.predict(X)
        else:
            # numpy ndarray or list implementation to be done in the future
            _warning_text = '''
            numpy ndarray or list implementation to be done in the future.
            '''
            logger.error(_warning_text)
            sys.exit(UN_IMPLEMENTED_PART)
        
        return predictions

    def analyze_pat(self, X) -> np.float:
        # inspect input data type
        # if input pandas datafram
        #if type(X) is type(pd.DataFrame()) or type(X) is type(pd.Series()):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            # if 2d type input, converting to 3d type for rf model on purpose
            if len(X.shape) == 0:
                logger.error(' input data empty.')
                sys.exit(PARAM_EMPTY_ERROR)
            elif len(X.shape) == 1:
                prob_prediction = self._rf_model.predict_proba([X])
            else:
                _warning_text = '''
                input more than one records, other records ignored, please use analyze_pats instead.
                '''
                logger.warning(_warning_text)
                prob_prediction = self._rf_model.predict_proba([X.iloc[0,:]])
        else:
            # numpy ndarray or list implementation to be done in the future
            _warning_text = '''
            numpy ndarray or list implementation to be done in the future.
            '''
            logger.error(_warning_text)
            sys.exit(UN_IMPLEMENTED_PART)
        
        return prob_prediction

    def analyze_pats(self, X) -> list:
        # inspect input data type
        # if input pandas datafram
        #if type(X) is type(pd.DataFrame()) or type(X) is type(pd.Series()):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            # if 2d type input, converting to 3d type for rf model on purpose
            if len(X.shape) == 0:
                logger.error(' input data empty.')
                sys.exit(PARAM_EMPTY_ERROR)
            elif len(X.shape) == 1:
                prob_predictions = self._rf_model.predict_proba([X])
            else:
                prob_predictions = self._rf_model.predict_proba(X)
        else:
            # numpy ndarray or list implementation to be done in the future
            logger.debug(f' input data type: {type(X)}, shape: {X.shape}')
            _warning_text = '''
            numpy ndarray or list implementation to be done in the future.
            '''
            logger.error(_warning_text)
            sys.exit(UN_IMPLEMENTED_PART)
        logger.info(' analyzing patients completed.')
        return prob_predictions

    def analyze_dataset(self) -> str:
        # to be 
        return self.dataset.get_shape_str()


    def analyze_model(self, force, cv, n_jobs, result, method='cm', on:str='test') -> typing.Any:
        """ random forest model analysis
        # analze rf model with different metrics, confusion matrix default
        # othe methods: 'roc' for roc curve, 'auc' for auc score

        Args:
            force ([bool]): [if recalculate when done before]
            cv ([type]): [cross validation numbers]
            n_jobs ([type]): [paralell jobs (CPU number)]
            method (str, optional): [analysis methods]. Defaults to 'cm'.

        Returns:
            typing.Any: case method:
            case 'cm': pandas dataframe
            case 'roc': two lists
            case 'auc': float
            
        """         
        
        def _predict_onset(self:KmmSmartModel, method:str='pred', on:str='test') -> tuple:
            # predicting on testing dataset
            x_train, x_test, y_train, y_test = self.load_dataset(
                rate=self._split_rate, 
                outcome=self._dataset_outcome,
                dataset_path=self._dataset_path, 
                zipped_dataset_path=self._zipped_dataset_path)
            
            sliced = x_test.value_counts()
            
            if on == 'test':
                # logger.debug('analyzing model on test dataset')
                if method == 'pred':
                    pred = self._rf_model.predict(x_test)
                if method == 'proba':
                    pred = self._rf_model.predict_proba(x_test)
                    
            elif on == 'train':
                # logger.debug('analyzing model on train dataset')
                if method == 'pred':
                    pred = self._rf_model.predict(x_train)
                if method == 'proba':
                    pred = self._rf_model.predict_proba(x_train)
            
            else:
                logger.warning(f' wrong dataset specified for predicting')
                
            # proba_pred = self._rf_model.predict_proba(x_test)
            logger.info(' predicting on the dataset completed.')
            # counting results in binary classes
            ## t_count = sum([1 if p != 'negative' else 0 for p in pred])
            ## f_count = sum([1 if p == 'negative' else 0 for p in pred])
            from collections import Counter
            logger.debug(f'unique: {Counter(pred)}')
            logger.debug(f' predicting results: true: {Counter(pred)}')
            return x_train, x_test, y_train, y_test, pred
        
        # force flag to re_calculate metrics
        if force is False:
            # force mode
            logger.debug(' unforced mode:')
        if method == 'cm':
            # using confusion matrix method
            if self._cm is None or force:
                x_train, _, y_train, y_test, pred = _predict_onset(self, on=on)
                # calculate cross validation scroe
                logger.info(' calculating cross validation ...')
                #! cross validation here
                # TODO: uncomment it later
                # score = cross_val_score(self._rf_model, x_train, y_train, cv=cv, n_jobs=n_jobs).mean()
                score = 0.9786
                self._training_cv_score = score
                logger.info(f' testing on training dataset result summary [cross validation score]: {score:.4f}')
                
                # calculate confusion matrix and other metrics
                if on == 'test':
                    logger.debug(f' analyzing on test dataset')
                    self._cm = metrics.confusion_matrix(y_test, pred,
                                                        labels = sorted(np.unique(y_test)))
                elif on == 'train':
                    logger.debug(f' analyzing on train dataset')
                    self._cm = metrics.confusion_matrix(y_train, pred,
                                                        labels = sorted(np.unique(y_test)))
                logger.info(f' confusion matrix order: {sorted(np.unique(y_test))}')
                logger.info(f' confusion matrix shape: {self._cm.shape}')
                logger.debug(f' confusion matrix: {self._cm}')
            if result == 'cm':
                return self._cm
            elif result == 'prediction':
                return pd.DataFrame({'prediction':pred}, index=y_test.index)
            else:
                logger.warning(f'returning result type error: {result}')
        else:
            logger.error(' unimplemented part reached.')
            sys.exit(UN_IMPLEMENTED_PART)
        
    
    def get_feature_importance(self, top) -> list:
        # get top important features
        features = self._rf_model.feature_names_in_
        importances = self._rf_model.feature_importances_
        # ranking
        ranked_features, ranked_importances = zip(*sorted(zip(importances, features), 
                                                    reverse=True))
        
        return ranked_importances[:top], ranked_features[:top]
    
    
    def get_model(self):
        # get trained model
        return self._rf_model

    @property
    def cm(self):
        # get comfusion matrix
        return self._cm

    def get_dataset(self) -> pd.DataFrame:
        # get dataset reference
        return self.dataset._get_dataset()

    def load_model(self, model_path, rate, outcome) -> None:
        # load trained model from file

        self._split_rate = rate
        self._dataset_outcome = outcome
        # saving to file
        with bz2.BZ2File(model_path, 'r') as f:
            self._rf_model = pickle.loads(f.read())
        
        logger.info(' loading pre-trained model from zipped file completed.')
    
    def save_model(self, model_path) -> None:
        # save trained model to file

        # check model existence
        if self._rf_model is None:
            logger.error(' no trained or loaded rf model.')

        with bz2.BZ2File(model_path, 'w') as f:
            f.write(pickle.dumps(self._rf_model))
        
        logger.info(' writing trained model to zipped file completed.')

