# kmms utilities
import typing
import sys

import bz2
import pickle
# from asyncssh import ConfigParseError
# from biplist import Data
import pandas as pd
from loguru import logger
from pandas.core.frame import DataFrame
from typing import Optional, Literal, Union, List, Tuple, overload


# error codes definition
OBJ_CONSTRUCT_ERROR = 2
PARA_MISTAKE_ERROR = 3
UN_IMPLEMENTED_PART = 4
PARAM_EMPTY_ERROR = 5


def get_report(context, ids:list, diseases:list, dataset:pd.DataFrame, 
               report_path:str) -> pd.DataFrame:
    """[generating a probability prediction report for grading system]

    Args:
        context ([kmms context]): [kmms context instance]
        ids (list[int]): [id column of the original table]
        diseases (list[str]): [diagnosis column of the original table]
        dataset (pd.DataFrame): [prediction matrix]
        report_path (str): [output report file path]
        
    Returns:
        pd.DataFrame: [report]
    """
    pred_list = context.diagnoze_pats(dataset)
    proba_matrix = context.analyze_pats(dataset).round(3)
    matrix_pd = pd.DataFrame(proba_matrix, columns=context.get_model().classes_)
    report_pd = pd.DataFrame({'id':ids, 'diagnosis':diseases, 'prediction':pred_list})
    report = pd.concat([report_pd, matrix_pd], axis=1)
    report.to_csv(report_path, index=False)
    return report



def rank_preds(report:pd.DataFrame) -> pd.DataFrame:
    """[sumfilter predictions with top probabilitiesary]

    Args:
        report (pd.DataFrame): [report ]

    Returns:
        pd.DataFrame: [re-filtered report ]
    """
    # diagnosis fields starting position
    _DIAG_START_ = 3
    # diagnosis list
    _diag_list = report.columns.to_list()[_DIAG_START_:]
    # report information part
    _report_info_part = report.iloc[:,:_DIAG_START_]
    
    # get top n diagnoses with formating
    def _append_list(list1, list2) -> str:
        # appending results to format
        text = ''
        if len(list1) != len(list2):
            sys.exit(PARA_MISTAKE_ERROR)
        return [str(list1[i]) + ':' + str(list2[i]) for i in range(len(list1))]
        
    # 
    def _rank_diag(diag_prob_list, full_diag_list, top) -> pd.Series:
        # sorting with index
        proba_top, diag_rank = zip(*sorted(
            zip(diag_prob_list, range(len(diag_prob_list))), reverse=True))
        # using index to get diagnosis name
        # logger.debug(f' index: {diag_rank}')
        # logger.debug(f' full_diag_list length: {len(full_diag_list)}')
        diag_rank_list = [full_diag_list[index] for index in diag_rank]
        return_l =  _append_list(diag_rank_list[:top], proba_top[:top])
        return pd.Series(return_l, index=['top'+str(i) for i in range(3)])
    
    # report top diagnosis part: diagnoses with probabilities 
    _report_tops = report.iloc[:,_DIAG_START_:].apply(
        lambda x: _rank_diag(x, _diag_list, 3), axis=1)
    
    return pd.concat([_report_info_part, _report_tops], axis=1)
    


@overload
def compute_cm(cm, col:Literal[None]) -> list: ...
    
@overload
def compute_cm(cm, col:List[str]) -> DataFrame: ...

def compute_cm(cm, col:Optional[list]=None) -> Union[list, DataFrame]:
    # compute accuracy matrix
    size_i = len(cm)
    size_c = len(cm[0])
    
    accr_list = [cm[i][i]/sum(cm[i]) for i in range(len(cm))]
    
    if col:
        return pd.DataFrame({'type':col, 'acc':accr_list})
    else: return accr_list
    


def pooling(data:pd.DataFrame, size:int) -> list:
    """pooling function for a pandas:DataFrame type matrix
    Args:
        data (pd.DataFrame): data matrix(should be)
        size (int): _description_

    Returns:
        list: _description_
    """
    h,w = data.shape
    img_new = []
    m = size
    n = size
    for i in range(0, h, m):
        line = []
        for j in range(0, w, n):
            x = data.iloc[i:i+m,j:j+n] #选取池化区域
            line.append(x.sum().sum())
        img_new.append(line)
    return img_new



def to_zip(dataset:pd.DataFrame, zip_file_path:str, format:str='pbz2') -> None:
    # update: parquet format support
    if format == 'pbz2':
        with bz2.BZ2File(zip_file_path, 'w') as f:
            pickle.dump(dataset, f)
    elif format == 'parquet':
        dataset.to_parquet(zip_file_path, compression='gzip')
    else:
        logger.error(f'format set error: {format}')



def read_zip(zip_file_path:str, format:str='pbz2') ->pd.DataFrame:
    # update: parquet format support
    if format == 'pbz2':
        with bz2.BZ2File(zip_file_path, 'r') as f:
            return pickle.load(f)
    elif format == 'parquet':
        return pd.read_parquet(zip_file_path)
    else:
        logger.error(f'format set error: {format}')



def apply_rules(rule_file_path:str, records:DataFrame, target:str='2-MMA') -> DataFrame:
    # return type: DataFrame: ['matched','rule_id']
    ## headers: id, level, type, {feature: threshold_l|threshold_h}, rule_count
    rule_table = pd.read_csv(rule_file_path)
    rule_table = rule_table.loc[rule_table['rule_level']==target]
    ## check only one crutial matched rule
    rule_table.sort_values(['rule_type'], inplace=True) 
        
    # filtered_rule_table = rule_table.loc[rule_table['level']]
    checks, indexs = [], []
    for j in range(records.shape[0]):
        ## loop for all rules
        for i in range(rule_table.shape[0]):            
            ins_counter = 0
            rule = rule_table.iloc[i,:].to_dict()
            rule_id = rule['rule_id']
            rule_type = rule['rule_type']
            rule_contents = rule['rule_content'][1:-1].split(';')
            rule_count = int(rule['rule_count'])
            ### loop for all components
            for content in rule_contents:
                if float(records.iloc[j,:].loc[content.split(':')[0]]) >= \
                    float(content.split(':')[1]):
                    ins_counter += 1
            ### check threshold matched quantity
            if ins_counter >= rule_count:
                checks.append([rule_type, rule_id])
                indexs.append(records.index[j])
                break
    return pd.DataFrame(checks, index=indexs, columns=['rule_type', 'rule_id'], dtype=str)

