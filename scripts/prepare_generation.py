#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare generation data
@author: jliu
"""

import pandas as pd


train_path =

def match_generation(train_path:str):
    
    '''
    segment the train dataset by utterances
    
    input: 
        path train.txt file original 

    Returns
    -------
        train: .csv train\num_token\generation(prompt_temp)
        
    '''
    with open(train_path, encoding="utf8") as f:
        Audiobook_lines = f.readlines()
        
        
    
    return generation
    

def prepare_generation():
    
    '''
    input:
        train: .csv; columns: train\num_token\generation(prompt_temp)
        .csv: 
            
    return 
        generation: .csv: prompts left to be genreated
    '''
    
    
    
    return generation