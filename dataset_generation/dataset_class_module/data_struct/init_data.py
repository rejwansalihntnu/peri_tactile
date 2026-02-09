# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


# NB: use self.__dict__.keys() to get list of attributes
class Pop_data:
    """
    Loads data that is used by pretty much every function

    Returns
    -------
    None.

    """
    
    # this if you want to fill it yourself
    def __init__(self, data_dir=None):
        
        if data_dir != None:
            self.load(data_dir)
        
        else:
            self.pop_sp = []
            self.pop_stims = []
            self.df = []
            self.spd = []
    
    
    def load(self, data_dir):
        """
        

        Parameters
        ----------
        data_dir : str
            Directory where your data is stored - make sure the data is 
            named as below or change these names.

        Returns
        -------
        None.

        """
        
        # load neurons' stims and spiking activity
        self.pop_sp = pd.read_pickle(f"{data_dir}/pop_spikes_dict.pkl")
        self.pop_stims = pd.read_pickle(f"{data_dir}/pop_stims_dict.pkl")
        self.df = pd.read_pickle(f"{data_dir}/pop_metadata.pkl")
        self.spd = pd.read_pickle(f"{data_dir}/pop_speed_dict.pkl")
        


"""This class was never implemented as it would have been extraneous. It is 
kept here for reference"""
# maybe make this and Pop_data class have options (with default of getting all
#data) of what data to extract in case new features are added such as video-
#ography, so __init__ would have (self,.....,extract_method=None) with None
#signifying get all data as you see below, else some attributes = []
class Mouse_data:
    """
    Subsets the Pop_data to contain just the mouse data provided
    
    Returns
    -------
    None.

    """
    
    # this if you want to fill it yourself
    def __init__(self, pop_data=None, mouse_id=None):
        
        if pop_data != None and mouse_id != None:
            self.load(pop_data, mouse_id)
        
        else:
            self.sp = []
            self.stims = []
            self.df = []
            self.spd = []
    
    def load(self, pop_data, mouse_id):
        # get dataframe for this mouse (just region and cluster)
        df = pop_data.df
        self.df = df[df['mouse_id'] == mouse_id]
        self.stims = pop_data.pop_stims[mouse_id] 
        self.spd = pop_data.spd[mouse_id]
        #for getting spikes I'll change format
        # popdat_spks = pop_data.pop_sp
        # popdat_regions = popdat_spks.keys()
        # self.sp = {reg: popdat_spks[reg][mouse_id] for reg in popdat_regions}
        self.sp = pop_data.pop_sp[mouse_id]
        

def get_run_bounds(mouse_id):
    # from threshold of 5cm/s for over 1s (5ms binning, 500ms smoothing)
    run_bounds = np.load("./speed_classification/" + 
                         f"{mouse_id}_running_bounds.npy")
    
    return run_bounds


    
