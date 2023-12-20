from typing import Iterable, Union
from torch.utils.data import Dataset
import numpy as np, pandas as pd
from . import transforms as transforms
from . import util


class UnimodalDataset(Dataset, metaclass=util.protect('__getitem__', 'process_item')):
    """ Base class for all unimodal datasets. These are able to be used with our multimodal datasets (MultimodalDataset and MultimodalDatasetPrealigned), but can also be used stand-alone.
        
        An 'item' at index 'idx' of this dataset will be the result of retrieving raw_item = self.data[idx], calling self.process_item(raw_item), and then 
        optionally dressing the item with its label and global key. This flow is handled in __getitem__ and cannot be overridden by child classes.
        
        process_item is also special. It is the function that MultimodalDataset uses after aligning multimodal data to process each 
        modality's piece of the multimodal data, and as such it also cannot be overridden. process_item ensures that the item is a dict, then calls 
        transform_item if the item is not missing, and returns the transformed item. transform_item is the real meat of this portion, and it is what child 
        classes override to customize this flow of data.
    """
    def __init__(self, *args, data=None, x_ret_key='inputs', global_keys=None, labels=None, with_label=False, with_global_key=False, **kwargs):
        """
        Args:
            data (Iterable): the input data (no labels) for your dataset. It must be indexable, and must be able to be passed to pd.DataFrame. Defaults to None.
            x_ret_key (str, optional): the key in the dict for your item. If your item has multiple key-value pairs, you will need to implement item_to_dict yourself.
            global_keys (Iterable, optional): global keys must be provided if you plan to use this in a multimodal context and your modalities do not perfectly 
                                              overlap (data perfectly overlaps if for each unimodal_dataset, unimodal_dataset[idx] represent the same row). 
                                              Defaults to None.
            labels (Iterable, optional): labels for the data. labels[i] must be the label for data[i]. Defaults to None.
            with_label (bool, optional): If True, then _dress_item will attach the label to your item dict under the 'label' key
            with_global_key (bool, optional): If True, then _dress_item will attach the global_key to your item dict under the 'global_key' key

        Raises:
            ValueError: If global_keys is provided and len(data) != len(global_keys)
            ValueError: If labels is provided and len(data) != len(labels)
        """        
        self.data, self.x_ret_key, self.global_keys, self.with_global_key, self.with_label = \
            data, x_ret_key, global_keys, with_global_key, with_label
        ### Handle global keys
        if global_keys is not None:
            if len(data) != len(global_keys):
                raise ValueError(f"data and global_keys must be of the same length. len(data)={len(data)}, len(global_keys)={len(global_keys)}")
        ### Handle labels
        if labels is not None:
            if len(labels) == 0 or (hasattr(labels, 'size') and labels.size==0):
                labels = None
            elif len(data) != len(labels):
                raise ValueError(f"data and labels must be of the same length. len(data)={len(data)}, len(labels)={len(labels)}")
        self.labels = labels

    def get_data_with_global_keys(self, cols=None):
        """Returns a tuple of (self.data in a pd.DataFrame with self.global_keys as the index, labels).

        Args:
            cols (list of strings, optional): optional column names for self.data. Defaults to None.

        Returns:
            (pd.DataFrame, Iterable): the data as a df with global_keys as the index, and the labels
        """        
        return pd.DataFrame(self.data, index=self.global_keys, columns=cols), self.labels
    
    def is_missing(self, item) -> bool:
        """ A check on a raw item (ie before process_item is called) to see if the modality is missing. A missing item may look different for a given 
            UnimodalDataset, so this can be overridden. In this parent class, an item is missing if it has any null's/NaNs within it.
        """
        if isinstance(item, dict):
            return np.any([np.any(pd.isnull(v)) for v in item.values()])
        elif isinstance(item, Iterable):
            return np.any(pd.isnull(item))
        else:
            return item is None
    
    def item_to_dict(self, item) -> dict:
        return {self.x_ret_key: item}

    def transform_item(self, item:dict, *args, **kwargs) -> dict:
        "Given an item of data, process the item. This is used by the multimodal dataset; all custom processing of your item MUST be performed here"
        return item
    
    def process_item(self, item):
        "Handles all processing of an item which must be done for use. It cannot be overridden; instead, override 'item_to_dict' and 'transform_item'"
        item = self.item_to_dict(item)
        if not self.is_missing(item):
            item = self.transform_item(item)
        return item

    def _dress_item(self, item, idx):
        "Adds the label and the global key to the item dict, based on the corresponding options set at init"
        if not isinstance(item, dict):
            raise ValueError("item must be an instance of dict.")
        if self.with_global_key:
            gk = self.global_keys[idx]
            item['global_key'] = gk
        if self.with_label:
            item['label'] = self.labels[idx]
        return item

    def __getitem__(self, idx) -> dict:
        "Gets the item, processes it, and then returns it optionally with its global key and label. Cannot be overriden"
        item = self.data[idx]
        item = self.process_item(item)
        item = self._dress_item(item, idx)
        return item
    
    def __len__(self):
        return len(self.data)
    
class SimpleDfDataset(UnimodalDataset):
    "This class is for datasets that can take a df, x and y cols, and a pipeline as input and not require any other processing."
    def __init__(self, df:pd.DataFrame, x_cols:list, y_cols:Union[list, str]=[], pipeline=transforms.Pipeline(), pipeline_tfm_kwargs={}, \
            global_keys_cols=[], x_ret_key='inputs', with_label=False, with_global_key=False, **kwargs):
        """runs the df through the given pipeline, retrieves x values, labels, and global keys from the processed df, and then run's super's init with these

        Args:
            df (pd.DataFrame): dataframe with x_cols, y_cols, and global_keys_cols, that will be run through the pipeline object
            x_cols (list): feature columns of df
            y_cols (Union[list, str], optional): label columns of df. Defaults to [].
            pipeline (transforms.Pipeline, optional): pipeline object to run df through. Defaults to transforms.Pipeline().
            pipeline_tfm_kwargs (dict, optional): keyword arguments to use when running the pipeline. Defaults to {}.
            global_keys_cols (list, optional): columns for the global keys in df. Defaults to []
            x_ret_key (str, optional): the key in the dict for your item. If your item has multiple key-value pairs, you will need to implement item_to_dict yourself.
            with_label (bool, optional): If True, then _dress_item will attach the label to your item dict under the 'label' key. Defaults to False.
            with_global_key (bool, optional): If True, then _dress_item will attach the global_key to your item dict under the 'global_key' key. Defaults to False
        """        
        if isinstance(y_cols, str): y_cols = [y_cols]
        self.original_df, self.x_cols, self.y_cols, self.pipeline = df, x_cols, y_cols, pipeline
        self.tfm_df = pipeline(df.copy(), **pipeline_tfm_kwargs)
        extra_cols = [col for col in self.tfm_df.columns if col not in self.x_cols + self.y_cols]
        self.tfm_df = self.tfm_df[self.x_cols + self.y_cols + extra_cols] # re-order columns for aesthetic
        
        # Numpy indexing is way faster than pandas indexing, so store the numpy arrays for easy indexing
        self.x_vals = self.tfm_df[self.x_cols].values.astype(np.float32) 
        self.y_vals = self.tfm_df[self.y_cols].values.astype(np.float32) if len(y_cols)!=1 else self.tfm_df[self.y_cols[0]].values.astype(int)
        
        # Now that data has been built, perform super's init
        global_keys = self.tfm_df[global_keys_cols].values if len(global_keys_cols) > 0 else None
        super().__init__(data=self.x_vals, x_ret_key=x_ret_key, global_keys=global_keys, labels=self.y_vals, with_label=with_label, with_global_key=with_global_key)

class TabDfDataset(SimpleDfDataset):
    "A unimodal dataset for columnar, tabular data"
    def __init__(self, df:pd.DataFrame, x_cols:list, y_cols:Union[list, str]=[], pipeline:transforms.Pipeline=transforms.Pipeline(), is_train_df:bool=False, \
                 global_keys_cols=[], x_ret_key='x_cont', with_label=False, with_global_key=False) -> None:
        """ Sets pipeline_tfm_kwargs to have is_train_df, and then runs super's init

        Args:
            df (pd.DataFrame): dataframe with x_cols, y_cols, and global_keys_cols, that will be run through the pipeline object
            x_cols (list): feature columns of df
            y_cols (Union[list, str], optional): label columns of df. Defaults to [].
            pipeline (transforms.Pipeline, optional): pipeline object to run df through. Defaults to None, which causes a TabPipeline to be made.
            pipeline_tfm_kwargs (dict, optional): keyword arguments to use when running the pipeline. Defaults to {}.
            global_keys_cols (list, optional): columns for the global keys in df. Defaults to []
            x_ret_key (str, optional): the key in the dict for your item. If your item has multiple key-value pairs, you will need to implement item_to_dict yourself.
            with_label (bool, optional): If True, then _dress_item will attach the label to your item dict under the 'label' key. Defaults to False.
            with_global_key (bool, optional): If True, then _dress_item will attach the global_key to your item dict under the 'global_key' key. Defaults to False
        """        
        if pipeline is None:
            raise ValueError("Unlike other SimpleDfDatasets, a default pipeline cannot be instantiated for TabDfDataset, as you must use the same tab pipeline object for all datasets (to keep the same normalization transform). If you want no pipeline object here (as data processing has already been done), then pass `pipeline=transforms.Pipeline()`")
        pipeline_tfm_kwargs = dict(is_train_df=is_train_df)
        super().__init__(df, x_cols, y_cols, pipeline, pipeline_tfm_kwargs, global_keys_cols, x_ret_key, with_label, with_global_key)

