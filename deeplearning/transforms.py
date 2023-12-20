from abc import abstractmethod
import warnings, pandas as pd
from . import util 
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

def remove_nulls(df, cols_for_null_removal, verbose=False):
    if verbose:
        print('--- Number of nulls in columns for null removal: ---')
        print(df[cols_for_null_removal].isna().sum())
        print(f"Number of rows before null removal: {len(df)}\n")                         
    for col in cols_for_null_removal:
        df = df[df[col].isna() == False]
    if verbose:
        print(f"Number of rows after null removal: {len(df)}")
    return df

class Transform():
    """These transforms follow a similar API to sklearn transforms, but operate on pandas dataframes.
    This is so that a transform can receive an entire df as input but know what subset of columns to operate on
    without requiring its caller to pass in that knowledge

    Init functions should result in a transformer that is ready to apply transformations (in sklearn jargon, init should 
    result in a fitted transformer).
    """    
    def __init__(self, for_inf=False, **kwargs) -> None:
        "If this transform should be applied to inference data when it comes time for production, for_inf should be True"
        self.for_inf=for_inf

    @abstractmethod
    def transform(self, df, **kwargs) -> pd.DataFrame:
        """ All children must implement this, and they must follow this function signature. The returned df should be the transformed df. 
            One kwarg that is useful is 'inference_mode', see KeepOneRedundantSubstringTfm and Pipeline for examples
        """
        pass

    def __call__(self, df, *args, **kwargs) -> pd.DataFrame:
        return self.transform(df, *args, **kwargs)

    def update_ds_obj(self, ds, **kwargs) -> Dataset:
        """If the transformation is being done from a dataset and you need to update the dataset so it doesn't break, implement this function
        See OneHotCatTfm for an example.
        """
        return ds
 
# ----------------------------------------------------------------------------------------------------------------------------------------
#preprocessing transforms section. Tranforms that do not require being fitted to data passed at init.
# ----------------------------------------------------------------------------------------------------------------------------------------
class RemoveNullsTfm(Transform):
    """Removes nulls. You may want to create one tfm for removing nulls from features and set for_inf to True, and another for removing nulls from the target 
    and set for_inf=False. Note that this transform does not require being fitted to data passed at init.
    """
    def __init__(self, cols_for_null_removal, for_inf=True, **kwargs) -> None:
        super().__init__(for_inf=for_inf, **kwargs)
        if not isinstance(cols_for_null_removal, list):
            cols_for_null_removal = [cols_for_null_removal]
        self.cols_for_null_removal = cols_for_null_removal

    def transform(self, df, verbose=False, **kwargs) -> pd.DataFrame:
        """Removes nulls from df[self.cols_for_null_removal]. If a column in self.columns_for_null_removal doesn't exist on df, it is ignored.
        This is useful if you want to remove nulls from TGT columns in development but ignore at inference (since there won't be TGT columns).

        Args:
            df (pd.DataFrame): incoming df of data
            verbose (bool, optional): if True, more print statements. Defaults to False.

        Returns:
            pd.DataFrame: df with all rows removed that had null values in cols_for_null_removal 
        """
        cols_for_null_removal = [col for col in self.cols_for_null_removal if col in df.columns]
        if len(cols_for_null_removal) < len(self.cols_for_null_removal):
            warnings.warn(f"Some of cols_for_null_removal are not in df and will be excluded. Cols excluded: {list(set(self.cols_for_null_removal) - set(cols_for_null_removal))}")
        return remove_nulls(df, cols_for_null_removal, verbose=verbose)


# ---------------------------------------------------------------------------------------------------------------------------------------- 
# This section are tranforms that require being fitted to data passed at init. These typically should be done after all transforms that remove rows
# ---------------------------------------------------------------------------------------------------------------------------------------- 

class NormalizeTfm(Transform):
    "Normalizes continuous feature data according to the mean and std of the training data provided at init"
    def __init__(self, df_train=None, tr_mean=None, tr_std=None, x_cont_cols=None, for_inf=True, **kwargs) -> None:
        """

        Args:
            df_train (pd.DataFrame, optional): if provided, gets the mean and std for x_cont_cols with df_train[x_cont_cols].mean() and .std(). Defaults to None.
            tr_mean (Iterable, optional): means of x_cont_cols in the training set. If df_train is included, this is ignored. Defaults to None.
            tr_std (Iterable, optional): standard deviations of x_cont_cols in the training set. If df_train is included, this is ignored. Defaults to None.
            x_cont_cols (Iterable, optional): column names of continuous features. These are the only parts of df that will be normalized. Defaults to None.
            for_inf (bool, optional): if True, this transform applies at inference time. Defaults to True.
        """        
        super().__init__(for_inf=for_inf, **kwargs)
        self.x_cont_cols = x_cont_cols
        if df_train is not None:
            self.tr_mean = df_train[x_cont_cols].mean()
            self.tr_std = df_train[x_cont_cols].std()
        elif tr_mean is not None and tr_std is not None:
            self.tr_mean, self.tr_std = tr_mean, tr_std
        else:
            self.tr_mean, self.tr_std = None, None
            warnings.warn("Since you did not provide df_train, you must run `transform(df, is_train_df=True)` where df is your df_train.\n" + \
                "This will set self.tr_mean and self.tr_std based on this df. All subsequent calls to transform (ie for val, test, or inf sets) should be with is_train_df=False")

    def transform(self, df, is_train_df=False, **kwargs) -> pd.DataFrame:
        if is_train_df and self.tr_mean is None:    
            self.tr_mean = df[self.x_cont_cols].mean()
            self.tr_std = df[self.x_cont_cols].std()
        if self.tr_mean is None or self.tr_std is None:
            raise ValueError("self.tr_mean and/or self.tr_std are None. You either need to pass df_train at init or call transform with is_train_df=True")
        df.loc[:,self.x_cont_cols] = util.normalize(df[self.x_cont_cols], self.tr_mean, self.tr_std)
        return df

class OneHotCatTfm(Transform):
    "Performs one-hot encodings of categorical data according to the categories in the training set provided at init"
    def __init__(self, df_train=None, cat_cols=None, sk_kwargs={}, for_inf=True, **kwargs) -> None:
        super().__init__(for_inf=for_inf, **kwargs)
        self.cat_cols = cat_cols
        self.sk_tfm = OneHotEncoder(**sk_kwargs).fit(df_train[cat_cols].values)
        # Determine columns for one hot variables
        self.onehotcols = []
        categories = self.sk_tfm.categories_
        for i,col in enumerate(self.cat_cols):
            self.onehotcols.extend([f'{col}_{cat}' for cat in categories[i]])

    
class Pipeline(Transform):
    "A special type of Transform that is actually a sequence of transforms. This allows for grouping transforms into abstract pipelines"
    def __init__(self, tfms:list=[]) -> None:
        """
        Args:
            tfms (list): the list of transforms that makeup the pipeline
        """        
        self.tfms = tfms
        for tfm in tfms:
            if tfm.__class__.__name__ == 'NormalizeTfm' and tfm.tr_mean is None:
                warnings.warn("A NormalizeTfm has been found in your pipeline without a tr_mean. Be aware that it must come after any transforms that " + \
                    "remove rows for it to calculate the training mean and std properly")
            if tfm.__class__.__name__ == 'AddLossWeightColTfm' and tfm.minbucket_cutoff is None:
                warnings.warn("A AddLossWeightColTfm has been found in your pipeline without a minbucket_cutoff. Be aware that it should come after any transforms that " + \
                    "remove rows for it to calculate the training minbucket_cutoff and maxbucket_cutoff properly")
        super().__init__(for_inf=True) # for_inf is set to True, so that a Pipeline is never ruled out from running using for_inf. 
                                       # Instead, pipelines can be run in inference mode or not, which can give them different behavior if necessary
    
    def transform(self, df, inference_mode:bool=False, **kwargs):
        """Executes self.tfms on df sequentially, treating inference_mode appropriately

        Args:
            df (pd.DataFrame): a pandas dataframe to run through the transform pipeline
            inference_mode (bool, optional): if True, only tfms with for_inf=True will be run, and each tfm will receive inference_mode=inference_mode to its transform method. Defaults to False.

        Returns:
            pd.DataFrame: The transformed dataframe, with the index reset
        """        
        for tfm in self.tfms:
            if inference_mode and not tfm.for_inf:
                continue
            else:
                df = tfm(df, inference_mode=inference_mode, **kwargs).reset_index(drop=True)
        return df
    
    def __getitem__(self, idx):
        "So that you can index the pipeline object's tfms like a list to get self.tfms[idx]"
        return self.tfms[idx]
    
    def __setitem__(self, idx, val):
        "So that you can index the pipeline object's tfms like a list to set self.tfms[idx]"
        self.tfms[idx] = val
    
    def __delitem__(self, idx):
        "So that you can index the pipeline object's tfms like a list to delete self.tfms[idx]"
        del self.tfms[idx]
    
    def __repr__(self) -> str: 
        "Represents the inner tfm objects as a string for easy viewing. This is what print(pipeline) shows"
        current_repr = self.__class__.__name__ + ': ['
        child_reprs = []
        for tfm in self.tfms:
            child_reprs.append(tfm.__repr__())
        inner_repr = '\n  '.join(child_reprs)
        return current_repr + '\n  ' + inner_repr + '\n]'    

class PrelimPipeline(Pipeline): 
    "A pipeline object we frequently use for all tasks, and should likely be the first of transforms in a bigger pipeline"
    def __init__(self, y_cols) -> None:
        tfms = [
            RemoveNullsTfm(y_cols, for_inf=False),
        ]
        super().__init__(tfms)

class TabPipeline(Pipeline):
    "A common pipeline for the tabular modality. Since this pipeline has a NormalizeTfm, it should likely be the last in a bigger pipeline"
    def __init__(self, x_cols, x_cont_cols=None, x_cat_cols=None, sk_kwargs={}) -> None:
        if x_cont_cols is None: x_cont_cols = x_cols
        if x_cat_cols is None: x_cat_cols = []
        tfms = [
            RemoveNullsTfm(x_cols, for_inf=True),
            # OneHotCatTfm(cat_cols=x_cat_cols, sk_kwargs=sk_kwargs),
            NormalizeTfm(x_cont_cols=x_cont_cols),
        ]
        super().__init__(tfms)
