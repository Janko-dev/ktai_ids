import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from prince import MCA

def PCA_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
     # one hot encode categorical variables and concatenate them with the numerical variables
    num_cols = df._get_numeric_data().columns[:-1]
    dummies_df = pd.concat([
        df[num_cols],
        pd.get_dummies(df['state']), 
        pd.get_dummies(df['service']), 
        pd.get_dummies(df['proto'])], 
        axis=1)

    # apply min-max scaler that scales the variables in the range [0-1]
    scl = MinMaxScaler()
    df = scl.fit_transform(dummies_df)

    # apply principal component analysis
    pca = PCA(n_components=0.95)
    pca_df = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_df)
    
    return pca_df

def PCA_MCA_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df._get_numeric_data().columns[:-1]

    # subtract numeric columns and classifier columns ('attack_cat', 'label') from all columns to get categorical columns 
    cat_cols = list(set(df.columns) - set(num_cols) - {'attack_cat', 'label'})
    
    # apply multiple correspondence analysis
    mca = MCA(n_components=-1)
    mca = mca.fit(df[cat_cols])
    mca_df = pd.DataFrame(mca)

    # apply min-max scaler that scales the variables in the range [0-1]
    scl = MinMaxScaler()
    df[num_cols] = scl.fit_transform(df[num_cols])

    # apply principal component analysis
    pca = PCA(n_components=0.95)
    pca = pca.fit_transform(df[num_cols])
    pca_df = pd.DataFrame(pca)

    # print(mca.row_contributions_.shape, pca.shape)
    return pd.concat([pca_df, mca_df], axis=1)
