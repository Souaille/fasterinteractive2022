import scipy.stats as st
import pandas as pd


def get_better_than_p_vals(df, ref_column, alternative="less", bonf=True):
    data = {}
    nb_tests = 0
    for col_name in df.columns:
        if col_name != ref_column:
            _, p_val = st.mannwhitneyu(
                df[col_name].tolist(),
                df[ref_column].tolist(),
                alternative=alternative,
            )
            nb_tests = nb_tests + 1
            data.update({col_name: [p_val]})

    # Create dataframe from data dictionnary
    p_val_df = pd.DataFrame.from_dict(data)
    if bonf:
        # Apply Bonferroni correction to p values
        p_val_df = p_val_df * nb_tests
    # Rename unique row with reference column name
    p_val_df.rename(index={0: ref_column}, inplace=True)

    return p_val_df


def get_significance(df):
    ns = df > 0.05
    three = (df < 0.001) & (df >= 0)
    two = (df < 0.01) & (df >= 0.001)
    one = (df < 0.05) & (df >= 0.01)

    df = df.astype(str)
    df[ns] = "NS"
    df[three] = "***"
    df[two] = "**"
    df[one] = "*"

    return df


def alpha_bold(val, alpha=0.05):

    bold = "bold" if val < alpha else ""

    return "font-weight: %s" % bold
