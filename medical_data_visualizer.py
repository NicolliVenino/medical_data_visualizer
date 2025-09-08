import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2

# Cria coluna overweight usando numpy 
if 'weight' in df.columns and 'height' in df.columns:
    df['overweight'] = np.where(df['weight'] / (df['height']/100)**2 > 25, 1, 0)

# Outra forma: 

# if ((df['weight'] / (df['height']/100)**2) > 25):
#     df['overweight'] = 1
# else:
#     df['overweight'] = 0

# Outra forma:

# df.loc[ (df['weight'] / df['height']^2) > 25, 'overweight'] = 1

# 3

df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
def draw_cat_plot():
    # Seleciona as colunas cholesterol e gluc
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc'])
    
    # Plot categórico
    fig = sns.catplot(
        data=df_cat, 
        x="variable", 
        hue="value", 
        kind="count"
    ).fig
    
    # Ajusta título
    fig.suptitle("Distribuição categórica de Cholesterol e Gluc")

    # 5
    df_cat = pd.melt(
        df, 
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7

    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Cria gráfico de contagem de recursos categóricos com Seaborn
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    ).fig

    fig.suptitle("Contagens de variáveis categóricas por presença de doença cardíaca", y=1.05)

    # 8
    fig = fig.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():

    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".1f", 
        center=0, 
        cmap="coolwarm", 
        square=True, 
        cbar_kws={"shrink": 0.5}
    )

    ax.set_title("Mapa de calor das correlações")


    # 16
    fig.savefig('heatmap.png')
    return fig