"""Importando das bibliotecas necessárias"""
import pandas as pd
import numpy as np

def analise_sazonalidade():
    meses = pd.date_range(start='2020-01-01', end='2024-12-31', freq='ME') # dados mensais entre 01/2020 e 12/2024
    np.random.seed(42) #função de aleatoridade dos dados

    vendas_data = [] # lista para armazenar as vendas
    categorias = ['Cimento', 'Piso', 'Tinta', 'Revestimentos', 'Ferro', 'Luminarias'] # categorias dos produtos

    for mes in meses: # loop para gerar vendas para cada mês
        for categoria in categorias:
            base_vendas = np.random.normal(1000, 200)
            if mes.month in[11,12,1,2,3]:
                multiplicador = np.random.uniform(1.3, 1.8) # aumento de vendas no verão
            else:
                multiplicador = np.random.uniform(0.7, 1.2) # diminuição de vendas nos demais meses
            vendas = max(0, int(base_vendas * multiplicador)) # garantindo vendas positivas
            # adicionando as vendas a lista com a função append
            vendas_data.append( 
                {
                    'data': mes,
                    'categoria': categoria,
                    'vendas': vendas,
                    'mes': mes.month,
                    'ano': mes.year,
                    'trimestre': f'T{(mes.month - 1) // 3 + 1}'
                }
            )

    vendas_df = pd.DataFrame(vendas_data) # criando um dataframe com as vendas

    # Agrupando por trimestre
    vendas_trimestre = vendas_df.groupby(['trimestre', 'categoria'])['vendas'].sum().reset_index()

    # pivotando o dataframe para mostrar as vendas por trimestre e por categoria
    vendas_pivot = vendas_trimestre.pivot(index='trimestre', columns='categoria', values='vendas')

    # vendas anual
    vendas_anual = vendas_df.groupby(['ano', 'categoria'])['vendas'].sum().reset_index()
    vendas_anual['crescimento'] = vendas_anual.groupby('categoria')['vendas'].pct_change() * 100 # calculando o crescimento anual

    print("Vendas por trimestre")
    print(vendas_pivot.round(0)) # mostrando as vendas por trimestre

    print("Crescimento Anual por Categoria")
    print(vendas_anual[vendas_anual['ano'] == 2021][['categoria', 'crescimento']].round(2)) # mostrando apenas o crescimento de 2021

    return vendas_df # retornando o dataframe


analise_sazonalidade() # executando a função