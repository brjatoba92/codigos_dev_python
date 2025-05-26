import pandas as pd
import numpy as np

def analise_performance_financeira():
    vendas_financeiro = {
        'vendedor': ['João Felix', 'Maria Silva', 'Pedro Santos', 'Ana Oliveira', 'Lucas Souza'] * 4,
        'regiao' : ['Norte', 'Sul', 'Centro-Oeste', 'Nordeste', 'Sudeste'] * 4,
        'mes' : ['Jan', 'Fev', 'Mar', 'Abr'] * 5,
        'vendas_brutas' : np.random.normal(50000, 15000, 20),
        'descontos' : np.random.uniform(2000, 8000, 20),
        'custos_produtos' : np.random.normal(35000, 10000, 20),
        'meta_mensal' : [45000] * 20
    }

    df_financeiro = pd.DataFrame(vendas_financeiro)
    df_financeiro['vendas_brutas'] = df_financeiro['vendas_brutas'].abs()
    df_financeiro['custos_produtos'] = df_financeiro['custos_produtos'].abs()

    # Calculos financeiros
    df_financeiro['vendas_liquidas'] = df_financeiro['vendas_brutas'] - df_financeiro['descontos']
    df_financeiro['lucro_bruto'] = df_financeiro['vendas_liquidas'] - df_financeiro['custos_produtos']
    df_financeiro['margem_bruta'] = (df_financeiro['lucro_bruto'] / df_financeiro['vendas_liquidas']* 100).round(2)
    df_financeiro['atingimento_meta'] = (df_financeiro['vendas_liquidas'] / df_financeiro['meta_mensal']* 100).round(2)

    # Performance por vendedor
    performance_vendedor = df_financeiro.groupby('vendedor').agg({
        'vendas_liquidas': 'sum',
        'lucro_bruto': 'sum',
        'margem_bruta': 'mean',
        'atingimento_meta': 'mean'  
    }).round(2)


    # performance por região
    performance_regiao = df_financeiro.groupby('regiao').agg({
        'vendas_liquidas': 'sum',
        'lucro_bruto': 'sum',
        'margem_bruta': 'mean',
    }).round(2)


    # Ranking de vendedores
    performance_vendedor['ranking'] = performance_vendedor['vendas_liquidas'].rank(ascending=False, method='dense').astype(int)
    performance_vendedor = performance_vendedor.sort_values('vendas_liquidas', ascending=False)

    print("\n == PERFORMANCE FINANCEIRA POR VENDEDOR == \n")
    print(performance_vendedor)

    print("\n == PERFORMANCE FINANCEIRA POR REGIAO == \n")
    print(performance_regiao.sort_values('vendas_liquidas', ascending=False))


    return df_financeiro

analise_performance_financeira()