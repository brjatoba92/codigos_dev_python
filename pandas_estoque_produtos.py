import pandas as pd
import numpy as np

def estoque_rotatividade():
    # carregando a tabela
    estoque_df = pd.read_csv("databases/dados_materiais_construcao.csv", sep=",")
    print(estoque_df.head())

    # calculos da rotatividade e lucratividade

    estoque_df['giro_estoque'] = estoque_df['vendas_do_mes'] / estoque_df['estoque_inicial']
    estoque_df['margem_lucro'] = ((estoque_df['preco_venda'] - estoque_df['custo_unitario']) / estoque_df['preco_venda'] * 100).round(2)
    estoque_df['receita_mes'] = (estoque_df['preco_venda'] * estoque_df['vendas_do_mes']).round(2)
    estoque_df['lucro_mes'] = (estoque_df['preco_venda'] - estoque_df['custo_unitario']) * estoque_df['vendas_do_mes'].round(2)

    # Classificação ABC por receita
    estoque_sorted_df = estoque_df.sort_values('receita_mes', ascending=False)
    estoque_sorted_df['receita_acumulada'] =  estoque_sorted_df['receita_mes'].cumsum()
    total_receita = estoque_sorted_df['receita_mes'].sum()
    estoque_sorted_df['percentual_acumulado'] = (estoque_sorted_df['receita_acumulada'] / total_receita * 100).round(2)

    def classificar_abc(perc):
        if perc <= 80:
            return 'A'
        elif perc <= 95:
            return 'B'
        else:
            return 'C'

    estoque_sorted_df['classificacao_abc'] = estoque_sorted_df['percentual_acumulado'].apply(classificar_abc)

    print("=== Classificação ABC por Receita ===")
    print(estoque_sorted_df[['produto', 'giro_estoque', 'margem_lucro', 'receita_mes', 'lucro_mes', 'classificacao_abc']].round(2))

    return estoque_sorted_df

estoque_rotatividade()

