import pandas as pd
import numpy as np

def fornecedores():
    fornecedores_data = {
            'fornecedor': ['Cimentos ABC', 'Cerâmica XYZ', 'Ferro & Aço SA', 'Tintas Premium', 'Agregados Norte'],
            'categoria_produto': ['Cimento', 'Cerâmica', 'Ferro', 'Tintas', 'Agregados'],
            'entregas_prazo': [0.95, 0.88, 0.92, 0.85, 0.78],
            'qualidade_media': [4.5, 4.2, 4.8, 3.9, 3.5],  # Escala 1-5
            'preco_competitivo': [4.0, 4.5, 3.8, 4.2, 4.8],  # Escala 1-5
            'total_pedidos': [150, 120, 200, 80, 90],
            'defeitos_reportados': [3, 8, 2, 12, 15],
            'tempo_resposta_dias': [2, 4, 1, 5, 7]
        }
    
    df_fornecedores = pd.DataFrame(fornecedores_data)

    df_fornecedores['taxa_defeito'] = (df_fornecedores['defeitos_reportados'] / df_fornecedores['total_pedidos']*100).round(2)

    #score ponderado
    peso_prazo = 0.3
    peso_qualidade = 0.4
    peso_preco = 0.1
    peso_defeito = 0.2

    df_fornecedores['score_prazo'] = df_fornecedores['entregas_prazo'] * 100
    df_fornecedores['score_qualidade'] = df_fornecedores['qualidade_media'] * 20
    df_fornecedores['score_preco'] = df_fornecedores['preco_competitivo'] * 20
    df_fornecedores['penalidade_defeito'] = df_fornecedores['taxa_defeito'] * 2

    df_fornecedores['score_final'] = (
        df_fornecedores['score_prazo'] * peso_prazo +
        df_fornecedores['score_qualidade'] * peso_qualidade +
        df_fornecedores['score_preco'] * peso_preco +
        df_fornecedores['penalidade_defeito'] * peso_defeito
    ).round(1)

    #classificação
    df_fornecedores['classificacao'] = pd.cut(df_fornecedores['score_final'],
                                            bins=[0, 70, 85, 100], 
                                            labels=['Critico', 'Bom', 'Excelente'])

    resultado = df_fornecedores[['fornecedor', 'categoria_produto', 'entregas_prazo', 'qualidade_media', 'taxa_defeito', 'score_final', 'classificacao']]

    print(resultado.sort_values('score_final', ascending=False))

    return df_fornecedores

fornecedores()