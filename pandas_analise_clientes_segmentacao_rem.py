import pandas as pd
import numpy as np
def segmentacao_clientes_rfm():
    """Segmentação de clientes usando análise RFM (Recency, Frequency, Monetary)"""
    """
    Recency: Tempo desde a ultima compra
    Frequency: Quantidade de compras
    Monetary: Valor total das compras
    """
    np.random.seed(42)
    
    # Dados de clientes
    clientes_data = [] 
    data_base = pd.Timestamp('2024-12-31')
    
    for i in range(150): # 150 clientes
        cliente_id = f'CLI{i+1:04d}'
        
        # Simulando diferentes perfis de clientes
        if i < 30:  # Clientes VIP e potenciais de compras regulares 0-29
            freq_compras = np.random.randint(8, 15) #8 a 15 compras
            valor_medio = np.random.uniform(3000, 8000) # R$3000 a R$8000
            ultima_compra = np.random.randint(1, 30) # 1 a 30 dias
        elif i < 80:  # Clientes regulares  30-79
            freq_compras = np.random.randint(3, 8) # 3 a 8 compras
            valor_medio = np.random.uniform(1000, 4000) # R$1000 a R$4000
            ultima_compra = np.random.randint(15, 90) # 15 a 90 dias
        else:  # Clientes esporádicos 80-149
            freq_compras = np.random.randint(1, 3) # 1 a 3 compras
            valor_medio = np.random.uniform(200, 1500) # R$200 a R$1500
            ultima_compra = np.random.randint(60, 365) #'60 a 365 dias
        
        # Dados dos clientes, adicionar na lista de clientes
        clientes_data.append({
            'cliente_id': cliente_id,
            'nome': f'Cliente {i+1}',
            'tipo': np.random.choice(['PF', 'PJ'], p=[0.6, 0.4]), # 60% PF, 40% PJ
            'ultima_compra_dias': ultima_compra,
            'frequencia_compras': freq_compras,
            'valor_total_compras': freq_compras * valor_medio * np.random.uniform(0.8, 1.2), # 80% a 120% do valor medio, adiciona 20% de variação
            'cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 'Salvador']),
            'canal_preferido': np.random.choice(['Loja Física', 'Online', 'Telefone'], p=[0.6, 0.3, 0.1]) # 60% Loja Física, 30% Online, 10% Telefone
        })
    
    df_clientes = pd.DataFrame(clientes_data)
    
    print("\n === DADOS DE CLIENTES ===")
    
    # Cálculo RFM
    df_clientes['recency'] = df_clientes['ultima_compra_dias']
    df_clientes['frequency'] = df_clientes['frequencia_compras']
    df_clientes['monetary'] = df_clientes['valor_total_compras']
    
    # Criando quantis para cada dimensão RFM
    df_clientes['r_score'] = pd.qcut(df_clientes['recency'], 5, labels=[5,4,3,2,1])  # 5 grupos iguais de 20% - 20% piores, 20% ruins, 20% medios, 20% bons, 20% melhores
    df_clientes['f_score'] = pd.qcut(df_clientes['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]) # method='first' - Ordem de aparição
    df_clientes['m_score'] = pd.qcut(df_clientes['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # Score RFM combinado
    df_clientes['rfm_score'] = (df_clientes['r_score'].astype(str) + 
                               df_clientes['f_score'].astype(str) + 
                               df_clientes['m_score'].astype(str))
    
    # Segmentação baseada no RFM
    def segmentar_cliente(row):
        r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions' # Compram recentemente, frequente e gastam muito
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers' # bons em todas as dimensões
        elif r >= 4 and f <= 2:
            return 'New Customers' # Compraram recentemente, mas pouco historico
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk' #Gastavam bem mas não compram há tempo
        elif r <= 2 and f <= 2:
            return 'Lost Customers' # Nao compram a muito tempo
        else:
            return 'Potential Loyalists' # Casos intermediarios
    
    df_clientes['segmento'] = df_clientes.apply(segmentar_cliente, axis=1)
    
    # Análise por segmento
    analise_segmentos = df_clientes.groupby('segmento').agg({
        'cliente_id': 'count', # quantidade de clientes
        'valor_total_compras': ['mean', 'sum'], # valor medio e total
        'frequencia_compras': 'mean', # frequencia média
        'ultima_compra_dias': 'mean' # recencia média
    }).round(2)
    
    # Visualização    
    print("\n=== SEGMENTAÇÃO RFM DE CLIENTES ===")
    print(f"Total de clientes analisados: {len(df_clientes)}")
    print("\nDistribuição por segmento:")
    print(df_clientes['segmento'].value_counts())
    
    print("\nAnálise detalhada por segmento:")
    print(analise_segmentos)
    
    return df_clientes, analise_segmentos

segmentacao_clientes_rfm()