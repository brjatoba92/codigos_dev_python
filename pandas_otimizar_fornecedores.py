import pandas as pd
import numpy as np
def otimizar_compras_fornecedores():
    """Otimização de compras e análise de relacionamento com fornecedores"""
    
    np.random.seed(42)
    
    # Dados de compras dos últimos 6 meses
    compras_data = []
    fornecedores = ['Cimentos ABC', 'Ferro & Aço SA', 'Cerâmica XYZ', 'Tintas Premium', 
                   'Agregados Norte', 'Madeiras Sul', 'Hidráulica Center']
    categorias = ['Cimento', 'Ferro', 'Cerâmica', 'Tintas', 'Agregados', 'Madeiras', 'Hidráulica']
    
    for i in range(300):  # 300 compras
        data_compra = pd.Timestamp('2024-07-01') + pd.Timedelta(days=np.random.randint(0, 180))
        fornecedor_idx = np.random.randint(0, len(fornecedores))
        
        compras_data.append({
            'compra_id': f'COMP{i+1:04d}',
            'data_compra': data_compra,
            'fornecedor': fornecedores[fornecedor_idx],
            'categoria': categorias[fornecedor_idx],
            'produto': f'Produto {np.random.choice(["A", "B", "C", "D"])}',
            'quantidade': np.random.randint(10, 500),
            'preco_unitario': np.random.uniform(5, 150),
            'valor_total': 0,  # Calculado depois
            'prazo_entrega_dias': np.random.randint(1, 15),
            'desconto_obtido': np.random.uniform(0, 15),  # %
            'qualidade_recebida': np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6]),  # 3-5
            'forma_pagamento': np.random.choice(['À Vista', '30 dias', '60 dias'], p=[0.3, 0.5, 0.2])
        })
    
    df_compras = pd.DataFrame(compras_data)
    df_compras['valor_total'] = df_compras['quantidade'] * df_compras['preco_unitario']
    df_compras['valor_com_desconto'] = df_compras['valor_total'] * (1 - df_compras['desconto_obtido']/100)
    df_compras['economia_desconto'] = df_compras['valor_total'] - df_compras['valor_com_desconto']
    
    # Análise de performance por fornecedor
    perf_fornecedor = df_compras.groupby(['fornecedor', 'categoria']).agg({
        'compra_id': 'count',
        'valor_com_desconto': 'sum',
        'preco_unitario': 'mean',
        'desconto_obtido': 'mean',
        'qualidade_recebida': 'mean',
        'prazo_entrega_dias': 'mean',
        'economia_desconto': 'sum'
    }).round(2)
    
    perf_fornecedor.columns = ['qtd_compras', 'valor_total', 'preco_medio', 'desconto_medio', 
                              'qualidade_media', 'prazo_medio', 'economia_total']
    
    # Análise de concentração de fornecedores (Risco)
    concentracao = df_compras.groupby('fornecedor')['valor_com_desconto'].sum().sort_values(ascending=False)
    concentracao_pct = (concentracao / concentracao.sum() * 100).round(1)
    
    # Identificar oportunidades de negociação
    oportunidades = []
    for fornecedor in df_compras['fornecedor'].unique():
        df_forn = df_compras[df_compras['fornecedor'] == fornecedor]
        
        volume_total = df_forn['valor_com_desconto'].sum()
        qtd_compras = len(df_forn)
        desconto_medio = df_forn['desconto_obtido'].mean()
        
        # Critérios para oportunidade de melhoria
        if volume_total > 50000 and desconto_medio < 8:
            potencial_economia = volume_total * 0.05  # 5% adicional
            oportunidades.append({
                'fornecedor': fornecedor,
                'volume_atual': volume_total,
                'desconto_atual': desconto_medio,
                'qtd_compras': qtd_compras,
                'potencial_economia': potencial_economia,
                'acao_sugerida': 'Renegociar desconto por volume'
            })
    
    df_oportunidades = pd.DataFrame(oportunidades)
    
    # Análise de sazonalidade de compras
    df_compras['mes'] = df_compras['data_compra'].dt.month
    sazonalidade_compras = df_compras.groupby(['mes', 'categoria'])['valor_com_desconto'].sum().reset_index()
    sazonalidade_pivot = sazonalidade_compras.pivot(index='mes', columns='categoria', values='valor_com_desconto').fillna(0)
    
    print("\n=== ANÁLISE DE COMPRAS E FORNECEDORES ===")
    print("Performance por Fornecedor:")
    print(perf_fornecedor.sort_values('valor_total', ascending=False).head())
    
    print(f"\nConcentração de Fornecedores (Top 3):")
    for forn, pct in concentracao_pct.head(3).items():
        print(f"{forn}: {pct}% do volume total")
    
    if len(df_oportunidades) > 0:
        print(f"\n💰 OPORTUNIDADES DE ECONOMIA ({len(df_oportunidades)} fornecedores):")
        print(df_oportunidades[['fornecedor', 'volume_atual', 'potencial_economia', 'acao_sugerida']])
        print(f"Economia total potencial: R$ {df_oportunidades['potencial_economia'].sum():,.2f}")
    
    return df_compras, perf_fornecedor, df_oportunidades

otimizar_compras_fornecedores()