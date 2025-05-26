import pandas as pd
import numpy as np


def analisar_concorrencia_precos():
    """Análise competitiva de preços e estratégias de precificação"""
    np.random.seed(42)
    # Dados de pesquisa de preços da concorrência
    produtos_estrategicos = ['Cimento 50kg', 'Tijolo Cerâmico', 'Tinta Látex 18L', 
                           'Ferro 10mm', 'Piso Cerâmico 60x60', 'Areia Lavada']
    
    concorrentes = ['Loja A', 'Loja B', 'Loja C', 'Loja D', 'Nossa Loja']
    
    precos_data = []
    
    for produto in produtos_estrategicos:
        # Definir preço base por produto
        if 'Cimento' in produto:
            preco_base = 32.0
        elif 'Tijolo' in produto:
            preco_base = 1.20
        elif 'Tinta' in produto:
            preco_base = 120.0
        elif 'Ferro' in produto:
            preco_base = 12.0
        elif 'Piso' in produto:
            preco_base = 65.0
        else:  # Areia
            preco_base = 50.0
        
        for concorrente in concorrentes:
            # Simular variação de preços entre concorrentes
            if concorrente == 'Nossa Loja':
                preco = preco_base
            else:
                variacao = np.random.uniform(-0.15, 0.20)  # -15% a +20%
                preco = preco_base * (1 + variacao)
            
            precos_data.append({
                'produto': produto,
                'concorrente': concorrente,
                'preco_atual': round(preco, 2),
                'data_pesquisa': pd.Timestamp('2024-12-01'),
                'disponibilidade': np.random.choice(['Em Estoque', 'Sob Consulta', 'Indisponível'], 
                                                  p=[0.8, 0.15, 0.05]),
                'prazo_entrega': np.random.choice([0, 1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.08, 0.02]),
                'forma_pagamento': np.random.choice(['À Vista/PIX', 'Cartão', 'Prazo'], p=[0.4, 0.3, 0.3])
            })
    
    df_precos = pd.DataFrame(precos_data)
    
    # Análise competitiva por produto
    analise_competitiva = []
    
    for produto in produtos_estrategicos:
        df_produto = df_precos[df_precos['produto'] == produto]
        nosso_preco = df_produto[df_produto['concorrente'] == 'Nossa Loja']['preco_atual'].iloc[0]
        
        # Estatísticas do mercado
        precos_mercado = df_produto[df_produto['concorrente'] != 'Nossa Loja']['preco_atual']
        preco_medio_mercado = precos_mercado.mean()
        preco_min_mercado = precos_mercado.min()
        preco_max_mercado = precos_mercado.max()
        
        # Posicionamento
        percentil_nosso_preco = (precos_mercado < nosso_preco).sum() / len(precos_mercado) * 100
        
        if percentil_nosso_preco <= 25:
            posicionamento = 'Muito Competitivo'
            recomendacao = 'Manter preço ou aumentar margem'
        elif percentil_nosso_preco <= 50:
            posicionamento = 'Competitivo'
            recomendacao = 'Preço adequado'
        elif percentil_nosso_preco <= 75:
            posicionamento = 'Acima da Média'
            recomendacao = 'Considerar redução ou agregar valor'
        else:
            posicionamento = 'Caro'
            recomendacao = 'Reduzir preço urgentemente'
        
        analise_competitiva.append({
            'produto': produto,
            'nosso_preco': nosso_preco,
            'preco_medio_mercado': round(preco_medio_mercado, 2),
            'preco_min_mercado': round(preco_min_mercado, 2),
            'preco_max_mercado': round(preco_max_mercado, 2),
            'diferenca_media': round(((nosso_preco / preco_medio_mercado) - 1) * 100, 1),
            'posicionamento': posicionamento,
            'recomendacao': recomendacao
        })
    
    df_analise = pd.DataFrame(analise_competitiva)
    
    # Análise de elasticidade simulada (impacto de mudança de preço nas vendas)
    elasticidade_data = []
    
    for produto in produtos_estrategicos:
        produto_info = df_analise[df_analise['produto'] == produto].iloc[0]
        
        # Simular diferentes cenários de preço
        cenarios_preco = [-10, -5, 0, 5, 10]  # % de mudança
        
        for mudanca_pct in cenarios_preco:
            novo_preco = produto_info['nosso_preco'] * (1 + mudanca_pct/100)
            
            # Elasticidade simulada (produtos básicos = mais elásticos)
            if produto in ['Cimento 50kg', 'Areia Lavada']:
                elasticidade = -1.5  # Muito elástico
            elif produto in ['Tijolo Cerâmico', 'Ferro 10mm']:
                elasticidade = -1.2  # Elástico
            else:
                elasticidade = -0.8  # Menos elástico
            
            # Impacto na demanda
            impacto_demanda = elasticidade * mudanca_pct
            vendas_esperadas = 100 * (1 + impacto_demanda/100)  # Base 100 unidades
            
            elasticidade_data.append({
                'produto': produto,
                'mudanca_preco_pct': mudanca_pct,
                'novo_preco': round(novo_preco, 2),
                'impacto_demanda_pct': round(impacto_demanda, 1),
                'vendas_esperadas': round(vendas_esperadas, 0)
            })
    
    df_elasticidade = pd.DataFrame(elasticidade_data)
    
    # Produtos críticos (preço muito alto vs concorrência)
    produtos_criticos = df_analise[df_analise['posicionamento'].isin(['Caro', 'Acima da Média'])]
    
    print("\n=== ANÁLISE DE CONCORRÊNCIA E PREÇOS ===")
    print("Posicionamento Competitivo:")
    print(df_analise[['produto', 'nosso_preco', 'preco_medio_mercado', 'diferenca_media', 'posicionamento']])
    
    if len(produtos_criticos) > 0:
        print(f"\n⚠️  PRODUTOS CRÍTICOS ({len(produtos_criticos)} produtos):")
        print(produtos_criticos[['produto', 'diferenca_media', 'recomendacao']])
    
    print("\nAnálise de Elasticidade (Cenário -5% no preço):")
    cenario_desconto = df_elasticidade[df_elasticidade['mudanca_preco_pct'] == -5]
    print(cenario_desconto[['produto', 'novo_preco', 'impacto_demanda_pct', 'vendas_esperadas']])
    
    return df_precos, df_analise, produtos_criticos


analisar_concorrencia_precos()