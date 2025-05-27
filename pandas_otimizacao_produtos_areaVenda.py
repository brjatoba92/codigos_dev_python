import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Simulando layout de loja com coordenadas e características
np.random.seed(42)

# Definindo setores da loja com características específicas
setores_loja = {
    'Entrada': {'x_range': (0, 10), 'y_range': (0, 5), 'trafego': 100, 'conversao': 0.15},
    'Cimentos_Cal': {'x_range': (0, 15), 'y_range': (5, 15), 'trafego': 60, 'conversao': 0.25},
    'Ferragens': {'x_range': (15, 25), 'y_range': (0, 10), 'trafego': 45, 'conversao': 0.30},
    'Tintas': {'x_range': (15, 25), 'y_range': (10, 20), 'trafego': 40, 'conversao': 0.20},
    'Ceramicas': {'x_range': (25, 40), 'y_range': (0, 15), 'trafego': 35, 'conversao': 0.18},
    'Madeiras': {'x_range': (25, 40), 'y_range': (15, 25), 'trafego': 30, 'conversao': 0.22},
    'Fundo_Loja': {'x_range': (40, 50), 'y_range': (0, 25), 'trafego': 15, 'conversao': 0.35}
}

# Criando grid de produtos com localização
produtos_layout = []
produto_id = 1

for setor, config in setores_loja.items():
    # Número de produtos por setor baseado no tamanho da área
    area = (config['x_range'][1] - config['x_range'][0]) * (config['y_range'][1] - config['y_range'][0])
    num_produtos = int(area / 8)  # ~1 produto a cada 8m²
    
    for i in range(num_produtos):
        x = np.random.uniform(config['x_range'][0], config['x_range'][1])
        y = np.random.uniform(config['y_range'][0], config['y_range'][1])
        
        # Simulando diferentes tipos de produtos por setor
        if setor == 'Cimentos_Cal':
            produtos_tipos = ['Cimento CP2', 'Cimento CP3', 'Cal Hidratada', 'Argamassa']
            margem_base = 0.15
        elif setor == 'Ferragens':
            produtos_tipos = ['Parafusos', 'Pregos', 'Dobradiças', 'Fechaduras']
            margem_base = 0.35
        elif setor == 'Tintas':
            produtos_tipos = ['Tinta Látex', 'Tinta Acrílica', 'Verniz', 'Primer']
            margem_base = 0.28
        elif setor == 'Ceramicas':
            produtos_tipos = ['Piso Cerâmico', 'Azulejo', 'Porcelanato', 'Rejunte']
            margem_base = 0.22
        elif setor == 'Madeiras':
            produtos_tipos = ['Tábua Pinus', 'Compensado', 'MDF', 'Sarrafo']
            margem_base = 0.18
        else:
            produtos_tipos = ['Diversos A', 'Diversos B', 'Diversos C']
            margem_base = 0.25
        
        produto_nome = np.random.choice(produtos_tipos)
        
        # Calculando métricas de performance por localização
        distancia_entrada = np.sqrt(x**2 + y**2)
        fator_visibilidade = max(0.3, 1 - (distancia_entrada / 60))  # Visibilidade decresce com distância
        
        # Vendas influenciadas por tráfego, visibilidade e conversão do setor
        vendas_mensais = (config['trafego'] * config['conversao'] * fator_visibilidade * 
                         np.random.normal(1, 0.3))
        vendas_mensais = max(5, vendas_mensais)
        
        preco_unitario = np.random.uniform(5, 200)
        receita_mensal = vendas_mensais * preco_unitario
        margem_ajustada = margem_base * np.random.normal(1, 0.15)
        
        produtos_layout.append({
            'produto_id': f'PROD{produto_id:04d}',
            'nome_produto': produto_nome,
            'setor': setor,
            'posicao_x': round(x, 1),
            'posicao_y': round(y, 1),
            'area_ocupada_m2': np.random.uniform(0.5, 3.0),
            'vendas_mensais_unid': round(vendas_mensais, 1),
            'preco_unitario': round(preco_unitario, 2),
            'receita_mensal': round(receita_mensal, 2),
            'margem_percentual': round(margem_ajustada, 3),
            'lucro_mensal': round(receita_mensal * margem_ajustada, 2),
            'distancia_entrada': round(distancia_entrada, 1),
            'fator_visibilidade': round(fator_visibilidade, 2),
            'trafego_setor': config['trafego'],
            'conversao_setor': config['conversao']
        })
        produto_id += 1

df_layout = pd.DataFrame(produtos_layout)

# Análise de rentabilidade por m²
df_layout['receita_por_m2'] = df_layout['receita_mensal'] / df_layout['area_ocupada_m2']
df_layout['lucro_por_m2'] = df_layout['lucro_mensal'] / df_layout['area_ocupada_m2']

print("Análise de Rentabilidade por Área:")
print("-" * 45)

# Identificando produtos com baixa performance para realocação
def analisar_otimizacao_espacial(df):
    """Identifica oportunidades de otimização do layout"""
    
    # Calculando percentis para classificação
    p75_receita_m2 = df['receita_por_m2'].quantile(0.75)
    p25_receita_m2 = df['receita_por_m2'].quantile(0.25)
    p75_lucro_m2 = df['lucro_por_m2'].quantile(0.75)
    
    # Classificando produtos
    df['classificacao'] = 'Médio'
    df.loc[(df['receita_por_m2'] >= p75_receita_m2) & (df['lucro_por_m2'] >= p75_lucro_m2), 'classificacao'] = 'Alto Valor'
    df.loc[(df['receita_por_m2'] <= p25_receita_m2) | (df['lucro_por_m2'] <= p25_receita_m2), 'classificacao'] = 'Baixo Valor'
    
    return df

df_layout = analisar_otimizacao_espacial(df_layout)

# Análise por setor
analise_setores = df_layout.groupby('setor').agg({
    'receita_por_m2': ['mean', 'sum'],
    'lucro_por_m2': ['mean', 'sum'],
    'area_ocupada_m2': 'sum',
    'produto_id': 'count'
}).round(2)

analise_setores.columns = ['receita_media_m2', 'receita_total_m2', 'lucro_medio_m2', 
                          'lucro_total_m2', 'area_total', 'num_produtos']

analise_setores['eficiencia_espacial'] = (analise_setores['lucro_total_m2'] / 
                                         analise_setores['area_total']).round(2)

analise_setores = analise_setores.sort_values('eficiencia_espacial', ascending=False)

print("Performance por Setor da Loja:")
for setor in analise_setores.index:
    dados = analise_setores.loc[setor]
    print(f"{setor:15} | Eficiência: R${dados['eficiencia_espacial']:6.2f}/m² | "
          f"Área: {dados['area_total']:5.1f}m² | Produtos: {dados['num_produtos']:2.0f}")

# Recomendações de realocação
print("\nProdutos Candidatos à Realocação (Baixo Valor em Área Nobre):")
candidatos_realocacao = df_layout[
    (df_layout['classificacao'] == 'Baixo Valor') & 
    (df_layout['distancia_entrada'] <= 25)  # Área nobre
].nlargest(5, 'area_ocupada_m2')

for _, produto in candidatos_realocacao.iterrows():
    print(f"{produto['produto_id']} ({produto['nome_produto']:15}) | "
          f"Setor: {produto['setor']:12} | R${produto['receita_por_m2']:6.2f}/m² | "
          f"Área: {produto['area_ocupada_m2']:4.1f}m²")
