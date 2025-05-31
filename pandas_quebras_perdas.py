import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analise_perdas_quebras():
    np.random.seed(42)

    setores_loja = {
        'Entrada': {'x_range': (0, 10), 'y_range': (0, 5), 'trafego': 100, 'conversao': 0.15},
        'Cimentos_Cal': {'x_range': (0, 15), 'y_range': (5, 15), 'trafego': 60, 'conversao': 0.25},
        'Ferragens': {'x_range': (15, 25), 'y_range': (0, 10), 'trafego': 45, 'conversao': 0.30},
        'Tintas': {'x_range': (15, 25), 'y_range': (10, 20), 'trafego': 40, 'conversao': 0.20},
        'Ceramicas': {'x_range': (25, 40), 'y_range': (0, 15), 'trafego': 35, 'conversao': 0.18},
        'Madeiras': {'x_range': (25, 40), 'y_range': (15, 25), 'trafego': 30, 'conversao': 0.22},
        'Fundo_Loja': {'x_range': (40, 50), 'y_range': (0, 25), 'trafego': 15, 'conversao': 0.35}
    }

    # Tipos de perdas específicas do setor
    tipos_perda = {
        'Quebra_Manuseio': {'freq': 0.25, 'valor_medio': 150, 'sazonalidade': [1.2, 1.0, 0.8]},  # Verão, meio, inverno
        'Vencimento': {'freq': 0.15, 'valor_medio': 80, 'sazonalidade': [0.8, 1.0, 1.3]},
        'Furto_Interno': {'freq': 0.12, 'valor_medio': 200, 'sazonalidade': [1.0, 1.1, 1.2]},
        'Furto_Externo': {'freq': 0.18, 'valor_medio': 120, 'sazonalidade': [1.1, 1.0, 1.3]},
        'Dano_Transporte': {'freq': 0.10, 'valor_medio': 300, 'sazonalidade': [1.0, 1.0, 1.0]},
        'Erro_Inventario': {'freq': 0.08, 'valor_medio': 180, 'sazonalidade': [0.9, 1.0, 1.1]},
        'Deterioracao': {'freq': 0.12, 'valor_medio': 90, 'sazonalidade': [1.4, 1.0, 0.7]}  # Umidade no verão
    }

    # Gerando dados de perdas ao longo do ano
    perdas_data = []
    data_inicio = pd.Timestamp('2024-01-01')

    for dia in range(365):
        data_atual = data_inicio + timedelta(days=dia)
    
        # Determinando estação (simplificado)
        if data_atual.month in [12, 1, 2]:
            estacao_idx = 0  # Verão
        elif data_atual.month in [6, 7, 8]:
            estacao_idx = 2  # Inverno
        else:
            estacao_idx = 1  # Meio-ano
    
        # Fator dia da semana (mais perdas em fins de semana - maior movimento)
        fator_dia_semana = 1.3 if data_atual.weekday() >= 5 else 1.0
    
        for tipo_perda, config in tipos_perda.items():
            # Probabilidade de ocorrência ajustada por sazonalidade
            prob_ocorrencia = config['freq'] * config['sazonalidade'][estacao_idx] * fator_dia_semana
        
            if np.random.random() < prob_ocorrencia:
                # Valor da perda com variação
                valor_perda = max(10, np.random.normal(config['valor_medio'], config['valor_medio'] * 0.4))
            
                # Atribuindo a produtos/setores específicos
                setor_afetado = np.random.choice(list(setores_loja.keys()))
            
                # Simulando detalhes adicionais
                turno = np.random.choice(['Manhã', 'Tarde', 'Noite'], p=[0.4, 0.4, 0.2])
                funcionario_id = f'FUNC{np.random.randint(1, 21):03d}'  # 20 funcionários
            
                perdas_data.append({
                    'data_ocorrencia': data_atual,
                    'tipo_perda': tipo_perda,
                    'setor_afetado': setor_afetado,
                    'valor_perda': round(valor_perda, 2),
                    'turno': turno,
                    'funcionario_responsavel': funcionario_id,
                    'dia_semana': data_atual.day_name(),
                    'mes': data_atual.month,
                    'estacao': ['Verão', 'Meio-ano', 'Inverno'][estacao_idx]
                })

    df_perdas = pd.DataFrame(perdas_data)

    print("Análise Detalhada de Perdas Operacionais:")
    print("-" * 50)

    # Análise multidimensional de perdas
    def analisar_perdas_complexo(df):
        """Análise avançada de padrões de perdas"""
    
        # 1. Análise temporal multi-nível
        df['valor_perda_acum'] = df.groupby([df['data_ocorrencia'].dt.to_period('M')])['valor_perda'].cumsum()
    
        # 2. Análise por combinações de fatores
        perdas_cruzadas = df.groupby(['tipo_perda', 'setor_afetado', 'turno']).agg({
                'valor_perda': ['sum', 'count', 'mean'],
            'funcionario_responsavel': 'nunique'
        }).round(2)
    
        perdas_cruzadas.columns = ['valor_total', 'ocorrencias', 'valor_medio', 'funcionarios_envolvidos']
        perdas_cruzadas = perdas_cruzadas.reset_index()
    
        # 3. Identificando padrões anômalos usando z-score
        df['valor_z_score'] = stats.zscore(df['valor_perda'])
        anomalias = df[abs(df['valor_z_score']) > 2]  # Outliers significativos
    
        return perdas_cruzadas, anomalias

    perdas_cruzadas, anomalias = analisar_perdas_complexo(df_perdas)

    # Top combinações problemáticas

    print("Top 10 Combinações Criticas (Tipo + Setor + Turno): ")
    top_problemas = perdas_cruzadas.nlargest(10, 'valor_total')
    for _, row in top_problemas.iterrows():
        print(f"{row['tipo_perda']:15} | {row['setor_afetado']:12} | {row['turno']:6} | "
        f"R${row['valor_total']:7,.0f} ({row['ocorrencias']:2.0f}x)")
    
    # analise de correlação entre funcionarios e tipos de perda
    print("\nAnálise de Funcionários com Maior Envolvimento em Perdas")
    funcionarios_perdas = (df_perdas.groupby(['funcionario_responsavel', 'tipo_perda'])['valor_perda'].sum()).reset_index().pivot(index='funcionario_responsavel', columns='tipo_perda', values='valor_perda').fillna(0)

    # Calculando score de risco por funcionário
    funcionarios_perdas['total_perdas'] = funcionarios_perdas.sum(axis=1)
    funcionarios_perdas['tipos_envolvidos'] = (funcionarios_perdas > 0).sum(axis=1)
    funcionarios_perdas['score_risco'] = (funcionarios_perdas['total_perdas'] * funcionarios_perdas['tipos_envolvidos']/100).round(2)

    funcionarios_risco = funcionarios_perdas.nlargest(5, 'score_risco')
    for func_id in funcionarios_risco.index:
        dados = funcionarios_risco.loc[func_id]
        print(f"{func_id} | Score Risco: {dados['score_risco']:5.2f} |"
        f"Total Perdas: R${dados['total_perdas']:6,.0f} |" 
        f"Tipos Envolvidos: {dados['tipos_envolvidos']:2.0f}")

analise_perdas_quebras()