import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DE PASTAS E ESTILO ---
RELATORIOS_DIR = Path("relatorios")
DASHBOARD_DIR = Path("dashboard")
DASHBOARD_DIR.mkdir(exist_ok=True)

# Configuração de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

# Paletas de cores
PALETA_GRAVIDADE = {1: '#d62728', 2: '#ff7f0e', 3: '#1f77b4', 4: '#2ca02c'}

# --- CONFIGURAÇÃO DE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FUNÇÕES ---
def carregar_dados_simulacao() -> pd.DataFrame:
    caminho_arquivo = RELATORIOS_DIR / "cronograma_simulacao.csv"
    if not caminho_arquivo.exists():
        caminho_arquivo = RELATORIOS_DIR / "cronograma_final_simulacao.csv"
        
    try:
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado em {RELATORIOS_DIR}")
            
        df = pd.read_csv(caminho_arquivo)
        
        rename_map = {}
        if 'fim_real' in df.columns: rename_map['fim_real'] = 'fim_previsto'
        if 'tempo_espera' in df.columns: rename_map['tempo_espera'] = 'tempo_espera_horas'
        df = df.rename(columns=rename_map)
        
        cols_num = ['inicio_real', 'fim_previsto', 'tempo_espera_horas', 'leito', 'gravidade']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        logger.info(f"Dados carregados: {len(df)} registros.")
        return df
    except Exception as erro:
        logger.error(f"Erro ao carregar dados: {erro}")
        return pd.DataFrame()

def analisar_metricas(df: pd.DataFrame) -> Dict:
    if df.empty: return {}
    metricas = {}
    metricas['pacientes_atendidos'] = len(df)
    metricas['tempo_medio_espera'] = df['tempo_espera_horas'].mean()
    metricas['tempo_mediano_espera'] = df['tempo_espera_horas'].median()
    metricas['espera_por_gravidade'] = df.groupby('gravidade')['tempo_espera_horas'].mean().to_dict()
    metricas['horas_ocupadas_totais'] = (df['fim_previsto'] - df['inicio_real']).sum()
    metricas['horas_disponiveis_totais'] = 20 * 720
    metricas['taxa_ocupacao'] = (metricas['horas_ocupadas_totais'] / metricas['horas_disponiveis_totais']) * 100
    return metricas

def gerar_dashboard_principal(metricas: Dict, df: pd.DataFrame):
    logger.info("Gerando Dashboard Estratégico...")
    
    fig, eixos = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('DASHBOARD DE ALOCAÇÃO DE UTI - RESULTADOS DA SIMULAÇÃO', 
                 fontsize=22, fontweight='bold', y=0.98, color='#333333')
    
    # 1. KPIs
    ax = eixos[0, 0]
    ax.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, facecolor='#f8f9fa', transform=ax.transAxes, zorder=-1)
    ax.add_patch(rect)
    
    kpis_text = [
        ("PACIENTES ATENDIDOS", f"{metricas.get('pacientes_atendidos', 0)}"),
        ("TAXA DE OCUPAÇÃO", f"{metricas.get('taxa_ocupacao', 0):.1f}%"),
        ("TEMPO MÉDIO DE ESPERA", f"{metricas.get('tempo_medio_espera', 0):.1f} h"),
        ("EFICIÊNCIA GLOBAL", f"{(metricas.get('horas_ocupadas_totais', 0)/metricas.get('horas_disponiveis_totais', 1)*100):.1f}%")
    ]
    for i, (label, valor) in enumerate(kpis_text):
        ax.text(0.1, 0.85 - i*0.2, label, fontsize=12, color='#666666', transform=ax.transAxes)
        ax.text(0.1, 0.75 - i*0.2, valor, fontsize=24, weight='bold', color='#2e86ab', transform=ax.transAxes)

    # 2. Pacientes por Gravidade
    ax = eixos[0, 1]
    if not df.empty:
        counts = df['gravidade'].value_counts().sort_index()
        colors = [PALETA_GRAVIDADE.get(g, '#666666') for g in counts.index]
        bars = ax.bar([str(i) for i in counts.index], counts.values, color=colors, alpha=0.85, edgecolor='black')
        ax.bar_label(bars, fontsize=12, weight='bold')
    ax.set_title('Volume de Atendimentos por Gravidade')
    ax.set_ylabel('Pacientes')
    
    # 3. Tempo de Espera por Gravidade
    ax = eixos[1, 0]
    if metricas.get('espera_por_gravidade'):
        gravs = sorted(metricas['espera_por_gravidade'].keys())
        vals = [metricas['espera_por_gravidade'][g] for g in gravs]
        colors = [PALETA_GRAVIDADE.get(g, '#666666') for g in gravs]
        bars = ax.bar([str(g) for g in gravs], vals, color=colors, alpha=0.85, edgecolor='black')
        ax.bar_label(bars, fmt='%.1f h', fontsize=12, weight='bold')
    ax.set_title('Tempo Médio de Espera (Priorização)')
    ax.set_ylabel('Horas')
    
    # 4. Ocupação Temporal
    ax = eixos[1, 1]
    if not df.empty:
        timeline = np.zeros(721)
        for _, row in df.iterrows():
            inicio = max(0, int(row['inicio_real']))
            fim = min(720, int(row['fim_previsto']))
            if fim > inicio:
                timeline[inicio:fim] += 1
        ax.plot(timeline, color='#2e86ab', linewidth=2)
        ax.axhline(y=20, color='red', linestyle='--', label='Capacidade Máx (20)')
        ax.fill_between(range(721), timeline, alpha=0.2, color='#2e86ab')
        ax.set_ylim(0, 25)
    ax.set_title('Curva de Ocupação de Leitos (0-720h)')
    ax.set_xlabel('Horas de Simulação')
    ax.set_ylabel('Leitos Ocupados')
    ax.legend(loc='upper right')

    # 5. Histograma de LOS (CORRIGIDO: Matplotlib Puro)
    ax = eixos[2, 0]
    if not df.empty:
        # Extrair valores para numpy array
        los_array = ((df['fim_previsto'] - df['inicio_real']) / 24).dropna().values
        
        # Usar Matplotlib direto (ax.hist) para evitar erro do Seaborn/Pandas
        ax.hist(los_array, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        media_los = np.mean(los_array)
        ax.axvline(media_los, color='red', linestyle='--', label=f'Média: {media_los:.1f} dias')
        
    ax.set_title('Distribuição do Tempo de Permanência (LOS)')
    ax.set_xlabel('Dias')
    ax.legend()

    # 6. Alocações por Turno
    ax = eixos[2, 1]
    if not df.empty:
        horas_admissao = df['inicio_real'] % 24
        turnos = pd.cut(horas_admissao, bins=[0, 6, 12, 18, 24], 
                        labels=['Madrugada', 'Manhã', 'Tarde', 'Noite'], right=False)
        counts_turno = turnos.value_counts().reindex(['Madrugada', 'Manhã', 'Tarde', 'Noite'])
        colors_turno = ['#2c3e50', '#f1c40f', '#e67e22', '#34495e']
        bars = ax.bar(counts_turno.index.astype(str), counts_turno.values, color=colors_turno, alpha=0.8, edgecolor='black')
        ax.bar_label(bars, fontsize=12)
    ax.set_title('Admissões por Turno')
    
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "dashboard_estrategico_uti.png", dpi=300)
    plt.close()
    logger.info(f"Dashboard salvo em {DASHBOARD_DIR}")

def gerar_gantt_leitos(df: pd.DataFrame):
    if df.empty: return
    logger.info("Gerando Gráfico de Gantt...")

    fig, ax = plt.subplots(figsize=(18, 10))
    df_sorted = df.sort_values(by=['leito', 'inicio_real'])
    
    for _, row in df_sorted.iterrows():
        duracao = row['fim_previsto'] - row['inicio_real']
        cor = PALETA_GRAVIDADE.get(row['gravidade'], '#999')
        ax.barh(y=row['leito'], width=duracao, left=row['inicio_real'], 
                color=cor, edgecolor='black', alpha=0.8, height=0.6)
        if duracao > 48:
            ax.text(row['inicio_real'] + duracao/2, row['leito'], 
                    f"P{int(row['paciente_id'])}", ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(range(1, 21))
    ax.set_ylabel("Leito UTI")
    ax.set_xlabel("Hora Simulação")
    ax.set_title("Ocupação Detalhada dos Leitos (Código de Cores = Gravidade)", fontweight='bold')
    
    markers = [plt.Rectangle((0,0),1,1, color=c) for c in PALETA_GRAVIDADE.values()]
    labels = [f"Gravidade {g}" for g in PALETA_GRAVIDADE.keys()]
    ax.legend(markers, labels, loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.savefig(DASHBOARD_DIR / "gantt_ocupacao_leitos.png", dpi=300, bbox_inches='tight')
    plt.close()

def gerar_relatorio_texto(metricas: Dict):
    logger.info("Gerando relatório textual...")
    caminho = DASHBOARD_DIR / "insights_executivos.txt"
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE INSIGHTS - SIMULAÇÃO UTI\n")
        f.write("=====================================\n\n")
        f.write(f"1. CAPACIDADE DE ATENDIMENTO\n")
        f.write(f"   - O sistema atendeu {metricas.get('pacientes_atendidos', 0)} pacientes.\n")
        f.write(f"   - Taxa de ocupação: {metricas.get('taxa_ocupacao', 0):.2f}%.\n")
        f.write(f"\n2. PRIORIZAÇÃO\n")
        espera = metricas.get('espera_por_gravidade', {})
        f.write(f"   - Espera G1: {espera.get(1, 0):.1f}h | Espera G3: {espera.get(3, 0):.1f}h.\n")

def main():
    logger.info("--- INICIANDO GERAÇÃO DE DASHBOARD ---")
    df = carregar_dados_simulacao()
    if df.empty: return
    metricas = analisar_metricas(df)
    gerar_dashboard_principal(metricas, df)
    gerar_gantt_leitos(df)
    gerar_relatorio_texto(metricas)
    logger.info("--- PROCESSO CONCLUÍDO ---")

if __name__ == "__main__":
    main()