import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES ---
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
DADOS_DIR = Path("dados_processados")
RELATORIOS_DIR = Path("relatorios")
ANALISE_DIR = Path("analise_estudo_dados")
ANALISE_DIR.mkdir(exist_ok=True)

# Configuração Visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ANALISE_DIR / 'log_analise_exploratoria.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AuditoriaDados:
    def __init__(self):
        self.df = None
        self.stats = {}

    def carregar_dados(self):
        """Carrega o dataset final processado (o mesmo que vai para o ML)."""
        caminho = DADOS_DIR / "dataset_para_ml.csv"
        logger.info(f"Carregando dataset de: {caminho}")
        
        try:
            self.df = pd.read_csv(caminho)
            logger.info(f"Dataset carregado: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas.")
        except FileNotFoundError:
            logger.error("Dataset não encontrado. Rode o script de ETL primeiro.")
            exit()

    def verificar_vazamento_dados(self):
        """Verifica se colunas financeiras proibidas vazaram para o dataset."""
        logger.info("--- AUDITORIA DE VAZAMENTO DE DADOS (DATA LEAKAGE) ---")
        
        colunas_proibidas = ['VAL_TOT', 'VAL_UTI', 'VAL_SH', 'VAL_SP', 'US_TOT', 'DIAS_PERM']
        vazamentos = [col for col in colunas_proibidas if col in self.df.columns]
        
        # Verificar correlação suspeita com LOS (se houver colunas numéricas não identificadas)
        cols_num = self.df.select_dtypes(include=[np.number]).columns
        correlacoes = self.df[cols_num].corr()['LOS_DIAS'].sort_values(ascending=False)
        
        suspeitas = correlacoes[correlacoes > 0.95].index.tolist()
        suspeitas = [c for c in suspeitas if c != 'LOS_DIAS']

        self.stats['auditoria_vazamento'] = {
            'colunas_financeiras_presentes': vazamentos,
            'status': 'APROVADO' if not vazamentos else 'REPROVADO',
            'correlacoes_suspeitas_95pct': suspeitas
        }
        
        if not vazamentos:
            logger.info("✅ SUCESSO: Nenhuma variável financeira detectada.")
        else:
            logger.error(f"❌ FALHA CRÍTICA: Variáveis financeiras detectadas: {vazamentos}")

    def analise_features_clinicas(self):
        """Analisa a distribuição das features criadas (Gravidade e Complexidade)."""
        logger.info("Analisando features de engenharia clínica...")
        
        # 1. Gravidade
        contagem_grav = self.df['gravidade_gi'].value_counts().sort_index()
        media_los_grav = self.df.groupby('gravidade_gi')['LOS_DIAS'].mean()
        
        # 2. Complexidade
        contagem_comp = self.df['CATEGORIA_COMPLEXIDADE'].value_counts()
        
        # 3. Cruzamento
        cruzamento = pd.crosstab(self.df['gravidade_gi'], self.df['CATEGORIA_COMPLEXIDADE'])
        
        self.stats['clinica'] = {
            'distribuicao_gravidade': contagem_grav.to_dict(),
            'los_medio_por_gravidade': media_los_grav.to_dict(),
            'distribuicao_complexidade': contagem_comp.to_dict()
        }
        
        return cruzamento

    def gerar_graficos_estudo(self):
        """Gera gráficos focados no entendimento do comportamento dos dados."""
        logger.info("Gerando gráficos de estudo...")
        
        # GRÁFICO 1: A "Realidade" do LOS (Boxplot por Gravidade)
        plt.figure(figsize=(12, 8))
        # Filtra outliers extremos visualmente (>60 dias) para o gráfico ficar legível
        df_plot = self.df[self.df['LOS_DIAS'] < 60]
        
        sns.boxplot(x='gravidade_gi', y='LOS_DIAS', data=df_plot, palette="Set2")
        plt.title('Distribuição Real do Tempo de Permanência por Gravidade (Corte < 60 dias)', fontsize=16)
        plt.xlabel('Gravidade (1 = Emergência, 4 = Eletivo)')
        plt.ylabel('Dias de Internação')
        plt.savefig(ANALISE_DIR / 'estudo_los_por_gravidade.png')
        plt.close()
        
        # GRÁFICO 2: Matriz de Correlação (Validar Relações)
        plt.figure(figsize=(12, 10))
        cols_corr = ['IDADE', 'NUM_COMORBIDADES', 'ESCORE_COMPLEXIDADE', 'gravidade_gi', 'LOS_DIAS']
        corr = self.df[cols_corr].corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title('Mapa de Calor de Correlações (Features Numéricas)', fontsize=16)
        plt.savefig(ANALISE_DIR / 'estudo_correlacoes.png')
        plt.close()
        
        # GRÁFICO 3: Pirâmide Etária da UTI
        plt.figure(figsize=(12, 6))
        if 'IDADE_CAT' in self.df.columns:
            ordem = ['0-18', '19-40', '41-60', '61-80', '80+']
            contagem = self.df['IDADE_CAT'].value_counts().reindex(ordem)
            
            bars = plt.bar(contagem.index.astype(str), contagem.values, color='#4c72b0', edgecolor='black')
            plt.bar_label(bars)
            plt.title('Perfil Demográfico dos Pacientes', fontsize=16)
            plt.xlabel('Faixa Etária')
            plt.ylabel('Quantidade')
            plt.savefig(ANALISE_DIR / 'estudo_perfil_etario.png')
            plt.close()

    def relatorio_textual_detalhado(self):
        """Gera um TXT explicativo para quem vai estudar o caso."""
        logger.info("Escrevendo relatório textual...")
        
        caminho = ANALISE_DIR / "relatorio_auditoria_dados.txt"
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE AUDITORIA E ANÁLISE EXPLORATÓRIA\n")
            f.write("===============================================\n\n")
            
            f.write("1. INTEGRIDADE DO DATASET\n")
            f.write(f"   - Total de Registros: {len(self.df)}\n")
            f.write(f"   - Colunas: {list(self.df.columns)}\n")
            f.write(f"   - Status Vazamento de Dados: {self.stats['auditoria_vazamento']['status']}\n\n")
            
            f.write("2. ANÁLISE DE GRAVIDADE (Regra de Negócio)\n")
            f.write("   Distribuição dos pacientes por nível de prioridade:\n")
            gravs = self.stats['clinica']['distribuicao_gravidade']
            los_grav = self.stats['clinica']['los_medio_por_gravidade']
            
            total = sum(gravs.values())
            for g in sorted(gravs.keys()):
                pct = (gravs[g] / total) * 100
                f.write(f"   - Nível {g}: {gravs[g]} pacientes ({pct:.1f}%) | Média LOS: {los_grav[g]:.1f} dias\n")
            
            f.write("\n3. ANÁLISE DE COMPLEXIDADE\n")
            comp = self.stats['clinica']['distribuicao_complexidade']
            for c, val in comp.items():
                f.write(f"   - {c}: {val} pacientes\n")
                
            f.write("\n4. ESTATÍSTICAS DESCRITIVAS GERAIS\n")
            desc = self.df[['IDADE', 'LOS_DIAS', 'NUM_COMORBIDADES']].describe().to_string()
            f.write(desc)

    def exportar_amostra_estudo(self):
        """Exporta um CSV pequeno (100 linhas) para inspeção visual manual."""
        amostra = self.df.head(100)
        amostra.to_csv(ANALISE_DIR / "amostra_para_inspecao_visual.csv", index=False)
        logger.info("Amostra de 100 linhas exportada.")

def main():
    logger.info("INICIANDO ANÁLISE EXPLORATÓRIA PARA ESTUDO")
    
    auditor = AuditoriaDados()
    auditor.carregar_dados()
    
    # Executar bateria de análises
    auditor.verificar_vazamento_dados()
    cruzamento = auditor.analise_features_clinicas()
    
    # Gerar saídas
    auditor.gerar_graficos_estudo()
    auditor.relatorio_textual_detalhado()
    auditor.exportar_amostra_estudo()
    
    # Salvar estatísticas brutas em JSON
    with open(ANALISE_DIR / 'estatisticas_dataset.json', 'w') as f:
        # Converter tipos numpy para nativos do python
        def convert(o):
            if isinstance(o, np.int64): return int(o)
            if isinstance(o, np.float64): return float(o)
            return o
        json.dump(auditor.stats, f, indent=4, default=convert)

    logger.info(f"Análise concluída. Resultados em: {ANALISE_DIR}")

if __name__ == "__main__":
    main()