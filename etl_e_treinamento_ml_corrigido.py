import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import List, Optional, Dict
import logging
import joblib
import matplotlib.pyplot as plt
from scipy import stats

# Importações de ML
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- CONFIGURAÇÃO DE PASTAS E LOGGING ---
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
DADOS_PROCESSADOS_DIR = Path("dados_processados")
LOGS_DIR = Path("logs")
MODELO_DIR = Path("modelo_salvo")
GRAFICOS_DIR = Path("graficos")
RELATORIOS_DIR = Path("relatorios")

for pasta in [DADOS_PROCESSADOS_DIR, LOGS_DIR, MODELO_DIR, GRAFICOS_DIR, RELATORIOS_DIR]:
    pasta.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'etl_e_treinamento.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURAÇÕES GERAIS ---
CID_GRAVIDADE_ALTA = {
    'I21', 'I22', 'I60', 'I61', 'A41', 'J80', 'S06', 'R57', 'J96',
    'K35', 'N17', 'I50', 'J18', 'A40', 'R65', 'G93', 'E87'
}

CID_CIRURGICO = {
    '0303', '0404', '0505', '0415', '0304', '0403', '0306'
}

# --- FUNÇÕES DE ETL MELHORADAS ---
def processar_chunk_avancado(chunk: pd.DataFrame) -> pd.DataFrame:
    """Processa chunk com filtros e transformações básicas."""
    # Tenta identificar a coluna de município (pode variar nos arquivos do SUS)
    col_municipio = 'MUNIC_MOV' if 'MUNIC_MOV' in chunk.columns else 'MUNIC_RES'
    
    if col_municipio not in chunk.columns:
        return pd.DataFrame()
    
    # Filtra Juiz de Fora (313670)
    chunk_filtrado = chunk[chunk[col_municipio] == 313670].copy()
    
    if not chunk_filtrado.empty:
        for coluna in ['DT_INTER', 'DT_SAIDA', 'IDADE']:
            if coluna in chunk_filtrado.columns:
                chunk_filtrado[coluna] = pd.to_numeric(chunk_filtrado[coluna], errors='coerce')
    
    return chunk_filtrado

def criar_features_avancadas_sem_vazamento(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cria features avançadas GARANTINDO que não há Data Leakage."""
    logger.info("Criando features avançadas (Safety Mode: ON)...")
    
    dataframe = dataframe.copy()
    
    # Conversão de datas
    dataframe['DT_INTER'] = pd.to_datetime(dataframe['DT_INTER'], format='%Y%m%d', errors='coerce')
    dataframe['DT_SAIDA'] = pd.to_datetime(dataframe['DT_SAIDA'], format='%Y%m%d', errors='coerce')
    dataframe.dropna(subset=['DT_INTER', 'DT_SAIDA'], inplace=True)
    
    # Length of Stay (variável alvo)
    dataframe['LOS_DIAS'] = (dataframe['DT_SAIDA'] - dataframe['DT_INTER']).dt.days
    dataframe = dataframe[(dataframe['LOS_DIAS'] >= 0) & (dataframe['LOS_DIAS'] <= 365)].copy()

    # Processamento de diagnósticos
    dataframe['DIAG_PRINC'] = dataframe['DIAG_PRINC'].astype(str).str.strip().str.upper()
    dataframe['CID_CATEGORIA'] = dataframe['DIAG_PRINC'].str[:3]
    dataframe['CID_CAPITULO'] = dataframe['DIAG_PRINC'].str[:1]
    
    # Processamento de diagnósticos secundários para comorbidades
    colunas_diag_secundarios = ['DIAG_SECUN', 'DIAGSEC1', 'DIAGSEC2', 'DIAGSEC3', 'DIAGSEC4', 
                               'DIAGSEC5', 'DIAGSEC6', 'DIAGSEC7', 'DIAGSEC8', 'DIAGSEC9']
    
    colunas_diag_presentes = [col for col in colunas_diag_secundarios if col in dataframe.columns]
    dataframe['NUM_COMORBIDADES'] = 0
    
    for coluna in colunas_diag_presentes:
        # Conta apenas se tiver valor válido diferente de zero
        mask = (dataframe[coluna].notna()) & (dataframe[coluna].astype(str) != '0000') & (dataframe[coluna].astype(str) != '0')
        dataframe.loc[mask, 'NUM_COMORBIDADES'] += 1

    # Idade e categorização
    dataframe['IDADE'] = pd.to_numeric(dataframe['IDADE'], errors='coerce')
    dataframe['IDADE_CAT'] = pd.cut(dataframe['IDADE'], bins=[0, 18, 40, 60, 80, 120], 
                                   labels=['0-18', '19-40', '41-60', '61-80', '80+'])
    
    # CORREÇÃO CRÍTICA: NÃO CRIAR FEATURES BASEADAS EM VALOR FINANCEIRO (VAL_TOT, ETC)
    # Essas colunas causavam vazamento de dados. Elas foram removidas desta etapa.
    
    # Indicador de procedimento complexo (mantido pois o código do procedimento existe na entrada)
    if 'PROC_REA' in dataframe.columns:
        dataframe['PROCEDIMENTO_COMPLEXO'] = dataframe['PROC_REA'].astype(str).str[:4].isin(CID_CIRURGICO).astype(int)
    
    # Categorização de variáveis
    for coluna in ['SEXO', 'CAR_INT', 'COMPLEX', 'FINANC']:
        if coluna in dataframe.columns:
            dataframe[coluna] = dataframe[coluna].astype('category')
    
    logger.info(f"Features criadas. Dataset com {len(dataframe):,} registros.")
    return dataframe

def classificar_prioridade_sem_vazamento(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Classificação de prioridade sem usar informações futuras."""
    logger.info("Aplicando classificação de prioridade...")
    
    def calcular_gravidade_segura(linha):
        diagnostico = str(linha.get('DIAG_PRINC', ''))[:3]
        carater_internacao = str(linha.get('CAR_INT', ''))
        idade = linha.get('IDADE', 0)
        
        # Lógica clínica:
        # 1: Emergência Imediata (IAM, AVC, Sepse)
        # 2: Urgência (Pneumonia grave, Idosos > 75)
        # 3: Eletivo Prioritário
        # 4: Eletivo Puro
        if diagnostico in CID_GRAVIDADE_ALTA:
            return 1
        elif (diagnostico in {'J18', 'N17', 'I50', 'I20', 'J96'}) or (carater_internacao == '02'):
            return 2
        elif idade >= 75:  
            return 2
        elif (carater_internacao == '01') or (diagnostico in {'Z51', 'Z00', 'Z40'}):
            return 4
        else:
            return 3
    
    dataframe['gravidade_gi'] = dataframe.apply(calcular_gravidade_segura, axis=1)
    logger.info("Classificação de prioridade concluída.")
    return dataframe

def criar_escore_complexidade(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cria escore de complexidade do paciente."""
    logger.info("Criando escore de complexidade...")
    
    dataframe = dataframe.copy()
    dataframe['ESCORE_COMPLEXIDADE'] = 0
    
    # Idade avançada
    dataframe.loc[dataframe['IDADE'] >= 75, 'ESCORE_COMPLEXIDADE'] += 2
    dataframe.loc[(dataframe['IDADE'] >= 60) & (dataframe['IDADE'] < 75), 'ESCORE_COMPLEXIDADE'] += 1
    
    # Múltiplas comorbidades
    dataframe.loc[dataframe['NUM_COMORBIDADES'] >= 3, 'ESCORE_COMPLEXIDADE'] += 2
    dataframe.loc[dataframe['NUM_COMORBIDADES'] == 2, 'ESCORE_COMPLEXIDADE'] += 1
    
    # Categorizar complexidade
    conditions = [
        dataframe['ESCORE_COMPLEXIDADE'] >= 4,
        (dataframe['ESCORE_COMPLEXIDADE'] >= 2) & (dataframe['ESCORE_COMPLEXIDADE'] < 4),
        dataframe['ESCORE_COMPLEXIDADE'] < 2
    ]
    choices = ['ALTA', 'MODERADA', 'BAIXA']
    dataframe['CATEGORIA_COMPLEXIDADE'] = np.select(conditions, choices, default='MODERADA')
    
    return dataframe

# --- SEÇÃO DE MACHINE LEARNING AVANÇADO ---
def criar_pipeline_ml_avancado(features: List[str]) -> Pipeline:
    """Cria pipeline de ML com pré-processamento avançado."""
    
    features_categoricas = ['SEXO', 'CAR_INT', 'CID_CATEGORIA', 'CID_CAPITULO', 'IDADE_CAT', 'CATEGORIA_COMPLEXIDADE']
    features_numericas = ['IDADE', 'gravidade_gi', 'NUM_COMORBIDADES', 'ESCORE_COMPLEXIDADE']
    
    # Filtrar features presentes
    cat_features_present = [f for f in features_categoricas if f in features]
    num_features_present = [f for f in features_numericas if f in features]
    
    logger.info(f"Features categóricas: {cat_features_present}")
    logger.info(f"Features numéricas: {num_features_present}")
    
    preprocessador = ColumnTransformer(
        transformers=[
            ('numerico', StandardScaler(), num_features_present),
            ('categorico', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features_present)
        ],
        remainder='drop',
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessador', preprocessador),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,        # Aumentado para compensar a falta de dados financeiros
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline

def salvar_log_validacao_detalhado(X_teste, y_teste, y_previsto, nome_modelo: str = "principal"):
    """Salva log detalhado de validação com todas as métricas."""
    logger.info(f"Criando log detalhado de validação para {nome_modelo}...")
    
    dataframe_validacao = X_teste.copy()
    dataframe_validacao['LOS_REAL'] = y_teste
    dataframe_validacao['LOS_PREVISTO'] = y_previsto
    
    # Métricas de erro
    dataframe_validacao['ERRO_ABSOLUTO_DIAS'] = (dataframe_validacao['LOS_PREVISTO'] - dataframe_validacao['LOS_REAL']).abs()
    
    # Análise por clusters (Logs extra para auditoria)
    if 'CATEGORIA_COMPLEXIDADE' in dataframe_validacao.columns:
        mae_por_complexidade = dataframe_validacao.groupby('CATEGORIA_COMPLEXIDADE')['ERRO_ABSOLUTO_DIAS'].mean()
        logger.info(f"MAE por complexidade - {nome_modelo}:")
        for complexidade, mae in mae_por_complexidade.items():
            logger.info(f"  {complexidade}: {mae:.2f} dias")
            
    if 'gravidade_gi' in dataframe_validacao.columns:
        mae_por_gravidade = dataframe_validacao.groupby('gravidade_gi')['ERRO_ABSOLUTO_DIAS'].mean()
        logger.info(f"MAE por gravidade - {nome_modelo}:")
        for grav, mae in mae_por_gravidade.items():
            logger.info(f"  Gravidade {grav}: {mae:.2f} dias")
    
    caminho_saida = RELATORIOS_DIR / f"validacao_modelo_teste_{nome_modelo}.csv"
    dataframe_validacao.to_csv(caminho_saida, index=True)
    
    logger.info(f"Log de validação {nome_modelo} salvo em: {caminho_saida}")

def gerar_relatorio_e_grafico_ml_avancado(y_teste, y_previsto, r2, mae, rmse):
    """Gera relatório e gráficos de performance completos."""
    
    # Relatório de performance
    with open(RELATORIOS_DIR / 'relatorio_performance_ml_avancado.txt', 'w', encoding='utf-8') as arquivo:
        arquivo.write("RELATÓRIO DE PERFORMANCE - MODELO PREDITIVO DE LOS (CORRIGIDO)\n")
        arquivo.write("=" * 70 + "\n\n")
        arquivo.write(f"Métrica R² (R-squared): {r2:.4f}\n")
        arquivo.write(f"Erro Médio Absoluto (MAE): {mae:.2f} dias\n")
        arquivo.write(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} dias\n\n")
        
        # Estatísticas adicionais
        erro_absoluto = np.abs(y_previsto - y_teste)
        arquivo.write(f"Mediana do Erro Absoluto: {np.median(erro_absoluto):.2f} dias\n")
        arquivo.write(f"Erro Absoluto Máximo: {np.max(erro_absoluto):.2f} dias\n")
    
    logger.info(f"Relatório de performance avançado salvo em {RELATORIOS_DIR}")

    # Gráfico de performance (Matplotlib completo)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_teste, y_previsto, alpha=0.4, edgecolor='k', s=20)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('LOS Real')
    plt.ylabel('LOS Previsto')
    plt.title(f'Real vs Previsto (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    residuos = y_previsto - y_teste
    plt.scatter(y_previsto, residuos, alpha=0.4, edgecolor='k', s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Previsto')
    plt.ylabel('Resíduos')
    plt.title('Resíduos')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(residuos, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Resíduos (dias)')
    plt.title('Distribuição dos Resíduos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAFICOS_DIR / 'performance_modelo_ml_avancado.png', dpi=300)
    plt.close()
    
    logger.info(f"Gráficos de performance salvos em {GRAFICOS_DIR}")

def treinar_e_avaliar_modelo_avancado(dataframe_ml: pd.DataFrame):
    """Função completa para treinar e avaliar modelo com Cross-Validation."""
    logger.info("--- INICIANDO FASE DE TREINAMENTO ---")
    
    # Features base (SEM colunas financeiras)
    features_base = [
        'IDADE', 'SEXO', 'CAR_INT', 'gravidade_gi', 
        'CID_CATEGORIA', 'CID_CAPITULO', 'IDADE_CAT',
        'NUM_COMORBIDADES', 'ESCORE_COMPLEXIDADE', 'CATEGORIA_COMPLEXIDADE'
    ]
    
    features_presentes = [f for f in features_base if f in dataframe_ml.columns]
    
    X = dataframe_ml[features_presentes]
    y = dataframe_ml['LOS_DIAS']
    
    # Split
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=X['CATEGORIA_COMPLEXIDADE']
    )
    
    logger.info(f"Dataset de treino: {len(X_treino):,} registros")
    logger.info(f"Dataset de teste: {len(X_teste):,} registros")
    
    # --- CORREÇÃO DE VIÉS: SAMPLE WEIGHTS ---
    # Dá mais peso para gravidade 1 e 2 (casos críticos que não podem ter erro grande)
    sample_weights = X_treino['gravidade_gi'].map({1: 5.0, 2: 3.0, 3: 1.0, 4: 1.0}).fillna(1.0)
    
    # Pipeline
    pipeline_principal = criar_pipeline_ml_avancado(X_treino.columns)
    
    logger.info("Treinando modelo XGBoost (com Sample Weights)...")
    pipeline_principal.fit(X_treino, y_treino, regressor__sample_weight=sample_weights)
    
    # Avaliar
    logger.info("Avaliando performance...")
    y_previsto = pipeline_principal.predict(X_teste)
    y_previsto = np.maximum(0, y_previsto) # LOS não pode ser negativo
    
    r2 = r2_score(y_teste, y_previsto)
    mae = mean_absolute_error(y_teste, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsto))
    
    logger.info(f"Performance: R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Validação Cruzada (Para garantir robustez)
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        scores_cv = cross_val_score(pipeline_principal, X_treino, y_treino, 
                                   cv=tscv, scoring='neg_mean_absolute_error')
        logger.info(f"Validação Cruzada MAE: {-scores_cv.mean():.2f} ± {scores_cv.std():.2f}")
    except Exception as erro_cv:
        logger.warning(f"Validação cruzada falhou: {erro_cv}")
    
    # Salvar artefatos
    caminho_modelo = MODELO_DIR / 'modelo_los_avancado.pkl'
    joblib.dump(pipeline_principal, caminho_modelo)
    
    gerar_relatorio_e_grafico_ml_avancado(y_teste, y_previsto, r2, mae, rmse)
    salvar_log_validacao_detalhado(X_teste, y_teste, y_previsto, "principal_sem_leak")

def main():
    logger.info("INICIANDO PIPELINE DE EXTRAÇÃO E TREINAMENTO COMPLETO")
    
    # Busca de Arquivos
    padrao = "ETLSIH.ST_MG_*.csv"
    arquivos = sorted(list(BASE_DIR.glob(padrao)))
    
    if not arquivos:
        logger.error("Nenhum arquivo encontrado.")
        return
        
    lista_dataframes = []
    
    for arquivo in arquivos:
        if arquivo.stat().st_size < 1024: continue
        try:
            chunks = pd.read_csv(arquivo, encoding='latin-1', sep=',', chunksize=50000, low_memory=False)
            for chunk in chunks:
                proc = processar_chunk_avancado(chunk)
                if not proc.empty:
                    lista_dataframes.append(proc)
        except Exception as erro:
            logger.error(f"Erro no arquivo {arquivo.name}: {erro}")
            
    if not lista_dataframes:
        logger.error("Nenhum dado carregado.")
        return
        
    dataframe_total = pd.concat(lista_dataframes, ignore_index=True)
    logger.info(f"Total registros brutos: {len(dataframe_total):,}")
    
    # Processamento
    df_proc = criar_features_avancadas_sem_vazamento(dataframe_total)
    df_proc = classificar_prioridade_sem_vazamento(df_proc)
    df_proc = criar_escore_complexidade(df_proc)
    
    # Limpeza
    df_proc.dropna(subset=['IDADE', 'DIAG_PRINC', 'gravidade_gi', 'LOS_DIAS'], inplace=True)
    df_proc['paciente_id'] = range(1, len(df_proc) + 1)
    
    # Exportação de Datasets (Para simulação e auditoria)
    cols_ml = ['paciente_id', 'IDADE', 'IDADE_CAT', 'SEXO', 'DIAG_PRINC', 
               'CID_CATEGORIA', 'CID_CAPITULO', 'CAR_INT', 'LOS_DIAS', 
               'gravidade_gi', 'NUM_COMORBIDADES', 'ESCORE_COMPLEXIDADE', 'CATEGORIA_COMPLEXIDADE']
    
    df_proc[[c for c in cols_ml if c in df_proc.columns]].to_csv(
        DADOS_PROCESSADOS_DIR / 'dataset_para_ml.csv', index=False
    )
    
    cols_opt = ['paciente_id', 'DT_INTER', 'gravidade_gi', 'LOS_DIAS', 'ESCORE_COMPLEXIDADE', 'IDADE']
    df_proc[cols_opt].to_csv(
        DADOS_PROCESSADOS_DIR / 'dataset_para_otimizacao.csv', index=False
    )
    
    # Treinamento
    if len(df_proc) > 1000:
        treinar_e_avaliar_modelo_avancado(df_proc)
    else:
        logger.warning("Dados insuficientes.")

if __name__ == "__main__":
    main()