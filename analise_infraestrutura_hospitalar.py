import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES ---
BASE_DIR = Path(r"D:\2025\UFJF\ETLSIH")
LOGS_DIR = Path("logs")
RELATORIOS_DIR = Path("relatorios")
LOGS_DIR.mkdir(exist_ok=True)
RELATORIOS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Dicionário de CNES conhecidos em Juiz de Fora (para dar nome aos códigos)
# Fonte: Dados públicos do CNES
MAPA_HOSPITAIS_JF = {
    2153033: "SANTA CASA DE MISERICORDIA",
    2152991: "HOSPITAL UNIVERSITARIO UFJF (HU)",
    2153025: "HOSPITAL DE PRONTO SOCORRO (HPS)",
    2152967: "HOSPITAL DR JOAO PENIDO",
    2769727: "HOSPITAL REGIONAL JOAO PENIDO",
    2152983: "HOSPITAL ASCOMCER",
    2153017: "HOSPITAL ALBERT SABIN",
    2153051: "HOSPITAL MONTE SINAI",
    2152975: "MATERNIDADE TEREZINHA DE JESUS"
}

def processar_arquivos_brutos():
    logger.info("Iniciando varredura de infraestrutura por CNES...")
    
    arquivos = sorted(list(BASE_DIR.glob("ETLSIH.ST_MG_*.csv")))
    
    stats_hospitais = {}
    
    for arquivo in arquivos:
        if arquivo.stat().st_size < 1024: continue
        
        try:
            # Ler apenas colunas necessárias para economizar memória
            # CNES: Código do Estabelecimento
            cols = ['CNES', 'MUNIC_MOV', 'DT_INTER', 'DT_SAIDA', 'DIAS_PERM']
            
            chunks = pd.read_csv(arquivo, encoding='latin-1', sep=',', usecols=lambda c: c in cols, chunksize=50000)
            
            for chunk in chunks:
                # Filtrar JF
                if 'MUNIC_MOV' in chunk.columns:
                    df = chunk[chunk['MUNIC_MOV'] == 313670].copy()
                else:
                    continue
                
                if df.empty: continue
                
                # Agrupar por CNES
                for cnes, dados in df.groupby('CNES'):
                    cnes = int(cnes)
                    if cnes not in stats_hospitais:
                        stats_hospitais[cnes] = {'internacoes': 0, 'dias_totais': 0}
                    
                    stats_hospitais[cnes]['internacoes'] += len(dados)
                    # Soma dias de permanência para estimar leitos necessários
                    # Se DIAS_PERM for 0 ou nulo, assume 1 dia para cálculo de ocupação
                    dias = pd.to_numeric(dados['DIAS_PERM'], errors='coerce').fillna(1)
                    dias = dias.replace(0, 1) 
                    stats_hospitais[cnes]['dias_totais'] += dias.sum()
                    
        except Exception as e:
            logger.warning(f"Erro ao ler {arquivo.name}: {e}")

    return stats_hospitais

def calcular_capacidade_estimada(stats, num_meses=29):
    """
    Estima o número de leitos ativos usando a fórmula de Little e Taxa de Ocupação.
    Leitos Necessários = (Total Dias Paciente / Dias no Período) / Taxa Ocupação Alvo (0.85)
    """
    logger.info("Calculando estimativa de leitos...")
    
    resultados = []
    dias_no_periodo = num_meses * 30
    
    for cnes, dados in stats.items():
        nome = MAPA_HOSPITAIS_JF.get(cnes, f"CNES {cnes}")
        
        media_mensal_pacientes = dados['internacoes'] / num_meses
        total_dias_ocupados = dados['dias_totais']
        
        # Média de leitos ocupados simultaneamente (Censo Médio Diário)
        censo_medio = total_dias_ocupados / dias_no_periodo
        
        # Leitos estimados para rodar com 85% de ocupação (folga de segurança)
        leitos_estimados = int(np.ceil(censo_medio / 0.85))
        
        resultados.append({
            'CNES': cnes,
            'Estabelecimento': nome,
            'Internacoes_Totais': dados['internacoes'],
            'Media_Pacientes_Mes': round(media_mensal_pacientes, 1),
            'Dias_Paciente_Total': int(total_dias_ocupados),
            'Leitos_Estimados_Ativos': leitos_estimados
        })
    
    df_res = pd.DataFrame(resultados)
    df_res.sort_values('Internacoes_Totais', ascending=False, inplace=True)
    return df_res

def main():
    # 1. Processar
    # Assume 29 arquivos conforme seu log anterior
    stats = processar_arquivos_brutos()
    
    # 2. Calcular
    df_capacidade = calcular_capacidade_estimada(stats, num_meses=29)
    
    # 3. Exibir e Salvar
    print("\n" + "="*80)
    print("Mapeamento de Infraestrutura Real - Juiz de Fora (Estimado via SIH)")
    print("="*80)
    print(df_capacidade[['Estabelecimento', 'Media_Pacientes_Mes', 'Leitos_Estimados_Ativos']].to_string(index=False))
    
    caminho_csv = RELATORIOS_DIR / "cenario_real_hospitais.csv"
    df_capacidade.to_csv(caminho_csv, index=False)
    logger.info(f"\nRelatório salvo em: {caminho_csv}")
    
    # 4. Sugestão para Simulação
    top_hospital = df_capacidade.iloc[0]
    print("\n" + "="*80)
    print("SUGESTÃO PARA CALIBRAR A SIMULAÇÃO (Baseado no maior hospital):")
    print(f"Hospital Alvo: {top_hospital['Estabelecimento']}")
    print(f">> Ajustar NUMERO_LEITOS para: {top_hospital['Leitos_Estimados_Ativos']}")
    print(f">> Ajustar tamanho da amostra (n) para: {int(top_hospital['Media_Pacientes_Mes'])}")
    print("="*80)

if __name__ == "__main__":
    main()