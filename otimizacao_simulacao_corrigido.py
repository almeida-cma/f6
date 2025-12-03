import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import time
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES DE PASTAS ---
DADOS_PROCESSADOS_DIR = Path("dados_processados")
MODELO_DIR = Path("modelo_salvo")
RELATORIOS_DIR = Path("relatorios")
LOGS_DIR = Path("logs")

for pasta in [RELATORIOS_DIR, LOGS_DIR]:
    pasta.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'otimizacao_simulacao.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PARÂMETROS DA SIMULAÇÃO ---
NUMERO_LEITOS = 20
HORIZONTE_SIMULACAO_HORAS = 720
PESOS_MCDM = {'paciente': 0.6, 'eficiencia': 0.3, 'estabilidade': 0.1}
TAXA_OCUPACAO_ALVO = 0.85

# --- IMPORTAÇÃO PYMOO ---
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.config import Config
    Config.warnings['not_compiled'] = False
    PYMOO_DISPONIVEL = True
    logger.info("Biblioteca Pymoo encontrada. Otimização NSGA-II ativa.")
except ImportError:
    PYMOO_DISPONIVEL = False
    logger.warning("Pymoo não encontrada. Usando modo de fallback.")

@dataclass
class PacienteSimulacao:
    paciente_id: int
    tempo_chegada_ci: int
    gravidade_gi: int
    escore_complexidade: int
    duracao_di_predita: int
    idade: int

class ProblemaAlocacaoUTICorrigido(ElementwiseProblem):
    """
    Problema de otimização CORRIGIDO.
    Foca em penalizar pesadamente a espera de pacientes de alta gravidade.
    """
    
    def __init__(self, pacientes_na_fila: List[PacienteSimulacao], estado_leitos: List[int], hora_atual: int):
        self.pacientes = pacientes_na_fila
        self.leitos_iniciais = np.array(estado_leitos)
        self.hora_atual = hora_atual
        
        # Variáveis de decisão: Tempo de início desejado (dentro de 48h)
        super().__init__(n_var=len(pacientes_na_fila), n_obj=2, 
                        xl=hora_atual, xu=hora_atual + 48)

    def _evaluate(self, x, out, *args, **kwargs):
        leitos_simulacao = self.leitos_iniciais.copy()
        custo_risco = 0.0
        horas_ocupadas_janela = 0.0
        
        # Ordenar execução pela sugestão do gene
        ordem_execucao = np.argsort(x)
        
        for idx in ordem_execucao:
            paciente = self.pacientes[idx]
            tempo_desejado = x[idx]
            
            # Encontrar leito livre
            leito_idx = np.argmin(leitos_simulacao)
            liberacao = leitos_simulacao[leito_idx]
            
            # O início real é limitado pela chegada do paciente e pela liberação do leito
            inicio_real = max(self.hora_atual, liberacao, paciente.tempo_chegada_ci)
            # O otimizador pode sugerir atrasar (para organizar a fila), mas não adiantar impossivelmente
            inicio_real = max(inicio_real, int(tempo_desejado))
            
            duracao = paciente.duracao_di_predita
            fim_real = inicio_real + duracao
            
            # Atualizar leito
            leitos_simulacao[leito_idx] = fim_real
            
            # --- CÁLCULO DE CUSTO (PRIORIDADE) ---
            espera = inicio_real - paciente.tempo_chegada_ci
            
            # Penalidade Exponencial (A CORREÇÃO DA LÓGICA)
            if paciente.gravidade_gi == 1:
                penalidade = (espera ** 2) * 500 # Penalidade massiva para G1
            elif paciente.gravidade_gi == 2:
                penalidade = (espera * 1.5) * 50
            else:
                penalidade = espera # Linear para G3/G4
            
            custo_risco += penalidade
            
            # --- CÁLCULO DE EFICIÊNCIA ---
            janela_fim = self.hora_atual + 48
            inicio_c = max(self.hora_atual, inicio_real)
            fim_c = min(janela_fim, fim_real)
            if fim_c > inicio_c:
                horas_ocupadas_janela += (fim_c - inicio_c)
        
        # Objetivo 2: Manter ocupação próxima de 85% (sem estourar)
        cap_total = len(self.leitos_iniciais) * 48
        ocupacao = horas_ocupadas_janela / cap_total if cap_total > 0 else 0
        desvio_ocup = abs(TAXA_OCUPACAO_ALVO - ocupacao) * 1000
        
        out["F"] = [custo_risco, desvio_ocup]

def executar_otimizacao_avancada(pacientes, leitos, hora_atual):
    if not PYMOO_DISPONIVEL or not pacientes: return None
    
    problema = ProblemaAlocacaoUTICorrigido(pacientes, leitos, hora_atual)
    algoritmo = NSGA2(pop_size=40, crossover=SBX(prob=0.9), mutation=PM(prob=0.1), 
                      sampling=IntegerRandomSampling(), eliminate_duplicates=True)
    
    res = minimize(problema, algoritmo, get_termination("n_gen", 50), verbose=False, seed=1)
    
    if res.X is not None and len(res.X) > 0:
        # Pega a solução com menor risco (Objetivo 0)
        idx_melhor = np.argmin(res.F[:, 0])
        return res.X[idx_melhor]
    return None

def otimizacao_prioridade_hibrida(pacientes):
    """Fallback se o Pymoo falhar: Ordenação simples."""
    pacientes_ordenados = sorted(pacientes, key=lambda p: (p.gravidade_gi, p.tempo_chegada_ci))
    plano = {}
    for i, p in enumerate(pacientes_ordenados):
        plano[p.paciente_id] = p.tempo_chegada_ci # Tenta alocar o mais cedo possível
    return plano

class SimulacaoUTIAvancada:
    def __init__(self, modelo_ml):
        self.modelo_ml = modelo_ml
        self.leitos_estado = [0] * NUMERO_LEITOS
        self.fila_de_espera = []
        self.cronograma_final = []
        self.log_eventos = [] # Log detalhado restaurado
        self.dataframe_pacientes = None
        self.plano_atual = {}

    def carregar_modelo_e_pacientes(self):
        try:
            df_opt = pd.read_csv(DADOS_PROCESSADOS_DIR / 'dataset_para_otimizacao.csv')
            df_ml = pd.read_csv(DADOS_PROCESSADOS_DIR / 'dataset_para_ml.csv')
            cols = df_ml.columns.difference(df_opt.columns).tolist() + ['paciente_id']
            df = pd.merge(df_opt, df_ml[cols], on='paciente_id')
            
            # Amostra para simulação
            self.dataframe_pacientes = df.sample(n=min(73, len(df)), random_state=42).copy() #ALTERADO DE 200 PARA 73
            
            # Poisson
            lamb = len(self.dataframe_pacientes) / (HORIZONTE_SIMULACAO_HORAS / 24)
            chegadas = np.cumsum(np.random.exponential(24/lamb, size=len(self.dataframe_pacientes))).astype(int)
            self.dataframe_pacientes['tempo_chegada_ci'] = np.clip(chegadas, 0, HORIZONTE_SIMULACAO_HORAS - 24)
            self.dataframe_pacientes = self.dataframe_pacientes.sort_values('tempo_chegada_ci')
            
            logger.info(f"{len(self.dataframe_pacientes)} pacientes carregados.")
        except Exception as e:
            logger.error(f"Erro carga: {e}")

    def prever_duracao_avancada(self, row):
        try:
            feats = ['IDADE', 'SEXO', 'CAR_INT', 'gravidade_gi', 'CID_CATEGORIA', 
                     'CID_CAPITULO', 'IDADE_CAT', 'NUM_COMORBIDADES', 
                     'ESCORE_COMPLEXIDADE', 'CATEGORIA_COMPLEXIDADE']
            X = pd.DataFrame([row])
            for c in feats: 
                if c not in X.columns: X[c] = 0
            dias = self.modelo_ml.predict(X[feats])[0]
            # Fator estocástico
            dias = max(1.0, dias * np.random.normal(1.0, 0.2))
            return int(dias * 24)
        except: return 72

    def executar_simulacao_avancada(self):
        logger.info(f"Simulando {HORIZONTE_SIMULACAO_HORAS} horas...")
        idx_paciente = 0
        total_pacientes = len(self.dataframe_pacientes)
        
        for hora in range(HORIZONTE_SIMULACAO_HORAS):
            # 1. Chegadas
            while idx_paciente < total_pacientes:
                prox = self.dataframe_pacientes.iloc[idx_paciente]
                if prox['tempo_chegada_ci'] == hora:
                    p = PacienteSimulacao(
                        int(prox['paciente_id']), hora, int(prox['gravidade_gi']),
                        int(prox['ESCORE_COMPLEXIDADE']), self.prever_duracao_avancada(prox), int(prox['IDADE'])
                    )
                    self.fila_de_espera.append(p)
                    self.log_eventos.append({'hora': hora, 'evento': 'CHEGADA', 'paciente_id': p.paciente_id, 'gravidade': p.gravidade_gi})
                    idx_paciente += 1
                else: break
            
            # 2. Otimização
            critico = any(p.gravidade_gi == 1 for p in self.fila_de_espera)
            if self.fila_de_espera and (hora % 6 == 0 or critico):
                subset = self.fila_de_espera[:15] # Otimiza os 15 primeiros para performance
                self.log_eventos.append({'hora': hora, 'evento': 'OTIMIZACAO_START', 'fila': len(self.fila_de_espera)})
                
                solucao = executar_otimizacao_avancada(subset, self.leitos_estado, hora)
                if solucao is not None:
                    for i, p in enumerate(subset):
                        self.plano_atual[p.paciente_id] = solucao[i]
                else:
                    self.plano_atual = otimizacao_prioridade_hibrida(subset)
            
            # 3. Alocação
            leitos_livres = [i for i, t in enumerate(self.leitos_estado) if t <= hora]
            
            if leitos_livres and self.fila_de_espera:
                # Ordenação da fila baseada no plano otimizado e gravidade
                self.fila_de_espera.sort(key=lambda p: (
                    0 if self.plano_atual.get(p.paciente_id, 9999) <= hora else 1,
                    p.gravidade_gi,
                    p.tempo_chegada_ci
                ))
                
                while leitos_livres and self.fila_de_espera:
                    paciente = self.fila_de_espera.pop(0)
                    leito_idx = leitos_livres.pop(0)
                    
                    fim_real = hora + paciente.duracao_di_predita
                    self.leitos_estado[leito_idx] = fim_real
                    
                    self.cronograma_final.append({
                        'paciente_id': paciente.paciente_id,
                        'leito': leito_idx + 1,
                        'gravidade': paciente.gravidade_gi,
                        'inicio_real': hora,
                        'fim_previsto': fim_real,
                        'tempo_espera_horas': hora - paciente.tempo_chegada_ci
                    })
                    
                    self.log_eventos.append({
                        'hora': hora, 'evento': 'ALOCACAO', 
                        'paciente_id': paciente.paciente_id, 'leito': leito_idx+1,
                        'espera': hora - paciente.tempo_chegada_ci
                    })

    def salvar_resultados_e_metricas_avancadas(self):
        if not self.cronograma_final: return
        
        # CSVs detalhados
        df_crono = pd.DataFrame(self.cronograma_final)
        df_log = pd.DataFrame(self.log_eventos)
        
        df_crono.to_csv(RELATORIOS_DIR / 'cronograma_simulacao.csv', index=False)
        df_log.to_csv(RELATORIOS_DIR / 'log_simulacao_detalhado.csv', index=False)
        
        # Relatório de Texto Rico
        with open(RELATORIOS_DIR / 'metricas_da_simulacao_avancada.txt', 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO AVANÇADO DE SIMULAÇÃO\n")
            f.write("===============================\n\n")
            f.write(f"Pacientes Atendidos: {len(df_crono)}\n")
            f.write(f"Pacientes na Fila Final: {len(self.fila_de_espera)}\n\n")
            
            esp_grav = df_crono.groupby('gravidade')['tempo_espera_horas'].mean()
            f.write("TEMPO MÉDIO DE ESPERA (Priorização):\n")
            for g, t in esp_grav.items():
                f.write(f"  Gravidade {g}: {t:.2f} horas\n")
            
            # Eficiência
            horas_ocupadas = 0
            for _, r in df_crono.iterrows():
                ini = max(0, r['inicio_real'])
                fim = min(HORIZONTE_SIMULACAO_HORAS, r['fim_previsto'])
                if fim > ini: horas_ocupadas += (fim - ini)
            
            ocup = (horas_ocupadas / (NUMERO_LEITOS * HORIZONTE_SIMULACAO_HORAS)) * 100
            f.write(f"\nEficiência Operacional:\n")
            f.write(f"  Taxa de Ocupação: {ocup:.2f}%\n")

def main():
    logger.info("--- INICIANDO SIMULAÇÃO COMPLETA ---")
    if not (MODELO_DIR / 'modelo_los_avancado.pkl').exists(): return
    
    sim = SimulacaoUTIAvancada(joblib.load(MODELO_DIR / 'modelo_los_avancado.pkl'))
    sim.carregar_modelo_e_pacientes()
    sim.executar_simulacao_avancada()
    sim.salvar_resultados_e_metricas_avancadas()
    logger.info("Simulação concluída com sucesso.")

if __name__ == "__main__":
    main()