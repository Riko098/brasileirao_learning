#!/usr/bin/env python3
"""
Projeto Machine Learning - Previsão Brasileirão
Implementação completa usando dados CSV
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, f1_score
from sklearn.utils import class_weight
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class BrasileiroELO:
    def __init__(self, k_factor=32, home_advantage=100):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_elos = {}
        self.initial_elo = 1500
        self.elo_history = []
    
    def get_team_elo(self, team):
        """Retorna ELO atual do time"""
        return self.team_elos.get(team, self.initial_elo)
    
    def expected_score(self, elo_a, elo_b):
        """Calcula probabilidade esperada"""
        return 1 / (1 + 10**((elo_b - elo_a) / 400))
    
    def update_elos(self, home_team, away_team, result, date):
        """Atualiza ELOs após uma partida"""
        
        # ELOs antes da partida
        home_elo_before = self.get_team_elo(home_team)
        away_elo_before = self.get_team_elo(away_team)
        
        # ELO efetivo (com vantagem de casa)
        home_elo_effective = home_elo_before + self.home_advantage
        away_elo_effective = away_elo_before
        
        # Probabilidades esperadas
        home_expected = self.expected_score(home_elo_effective, away_elo_effective)
        away_expected = 1 - home_expected
        
        # Resultado real
        if result == 'H':
            home_actual, away_actual = 1, 0
        elif result == 'A':
            home_actual, away_actual = 0, 1
        else:  # Empate
            home_actual, away_actual = 0.5, 0.5
        
        # Novos ELOs
        home_elo_new = home_elo_before + self.k_factor * (home_actual - home_expected)
        away_elo_new = away_elo_before + self.k_factor * (away_actual - away_expected)
        
        # Atualizar ELOs
        self.team_elos[home_team] = home_elo_new
        self.team_elos[away_team] = away_elo_new
        
        # Salvar no histórico
        self.elo_history.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_elo_before': home_elo_before,
            'away_elo_before': away_elo_before,
            'home_elo_after': home_elo_new,
            'away_elo_after': away_elo_new,
            'home_expected': home_expected,
            'result': result
        })

def load_and_clean_data(full_csv, stats_csv, gols_csv, cartoes_csv):
    """Carrega e limpa os dados dos CSVs"""
    
    # Carregar dados
    df_full = pd.read_csv(full_csv)
    df_stats = pd.read_csv(stats_csv)
    df_gols = pd.read_csv(gols_csv)
    df_cartoes = pd.read_csv(cartoes_csv)
    
    # Merge dataframes
    df = pd.merge(df_full, df_stats, left_on='ID', right_on='partida_id', how='left', suffixes=('', '_stats'))
    df = pd.merge(df, df_gols, left_on='ID', right_on='partida_id', how='left', suffixes=('', '_gols'))
    df = pd.merge(df, df_cartoes, left_on='ID', right_on='partida_id', how='left', suffixes=('', '_cartoes'))
    
    # Converter data
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    
    # Remover jogos sem resultado
    df = df.dropna(subset=['vencedor', 'mandante_Placar', 'visitante_Placar'])
    
    # Corrigir a coluna 'vencedor'
    df['vencedor'] = df.apply(lambda row: 'H' if row['vencedor'] == row['mandante'] else ('A' if row['vencedor'] == row['visitante'] else 'D'), axis=1)
    
    # Ordenar por data
    df = df.sort_values('data').reset_index(drop=True)
    
    print(f"[+] Dados carregados: {len(df)} jogos")
    print(f"[+] Times únicos: {len(set(df['mandante'].unique()) | set(df['visitante'].unique()))}")
    
    return df



def prepare_ml_data(df):
    """Prepara dataset final para ML"""
    
    print("[+] Preparando dados para ML...")
    
    ml_df = df.copy()

    # Converter colunas de porcentagem para float
    for col in ['posse_de_bola', 'precisao_passes']:
        if ml_df[col].dtype == 'object':
            ml_df[col] = ml_df[col].str.replace('%', '').astype(float) / 100.0
    
    # Features finais
    feature_columns = [
        'chutes',
        'chutes_no_alvo',
        'posse_de_bola',
        'passes',
        'precisao_passes',
        'faltas',
        'cartao_amarelo',
        'cartao_vermelho',
        'impedimentos',
        'escanteios'
    ]
    
    # Remover jogos com dados insuficientes
    ml_df = ml_df.dropna(subset=feature_columns + ['escanteios']) # Ensure 'escanteios' is not NaN for regression
    
    # Separar features e target para classificação
    X_clf = ml_df[feature_columns]
    y_clf = ml_df['vencedor']

    # Features para regressão (excluindo 'escanteios' como feature para prever escanteios)
    regression_feature_columns = [col for col in feature_columns if col != 'escanteios']
    X_reg = ml_df[regression_feature_columns]
    y_reg_corners = ml_df['escanteios']
    
    print(f"[+] Dataset final: {len(X_clf)} jogos, {len(feature_columns)} features")
    print(f"[+] Distribuição (classificação): {y_clf.value_counts().to_dict()}")
    print(f"[+] Escanteios (regressão): Média={y_reg_corners.mean():.2f}, Desvio Padrão={y_reg_corners.std():.2f}")
    
    return X_clf, y_clf, X_reg, y_reg_corners, ml_df, feature_columns, regression_feature_columns


def train_classification_models(X, y, feature_columns):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.utils import class_weight

    """Treina múltiplos modelos e compara performance"""
    
    # Dividir dados cronologicamente (80/20)
    split_point = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"\n[+] Divisão dos dados:")
    print(f"  Treino: {len(X_train)} jogos")
    print(f"  Teste: {len(X_test)} jogos")
    
    models = {}
    results = {}
    
    # Calcular pesos para balancear as classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    
    sample_weights_rf = y_train.map(class_weights_dict)
    
    # 1. Random Forest
    print("\n[+] Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train.fillna(0), y_train, sample_weight=sample_weights_rf)
    rf_pred = rf.predict(X_test.fillna(0))
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['Random Forest'] = rf
    results['Random Forest'] = rf_accuracy
    
    print(f"  Accuracy: {rf_accuracy:.3f}")
    
    # 2. XGBoost
    print("\n[+] Treinando XGBoost com GridSearchCV...")
    
    # Encoding para XGBoost
    label_map = {'H': 0, 'D': 1, 'A': 2}
    y_train_encoded = y_train.map(label_map)
    y_test_encoded = y_test.map(label_map)
    
    # Calcular sample_weights para XGBoost (usando os mesmos pesos de classe)
    sample_weights_xgb = y_train_encoded.map({label_map[cls]: weight for cls, weight in class_weights_dict.items()})
    
    # Definir grade de parâmetros para otimização
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    xgb_base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False, # Para evitar o warning
        eval_metric='mlogloss' # Para evitar o warning
    )
    
    grid_search_xgb = GridSearchCV(
        estimator=xgb_base_model,
        param_grid=param_grid_xgb,
        scoring='f1_macro', # Otimizar para F1-score macro-médio
        cv=3, # 3-fold cross-validation
        verbose=1,
        n_jobs=-1
    )
    
    grid_search_xgb.fit(X_train.fillna(0), y_train_encoded, sample_weight=sample_weights_xgb)
    
    xgb_model = grid_search_xgb.best_estimator_
    xgb_pred_encoded = xgb_model.predict(X_test.fillna(0))
    
    # Decodificar previsões
    reverse_label_map = {0: 'H', 1: 'D', 2: 'A'}
    xgb_pred = [reverse_label_map[pred] for pred in xgb_pred_encoded]
    
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = xgb_accuracy
    
    print(f"  Melhores parâmetros XGBoost: {grid_search_xgb.best_params_}")
    print(f"  Accuracy do melhor modelo XGBoost: {xgb_accuracy:.3f}")
    
    # Relatório detalhado do melhor modelo
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    
    print(f"\n[+] RESULTADOS FINAIS:")
    print(f"Random Forest: {results['Random Forest']:.3f}")
    print(f"XGBoost: {results['XGBoost']:.3f}")
    print(f"Melhor Modelo: {best_model_name} ({best_accuracy:.3f})")
    
    if best_model_name == 'XGBoost':
        print(f"\n[+] Relatório Detalhado (XGBoost - Melhor Modelo):")
        report = classification_report(y_test, xgb_pred, output_dict=True)
        print(f"  Casa (H): Precisão {report['H']['precision']:.3f}, Recall {report['H']['recall']:.3f}, F1-Score {report['H']['f1-score']:.3f}")
        print(f"  Empate (D): Precisão {report['D']['precision']:.3f}, Recall {report['D']['recall']:.3f}, F1-Score {report['D']['f1-score']:.3f}")
        print(f"  Visitante (A): Precisão {report['A']['precision']:.3f}, Recall {report['A']['recall']:.3f}, F1-Score {report['A']['f1-score']:.3f}")
        print(f"  F1-Macro (Média): {report['macro avg']['f1-score']:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    else:
        # Manter o relatório do Random Forest se for o melhor
        print(f"\n[+] Relatório Detalhado (Random Forest):")
        report = classification_report(y_test, rf_pred, output_dict=True)
        print(f"  Casa (H): Precisão {report['H']['precision']:.3f}, Recall {report['H']['recall']:.3f}, F1-Score {report['H']['f1-score']:.3f}")
        print(f"  Empate (D): Precisão {report['D']['precision']:.3f}, Recall {report['D']['recall']:.3f}, F1-Score {report['D']['f1-score']:.3f}")
        print(f"  Visitante (A): Precisão {report['A']['precision']:.3f}, Recall {report['A']['recall']:.3f}, F1-Score {report['A']['f1-score']:.3f}")
        print(f"  F1-Macro (Média): {report['macro avg']['f1-score']:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    
    print(f"\n[+] Top 10 Features Mais Importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return models, results, feature_importance, feature_columns

def get_sample_input(ml_df, regression_feature_columns):
    """Gera um DataFrame de exemplo para previsão usando a média das features."""
    sample_input = pd.DataFrame(ml_df[regression_feature_columns].mean()).T
    return sample_input
    





def train_regression_model(X_reg, y_reg, feature_columns):
    """Treina modelo de regressão para escanteios"""
    print("[+] Treinando modelo de regressão para escanteios...")
    
    split_point = int(len(X_reg) * 0.8)
    X_train_reg, X_test_reg = X_reg.iloc[:split_point], X_reg.iloc[split_point:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_point], y_reg.iloc[split_point:]

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_reg.fillna(0), y_train_reg)
    
    reg_pred = reg_model.predict(X_test_reg.fillna(0))
    rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))
    r2 = r2_score(y_test_reg, reg_pred)
    
    print(f"  RMSE (Escanteios): {rmse:.3f}")
    print(f"  R2 Score (Escanteios): {r2:.3f}")
    
    return reg_model

def main_pipeline(full_csv, stats_csv, gols_csv, cartoes_csv):
    """Pipeline completo do projeto"""
    
    print("PROJETO ML - PREVISÃO BRASILEIRÃO")
    print("="*50)
    
    # 1. Carregar dados
    df = load_and_clean_data(full_csv, stats_csv, gols_csv, cartoes_csv)
    
    # 2. Preparar dados para ML
    X_clf, y_clf, X_reg, y_reg_corners, ml_df, feature_columns, regression_feature_columns = prepare_ml_data(df)
    
    # 3. Treinar modelos de classificação
    models, results, feature_importance, feature_columns_clf = train_classification_models(X_clf, y_clf, feature_columns)
    
    # 4. Treinar modelo de regressão para escanteios
    corners_model = train_regression_model(X_reg, y_reg_corners, regression_feature_columns)
    
    # 5. Salvar modelos e dados
    print(f"[+] Salvando resultados...")
    try:
        joblib.dump({'classification_models': models, 'corners_model': corners_model, 'feature_columns': feature_columns, 'regression_feature_columns': regression_feature_columns}, 'brasileirao_models.pkl')
        ml_df.to_csv('brasileirao_ml_data.csv', index=False)
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("  [+] Arquivos salvos com sucesso!")
    except Exception as e:
        print(f"  [-] Erro ao salvar: {e}")
    
    print("[+] PIPELINE COMPLETO!")
    print(f"[+] Melhor accuracy (classificação): {max(results.values()):.1%}")
    
    return models, corners_model, ml_df, feature_importance

