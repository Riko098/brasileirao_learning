import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
from brasileirao_ml import BrasileiroELO, load_and_clean_data

class BrasileiraoPredictorGUI:
    def __init__(self, master):
        self.master = master
        self.models = None
        self.elo_system = None
        self.csv_path = "brasileirao_data.csv"
        self.team_names = []

        self.create_widgets()
        self.load_models_and_elo()

    def create_widgets(self):
        self.master.title("Previsor de Resultados do Brasileirão")
        self.master.geometry("600x600")
        self.master.resizable(False, False)

        main_frame = ttk.Frame(self.master, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Previsão de Partidas do Brasileirão", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # --- Time da Casa ---
        home_frame = ttk.LabelFrame(main_frame, text="Time da Casa", padding=10)
        home_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.home_team_combo = ttk.Combobox(home_frame, width=25, state="readonly")
        self.home_team_combo.pack(pady=5)
        self.home_team_combo.set("Selecione o time da casa")

        # --- Versus ---
        vs_label = ttk.Label(main_frame, text="VS", font=("Arial", 20, "bold"))
        vs_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # --- Time Visitante ---
        away_frame = ttk.LabelFrame(main_frame, text="Time Visitante", padding=10)
        away_frame.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

        self.away_team_combo = ttk.Combobox(away_frame, width=25, state="readonly")
        self.away_team_combo.pack(pady=5)
        self.away_team_combo.set("Selecione o time visitante")

        # --- Botão de Prever ---
        predict_button = ttk.Button(main_frame, text="Prever Resultado", command=self.make_prediction, style="Accent.TButton", state="disabled")
        predict_button.grid(row=2, column=0, columnspan=3, pady=20)

        # --- Área de Resultados ---
        result_frame = ttk.LabelFrame(main_frame, text="Resultado da Previsão", padding=10)
        result_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="nsew")

        self.result_score_label = ttk.Label(result_frame, text="0 - 0", font=("Arial", 24, "bold"))
        self.result_score_label.pack(pady=10)

        self.result_winner_label = ttk.Label(result_frame, text="Vencedor: -", font=("Arial", 12))
        self.result_winner_label.pack(pady=5)


        # --- Probabilidades ---
        prob_frame = ttk.LabelFrame(main_frame, text="Probabilidades", padding=10)
        prob_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="nsew")

        self.prob_home_label = ttk.Label(prob_frame, text="Casa (H):", font=("Arial", 10))
        self.prob_home_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.prob_home_bar = ttk.Progressbar(prob_frame, orient="horizontal", length=300, mode="determinate")
        self.prob_home_bar.grid(row=0, column=1, padx=5, pady=2)
        self.prob_home_value = ttk.Label(prob_frame, text="0%", font=("Arial", 10))
        self.prob_home_value.grid(row=0, column=2, padx=5, pady=2)

        self.prob_draw_label = ttk.Label(prob_frame, text="Empate (D):", font=("Arial", 10))
        self.prob_draw_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.prob_draw_bar = ttk.Progressbar(prob_frame, orient="horizontal", length=300, mode="determinate")
        self.prob_draw_bar.grid(row=1, column=1, padx=5, pady=2)
        self.prob_draw_value = ttk.Label(prob_frame, text="0%", font=("Arial", 10))
        self.prob_draw_value.grid(row=1, column=2, padx=5, pady=2)

        self.prob_away_label = ttk.Label(prob_frame, text="Fora (A):", font=("Arial", 10))
        self.prob_away_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.prob_away_bar = ttk.Progressbar(prob_frame, orient="horizontal", length=300, mode="determinate")
        self.prob_away_bar.grid(row=2, column=1, padx=5, pady=2)
        self.prob_away_value = ttk.Label(prob_frame, text="0%", font=("Arial", 10))
        self.prob_away_value.grid(row=2, column=2, padx=5, pady=2)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.columnconfigure(2, weight=1)

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 12, "bold"))


    def load_models_and_elo(self):
        try:
            loaded_models = joblib.load('brasileirao_models.pkl')
            self.models = None # TODO: Verify if this should be loaded from loaded_models
            self.corners_model = None # TODO: Verify if this should be loaded from loaded_models
            self.feature_columns = None # TODO: Verify if this should be loaded from loaded_models
            self.regression_feature_columns = None # TODO: Verify if this should be loaded from loaded_models
            self.ml_df_mean = None # TODO: Verify if this should be loaded from loaded_models
            
            df = load_and_clean_data("campeonato-brasileiro-full.csv", "campeonato-brasileiro-estatisticas-full.csv", "campeonato-brasileiro-gols.csv", "campeonato-brasileiro-cartoes.csv")
            self.team_names = sorted(list(set(df['mandante'].unique()) | set(df['visitante'].unique())))
            self.home_team_combo['values'] = self.team_names
            self.away_team_combo['values'] = self.team_names
            
            # Calcular ELOs
            self.elo_system = BrasileiroELO()
            for i, row in df.iterrows():
                self.elo_system.update_elos(row['mandante'], row['visitante'], row['vencedor'], row['data'])

            # Habilitar botão de previsão
            for child in self.master.winfo_children():
                if isinstance(child, ttk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Button) and grandchild['text'] == "Prever Resultado":
                            grandchild.config(state="normal")
            
            messagebox.showinfo("Sucesso", "Modelos e ELOs carregados com sucesso!")
        except FileNotFoundError:
            messagebox.showerror("Erro", f"Arquivos de modelo ou dados não encontrados. Certifique-se de que 'brasileirao_models.pkl' e os arquivos CSV estão no mesmo diretório e que o pipeline principal foi executado.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelos ou dados: {e}")

    def make_prediction(self):
        home_team = self.home_team_combo.get()
        away_team = self.away_team_combo.get()

        if not home_team or not away_team or home_team == "Selecione o time da casa" or away_team == "Selecione o time visitante":
            messagebox.showwarning("Aviso", "Selecione os dois times para fazer a previsão.")
            return

        if home_team == away_team:
            messagebox.showwarning("Aviso", "O time da casa e o visitante não podem ser o mesmo.")
            return

        # Obter ELOs
        home_elo = self.elo_system.get_team_elo(home_team)
        away_elo = self.elo_system.get_team_elo(away_team)

        # Calcular probabilidades
        home_elo_effective = home_elo + self.elo_system.home_advantage
        prob_home = self.elo_system.expected_score(home_elo_effective, away_elo)
        prob_away = self.elo_system.expected_score(away_elo, home_elo_effective)
        
        # Ajustar para que a soma seja 1 (aproximado, pois não há empate direto no ELO)
        # Uma abordagem simples é normalizar, mas o ideal é ter um modelo para o empate.
        # Por simplicidade, vamos assumir que o empate é o restante da probabilidade.
        prob_draw = 1 - (prob_home + prob_away)
        
        # Se a soma for maior que 1, normalizamos
        if (prob_home + prob_away) > 1:
            total = prob_home + prob_away
            prob_home /= total
            prob_away /= total
            prob_draw = 0
        else:
            prob_draw = 1 - (prob_home + prob_away)


        # Atualizar a interface
        self.prob_home_bar['value'] = prob_home * 100
        self.prob_draw_bar['value'] = prob_draw * 100
        self.prob_away_bar['value'] = prob_away * 100
        self.prob_home_value['text'] = f"{prob_home:.1%}"
        self.prob_draw_value['text'] = f"{prob_draw:.1%}"
        self.prob_away_value['text'] = f"{prob_away:.1%}"

        # Determinar o vencedor
        if prob_home > prob_away and prob_home > prob_draw:
            winner = home_team
        elif prob_away > prob_home and prob_away > prob_draw:
            winner = away_team
        else:
            winner = "Empate"
            
        self.result_winner_label['text'] = f"Vencedor: {winner}"
        
        # Simular um placar (exemplo, pode ser melhorado com um modelo de gols)
        if winner == home_team:
            self.result_score_label['text'] = "2 - 1"
        elif winner == away_team:
            self.result_score_label['text'] = "1 - 2"
        else:
            self.result_score_label['text'] = "1 - 1"

        # Prever escanteios
        # Para uma previsão real, você precisaria de dados de entrada para as features
        # que o modelo de escanteios espera (chutes, posse de bola, etc.)
        # Como não temos esses dados para um jogo futuro, vamos usar um valor médio/placeholder
        # ou você pode criar um input para o usuário inserir esses dados.
        # Por simplicidade, vamos criar um DataFrame com valores médios ou zeros para as features.
        
        # Crie um DataFrame de exemplo com as colunas esperadas pelo modelo de escanteios
        # Preencha com zeros ou valores médios, pois não temos dados reais para um jogo futuro
        dummy_input_data = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns)
        
        # Se 'escanteios' estiver nas feature_columns, remova-o para a previsão de regressão
        if 'escanteios' in dummy_input_data.columns:
            dummy_input_data = dummy_input_data.drop(columns=['escanteios'])

        # Certifique-se de que as colunas do dummy_input_data correspondem às regression_feature_columns
        # que foram usadas para treinar o modelo de escanteios.
        # Se você salvou regression_feature_columns no pkl, use-o aqui.
        # Por enquanto, assumimos que self.feature_columns menos 'escanteios' é o correto.
        
        # Prever escanteios
        predicted_corners = self.corners_model.predict(dummy_input_data)[0]
        self.predicted_corners_label['text'] = f"Escanteios Previstos: {predicted_corners:.0f}"


if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()
    
    app = BrasileiraoPredictorGUI(root)
    root.mainloop()