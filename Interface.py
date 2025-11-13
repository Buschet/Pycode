import tkinter as tk
from tkinter import ttk, messagebox
import math
import datetime
import random

class RouletteSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simulatore Roulette - Inserimento da Tastiera")
        self.root.geometry("500x900+50+50")
        self.root.configure(bg='#0F5132')
        
        # Numeri della roulette europea (0-36)
        self.roulette_numbers = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        
        # Colori dei numeri
        self.number_colors = {0: '#00AA00'}  # Verde per lo 0
        red_numbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        black_numbers = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]
        
        for num in red_numbers:
            self.number_colors[num] = '#DC143C'  # Rosso
        for num in black_numbers:
            self.number_colors[num] = '#000000'  # Nero
        
        # Storia dei numeri estratti
        self.number_history = []
        self.current_input = ""
        self.radius_num = {}
        
        # Variabili per la logica di previsione
        self.predictions = []  # Mantieni le previsioni tra i numeri
        self.prediction_accuracy = []
        self.wins = 0
        self.total_predictions = 0
        
        self.setup_ui()
        self.setup_keyboard_input()
    
    def setup_ui(self):
        """Configura l'interfaccia utente"""
        # Frame principale
        main_frame = tk.Frame(self.root, bg='#0F5132')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titolo
        title_label = tk.Label(
            main_frame, 
            text="SIMULATORE",
            font=('Arial', 20, 'bold'),
            fg='#FFD700',
            bg='#0F5132'
        )
        title_label.pack(pady=(0, 20))

        # Frame principale diviso in due colonne
        content_frame = tk.Frame(main_frame, bg='#0F5132')
        content_frame.pack(fill=tk.BOTH, expand=False)
        
        # Colonna sinistra - Input e cronologia
        left_frame = tk.Frame(content_frame, bg='#0F5132')
        left_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === COLONNA SINISTRA ===
        
        # Sezione input
        input_section = tk.LabelFrame(
            left_frame, 
            text="📝 Inserimento Numero",
            font=('Arial', 12, 'bold'),
            fg='#FFD700',
            bg='#0F5132',
            padx=10, pady=10
        )
        input_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.input_label = tk.Label(
            input_section,
            text="Numero in corso: ",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#0F5132'
        )
        self.input_label.pack()
                
        # Sezione cronologia
        history_section = tk.LabelFrame(
            content_frame,
            text="📊 Cronologia Numeri",
            font=('Arial', 12, 'bold'),
            fg='#FFD700',
            bg='#0F5132',
            padx=10, pady=10
        )
        history_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Scrollable text per cronologia
        history_scroll = tk.Scrollbar(history_section)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_text = tk.Text(
            history_section,
            height=15,
            font=('Courier', 10),
            bg='#2A2A2A',
            fg='white',
            yscrollcommand=history_scroll.set,
            state=tk.DISABLED
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_text.yview)
        
        # Colonna destra - Previsioni e statistiche (più larga)


        # Sezione previsioni
        predictions_section = tk.LabelFrame(
            content_frame,
            text="🔮 Previsioni Prossimo Numero",
            font=('Arial', 12, 'bold'),
            fg='#FFD700',
            bg='#0F5132',
            padx=10, pady=10
        )
        predictions_section.pack(fill=tk.X, pady=(0, 10), expand=True)
        
        self.predictions_text = tk.Text(
            predictions_section,
            height=10,
            font=('Courier', 11, 'bold'),
            bg='#2A2A2A',
            fg='#00FF00',
            state=tk.DISABLED
        )
        self.predictions_text.pack(fill=tk.X, expand=True)
        
        
    def setup_keyboard_input(self):
        """Configura l'input da tastiera per inserire numeri"""
        self.root.focus_set()
        self.root.bind('<KeyPress>', self.handle_keypress)

    def handle_keypress(self, event):
        """Gestisce gli eventi di pressione dei tasti"""
        try:
            key = event.char
            keysym = event.keysym
            
            # Gestisce i numeri
            if key.isdigit():
                self.current_input += key
                self.input_label.config(text=f"Numero in corso: {self.current_input}")
                
                # Se l'input è completo (1-2 cifre), controlla se è valido
                if len(self.current_input) >= 2 or (len(self.current_input) == 1 and int(self.current_input) > 3):
                    self.process_number()
            
            # Gestisce Invio per confermare numero singolo
            elif keysym == 'Return':
                if self.current_input:
                    self.process_number()
            
            # Gestisce Backspace per cancellare
            elif keysym == 'BackSpace':
                if self.current_input:
                    self.current_input = self.current_input[:-1]
                    self.input_label.config(text=f"Numero in corso: {self.current_input}")
            
            # Gestisce Escape per cancellare tutto
            elif keysym == 'Escape':
                self.current_input = ""
                self.input_label.config(text="Numero in corso: ")
                
        except Exception as e:
            pass

    def process_number(self):
        """Processa il numero inserito"""
        try:
            if not self.current_input:
                return
                
            number = int(self.current_input)
            
            if 0 <= number <= 36:
                self.add_number_to_history(number)
                self.current_input = ""
                self.input_label.config(text=f"Numero in corso: {number}")
            else:
                self.current_input = ""
                self.input_label.config(text=f"Numero in corso: {number}")
                
        except ValueError:
            self.current_input = ""
            self.input_label.config(text=f"Numero in corso: {number}")

    def add_number_to_history(self, number):
        """Aggiunge un numero alla cronologia"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.number_history.append((number, timestamp))
        
        self.update_history_display()

        # Esegui la logica di previsione dopo aver aggiunto il numero
        self.prediction_logic(number)

    def update_history_display(self):
        self.printvalue=self.number_history
        # Mantiene solo gli ultimi 50 numeri per analisi più approfondite
        if len(self.printvalue) > 12:
            self.printvalue.pop(0)

        """Aggiorna la visualizzazione della cronologia"""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete('1.0', tk.END)
        
        if not self.printvalue:
            self.history_text.insert(tk.END, "Nessun numero inserito ancora...")
        else:
            self.history_text.insert(tk.END, "NUM  ORARIO   COLORE\n")
            self.history_text.insert(tk.END, "─" * 25 + "\n")
            
            for number, timestamp in reversed(self.printvalue):  # Più recenti in alto
                color_name = self.get_color_name(number)
                line = f"{number:2d}   {timestamp}   {color_name}\n"
                self.history_text.insert(tk.END, line)
        
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)


    def get_color_name(self, number):
        """Restituisce il nome del colore per un numero"""
        if number == 0:
            return "VERDE"
        elif self.number_colors[number] == '#DC143C':
            return "ROSSO"
        else:
            return "NERO"


        


    def prediction_logic(self, ultimo_numero):
        """
        Logica di previsione corretta - controlla prima se hai vinto/perso,
        poi genera nuove previsioni per il prossimo numero
        """
        result_message = ""
        
        # FASE 1: Controlla se avevi previsto correttamente il numero appena uscito
        if len(self.number_history) > 1:  # Se non è il primo numero
            if self.predictions and ultimo_numero in self.predictions:
                result_message = "🎉 HAI VINTO! 🎉\n"
                self.wins += 1
                self.status_label.config(text=f"🎉 VINCITA! Numero {ultimo_numero} era tra le tue previsioni!")
            elif self.predictions:  # Solo se c'erano previsioni da controllare
                result_message = "😞 HAI PERSO! 😞\n"
                self.status_label.config(text=f"😞 Peccato! Numero {ultimo_numero} non era tra le previsioni.")
            
            if self.predictions:  # Conta solo se c'erano previsioni
                self.total_predictions += 1
        
        # FASE 2: Genera nuove previsioni per il PROSSIMO numero
        self.generate_new_predictions()
        
        # FASE 3: Aggiorna il display
        self.update_predictions_display(result_message)

    def generate_new_predictions(self):
        """Genera nuove previsioni basate sulla cronologia"""
        self.predictions = []
        
        if len(self.number_history) < 3:
            return
        
        # Strategia semplice: numeri vicini sulla ruota + numeri caldi
        ultimo_numero = self.number_history[-1][0]
        
        # 1. Numeri adiacenti sulla ruota fisica
        try:
            index = self.roulette_numbers.index(ultimo_numero)
            # Prendi 2 numeri a sinistra e 2 a destra
            for i in range(-2, 3):
                if i != 0:  # Escludi il numero stesso
                    adj_index = (index + i) % len(self.roulette_numbers)
                    self.predictions.append(self.roulette_numbers[adj_index])
        except ValueError:
            pass
        
        # 2. Analisi frequenze (numeri "caldi")
        if len(self.number_history) >= 10:
            numbers_only = [num for num, _ in self.number_history[-20:]]  # Ultimi 20
            freq_count = {}
            for num in numbers_only:
                freq_count[num] = freq_count.get(num, 0) + 1
            
            # Aggiungi i 3 numeri più frequenti recenti
            most_frequent = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)
            for num, count in most_frequent[:3]:
                if num not in self.predictions:
                    self.predictions.append(num)
        
        # 3. Numeri dello stesso colore dell'ultimo
        if ultimo_numero != 0:  # Se non è lo zero
            same_color_nums = []
            ultimo_colore = self.number_colors[ultimo_numero]
            for num, color in self.number_colors.items():
                if color == ultimo_colore and num != ultimo_numero and num not in self.predictions:
                    same_color_nums.append(num)
            
            # Aggiungi 2 numeri casuali dello stesso colore
            if same_color_nums:
                random.shuffle(same_color_nums)
                self.predictions.extend(same_color_nums[:2])
        
        # Rimuovi duplicati e limita a 8 previsioni max
        self.predictions = list(dict.fromkeys(self.predictions))[:8]

    def update_predictions_display(self, result_message=""):
        """Aggiorna la visualizzazione delle previsioni"""
        self.predictions_text.config(state=tk.NORMAL)
        self.predictions_text.delete('1.0', tk.END)
        
        if len(self.number_history) < 3:
            self.predictions_text.insert(tk.END, "Servono almeno 3 numeri\nper generare previsioni...")
        else:
            # Mostra il risultato della previsione precedente
            if result_message:
                self.predictions_text.insert(tk.END, result_message)
                
                # Aggiungi statistiche vincite
                if self.total_predictions > 0:
                    win_rate = (self.wins / self.total_predictions) * 100
                    self.predictions_text.insert(tk.END, f"Vincite: {self.wins}/{self.total_predictions} ({win_rate:.1f}%)\n")
                self.predictions_text.insert(tk.END, "\n" + "─" * 25 + "\n")
            
            # Mostra le nuove previsioni
            self.predictions_text.insert(tk.END, "🔮 PROSSIME PREVISIONI:\n\n")
            
            if self.predictions:
                for i, pred in enumerate(self.predictions, 1):
                    color_name = self.get_color_name(pred)
                    self.predictions_text.insert(tk.END, f"{i:2}. Numero {pred:2d} ({color_name})\n")
            else:
                self.predictions_text.insert(tk.END, "Nessuna previsione generata")

        self.predictions_text.config(state=tk.DISABLED)
        self.predictions_text.see(tk.END)



















    def run(self):
        """Avvia l'applicazione"""
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Errore Applicazione", f"Si è verificato un errore: {str(e)}")



# Esegui l'applicazione
if __name__ == "__main__":
    try:
        app = RouletteSimulator()
        app.run()
    except Exception as e:
        print(f"Impossibile avviare l'applicazione: {e}")
        import traceback
        traceback.print_exc()