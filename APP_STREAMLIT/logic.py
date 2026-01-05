# Modulo per i dati di input - Simulazione usura + ordinamento:

import numpy as np

def get_sample_usura():
    """
    Genera casualmente 4 valori interi di usura tra 0 e 3.
    (0 = nuova, 1 = buona, 2 = discreta, 3 = pessima)
    """
    return np.random.uniform(0, 3, size=4).round(2).tolist()

#def save_usura_to_file(usure, path="data/sample_usura.json"):
    """
    Salva i valori di usura in un file JSON.
    Crea automaticamente la cartella se non esiste.
    
    Args:
        usure (list[int]): Valori di usura (lista di 4 interi tra 0 e 3)
        path (str): Percorso del file dove salvare i dati
    """


#def load_usura_from_file(path="data/sample_usura.json"):
    """
    Carica eventuali valori salvati su file JSON.
    """
usure = (get_sample_usura())

def order_wheels(usure):
    """
    Ordina le ruote in base al livello di usura crescente.
    Restituisce gli indici ordinati e i valori riordinati.
    
    Esempio:
        input  -> [2, 0, 3, 1]
        output -> ([1, 3, 0, 2], [0, 1, 2, 3])
    """
    usure = np.array(usure)
    indices = np.argsort(usure)
    ordered = usure[indices].tolist()
    return indices, ordered

print(get_sample_usura())
print(usure)