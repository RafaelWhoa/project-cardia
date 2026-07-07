"""
Pré-processamento do MIT-BIH Arrhythmia Database.
Segmenta o sinal em janelas centradas em cada batimento (pico R)
e mapeia os símbolos de anotação para as 5 classes AAMI.
"""

import numpy as np
import wfdb
import os

data_version = '1.0.0'
data_folder = f'data/physionet.org/files/mitdb/{data_version}'

# Mapeamento AAMI: agrupa os ~15 símbolos originais em 5 classes
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',        # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',                   # Supraventricular
    'V': 'V', 'E': 'V',                                        # Ventricular
    'F': 'F',                                                   # Fusão
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                               # Desconhecido/paced
}

CLASS_TO_IDX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

# Lista padrão de registros do MIT-BIH (48 registros)
RECORD_IDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


def segment_record(record_path, window_size=180):
    """
    Lê um registro e retorna janelas de sinal centradas em cada
    batimento anotado, junto com o label correspondente.

    Args:
        record_path: caminho sem extensão, ex 'data/raw/mitdb/100'
        window_size: metade da janela em amostras (total = 2*window_size)

    Returns:
        X: array (n_batimentos, 2*window_size)
        y: array (n_batimentos,) com índices de classe (0-4)
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]  # usa o primeiro canal (geralmente MLII)
    signal_len = len(signal)

    X, y = [], []

    for peak, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in AAMI_MAP:
            continue  # ignora símbolos que não são batimentos (ex. marcações de ruído)

        start = peak - window_size
        end = peak + window_size

        if start < 0 or end > signal_len:
            continue  # descarta batimentos incompletos nas bordas do sinal

        window = signal[start:end]
        label = CLASS_TO_IDX[AAMI_MAP[symbol]]

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)


def build_dataset(data_dir, record_ids):
    """
    Monta o dataset completo a partir de uma lista de registros.

    Args:
        data_dir: pasta com os arquivos .dat/.hea/.atr
        record_ids: lista de nomes de registro, ex ['100', '101', ...]

    Returns:
        X: array (n_total, 2*window_size)
        y: array (n_total,)
    """
    X_list, y_list = [], []

    for rid in record_ids:
        path = os.path.join(data_dir, rid)
        try:
            X, y = segment_record(path)
            X_list.append(X)
            y_list.append(y)
            print(f'Registro {rid}: {len(y)} batimentos')
        except Exception as e:
            print(f'Erro no registro {rid}: {e}')

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


if __name__ == '__main__':
    X, y = build_dataset(data_folder, RECORD_IDS)

    print(f'\nTotal: {X.shape[0]} batimentos, janela de {X.shape[1]} amostras')
    print('Distribuição de classes:')
    for cls, idx in CLASS_TO_IDX.items():
        count = np.sum(y == idx)
        print(f'  {cls}: {count} ({100*count/len(y):.2f}%)')

    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', y)
    print('\nSalvo em data/processed/')