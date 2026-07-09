"""
Inferência: carrega o modelo treinado e classifica batimentos individuais
de ECG, seja a partir de um registro do MIT-BIH ou de uma janela de sinal
já extraída.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import wfdb

from model import CNNLSTMClassifier

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_DESCRIPTIONS = {
    'N': 'Normal',
    'S': 'Supraventricular (ex: contração atrial prematura)',
    'V': 'Ventricular (ex: PVC - contração ventricular prematura)',
    'F': 'Fusão (mistura de batimento normal e ventricular)',
    'Q': 'Desconhecido/não classificável (ex: marcapasso)',
}


def load_model(checkpoint_path, device=None):
    """
    Carrega o modelo treinado a partir do checkpoint salvo.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNLSTMClassifier(n_classes=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def predict_window(model, window, device):
    """
    Classifica uma única janela de sinal (array 1D, ex. 360 amostras).

    Returns:
        pred_class: nome da classe prevista (ex. 'N')
        probs: dict {classe: probabilidade}
    """
    window = np.asarray(window, dtype=np.float32)

    # normalização igual à usada no treino (z-score por janela)
    mean, std = window.mean(), window.std()
    if std > 0:
        window = (window - mean) / std

    # formato esperado: (batch=1, seq_len, 1)
    x = torch.from_numpy(window).reshape(1, -1, 1).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return pred_class, probs_dict


def predict_from_record(model, device, record_path, beat_index=0, window_size=180):
    """
    Extrai um batimento específico de um registro do MIT-BIH (pelo índice
    da anotação, não pela amostra) e classifica.

    Args:
        record_path: caminho sem extensão, ex 'data/mitdb/100'
        beat_index: qual batimento anotado usar (0 = primeiro do registro)
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]
    peak = annotation.sample[beat_index]
    true_symbol = annotation.symbol[beat_index]

    start, end = peak - window_size, peak + window_size
    if start < 0 or end > len(signal):
        raise ValueError('Batimento muito próximo da borda do sinal, escolha outro índice.')

    window = signal[start:end]
    pred_class, probs = predict_window(model, window, device)

    return window, pred_class, probs, true_symbol


def plot_prediction(window, pred_class, probs, true_symbol=None):
    """
    Plota a forma de onda do batimento junto com a previsão do modelo.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={'width_ratios': [2, 1]})

    # ---- forma de onda ----
    ax1.plot(window, color='steelblue')
    title = f'Previsto: {pred_class} ({CLASS_DESCRIPTIONS[pred_class]})'
    if true_symbol is not None:
        title += f'\nAnotação original: {true_symbol}'
    ax1.set_title(title, fontsize=11)
    ax1.set_xlabel('Amostras')
    ax1.set_ylabel('Amplitude (normalizada)')
    ax1.grid(alpha=0.3)

    # ---- barras de probabilidade ----
    classes = list(probs.keys())
    values = list(probs.values())
    colors = ['crimson' if c == pred_class else 'lightgray' for c in classes]

    ax2.barh(classes, values, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probabilidade')
    ax2.set_title('Confiança do modelo')
    for i, v in enumerate(values):
        ax2.text(v + 0.02, i, f'{v:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Exemplo de uso
    model, device = load_model('models/best_model.pt')

    window, pred_class, probs, true_symbol = predict_from_record(
        model, device, 'data/mitdb/100', beat_index=10
    )

    print(f'Previsão: {pred_class} | Anotação real: {true_symbol}')
    print(f'Probabilidades: {probs}')

    plot_prediction(window, pred_class, probs, true_symbol)