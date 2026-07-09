"""
Visualizações para avaliação do modelo: matriz de confusão em heatmap
e curvas de treino.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names=('N', 'S', 'V', 'F', 'Q'),
                           normalize=True, save_path=None):
    """
    Plota a matriz de confusão como heatmap.

    Args:
        y_true, y_pred: listas/arrays com os labels verdadeiros e previstos
        class_names: nomes das classes na ordem dos índices (0,1,2,3,4)
        normalize: se True, mostra proporção por linha (recall visual);
                   se False, mostra contagem absoluta
        save_path: caminho para salvar a imagem (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_display = cm.astype('float')

    if normalize:
        cm_display = cm_display / cm_display.sum(axis=1, keepdims=True)
        fmt = '.2%'
        title = 'Matriz de Confusão (normalizada por classe real)'
    else:
        fmt = 'd'
        cm_display = cm
        title = 'Matriz de Confusão (contagem absoluta)'

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_display, cmap='Blues')

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Classe Prevista')
    ax.set_ylabel('Classe Real')
    ax.set_title(title)

    # anota cada célula com o valor
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_display[i, j]
            text = f'{value:.2%}' if normalize else f'{int(value)}'
            color = 'white' if value > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label='Proporção' if normalize else 'Contagem')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Salvo em {save_path}')

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plota as curvas de treino: loss (treino vs validação) e F1-macro de validação.

    Args:
        history: dict com chaves 'train_loss', 'val_loss', 'val_f1_macro',
                 cada uma uma lista com um valor por época (retornado por train_model)
        save_path: caminho para salvar a imagem (opcional)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- gráfico de loss ----
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Treino', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Validação', marker='o', markersize=3)
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.set_title('Loss por época')
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- gráfico de F1-macro ----
    ax = axes[1]
    ax.plot(epochs, history['val_f1_macro'], label='Validação', color='green', marker='o', markersize=3)
    best_epoch = int(np.argmax(history['val_f1_macro'])) + 1
    best_f1 = max(history['val_f1_macro'])
    ax.scatter([best_epoch], [best_f1], color='red', zorder=5,
               label=f'Melhor: época {best_epoch} ({best_f1:.4f})')
    ax.set_xlabel('Época')
    ax.set_ylabel('F1-macro')
    ax.set_title('F1-macro (validação) por época')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Salvo em {save_path}')

    plt.show()