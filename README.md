# CARDIA: Detecção de Arritmias em ECG com Redes Neurais Recorrentes

Projeto de deep learning para classificar batimentos cardíacos em sinais de ECG usando uma arquitetura híbrida **CNN + BiLSTM** em PyTorch, treinada na base **MIT-BIH Arrhythmia Database** (PhysioNet).

> **Aviso:** projeto educacional. Não é um dispositivo médico e não deve ser usado para diagnóstico clínico.

---

## Visão geral

Um ECG contínuo pode ter milhares de batimentos. Este projeto classifica **cada batimento individualmente** em uma das 5 classes do padrão AAMI. Qualquer batimento classificado como S, V ou F é sinalizado como um possível evento de arritmia, indicando onde um profissional deveria olhar com mais atenção.

| Classe | Significado | Arritmia? |
|---|---|---|
| **N** | Normal (inclui bloqueios de ramo L e R) | Não |
| **S** | Supraventricular (ex.: contração atrial prematura) | Sim |
| **V** | Ventricular (ex.: PVC, contração ventricular prematura) | Sim |
| **F** | Fusão (mistura de batimento normal e ventricular) | Sim |
| **Q** | Desconhecido / não classificável (ex.: marcapasso) | Ambíguo |

## Dataset

- **Fonte:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (48 registros, 360 Hz)
- **Janela:** 360 amostras (1 segundo) centradas em cada pico R, usando o primeiro canal de cada registro
- **Total:** 109.438 batimentos

| Classe | Batimentos | % |
|---|---|---|
| N | 90.582 | 82,77% |
| S | 2.781 | 2,54% |
| V | 7.235 | 6,61% |
| F | 802 | 0,73% |
| Q | 8.038 | 7,34% |

O forte desbalanceamento (N domina) é o principal desafio do projeto e é tratado com pesos de classe na função de perda.

## Pipeline

1. **Segmentação** (`preprocessing.py`): extrai uma janela por batimento anotado e mapeia os símbolos originais do MIT-BIH para as 5 classes AAMI. Anotações de ritmo/evento (ex.: `!`, `+`) são descartadas.
2. **Dataset** (`dataset.py`): normalização z-score por batimento, split estratificado 70/15/15 (treino/validação/teste) e cálculo de pesos de classe inversamente proporcionais à frequência.
3. **Modelo** (`model.py`): CNN para extrair features morfológicas locais, seguida de BiLSTM para modelar a dependência temporal.
4. **Treino** (`train.py`): Adam, `CrossEntropyLoss` com pesos de classe, `ReduceLROnPlateau` e early stopping com base no F1-macro de validação.
5. **Avaliação e visualização** (`visualize.py`): matriz de confusão em heatmap e curvas de treino.
6. **Inferência** (`inference.py`): classifica batimentos individuais mostrando a distribuição de probabilidade entre as classes.

## Arquitetura

```
Entrada (360, 1)
  └─ Conv1d(1→32,  k=7) + BatchNorm + ReLU + MaxPool(2)
  └─ Conv1d(32→64, k=5) + BatchNorm + ReLU + MaxPool(2)
  └─ Conv1d(64→128, k=3) + BatchNorm + ReLU + MaxPool(2)
  └─ BiLSTM (2 camadas, 64 unidades por direção, dropout 0.3)
  └─ Dropout(0.3) + Linear(128 → 5)
Saída: logits para N, S, V, F, Q
```

**Hiperparâmetros:** batch size 128, learning rate 1e-3, até 30 épocas, early stopping com paciência 5.

## Resultados

Conjunto de teste (16.416 batimentos):

| Classe | Precision | Recall | F1-score | Suporte |
|---|---|---|---|---|
| N | 0,997 | 0,980 | 0,988 | 13.588 |
| S | 0,722 | 0,952 | 0,821 | 417 |
| V | 0,939 | 0,973 | 0,956 | 1.085 |
| F | 0,633 | 0,926 | 0,752 | 121 |
| Q | 0,987 | 0,996 | 0,991 | 1.205 |

**Acurácia:** 0,979 | **F1-macro:** 0,902

Como o treino não usa seed fixa, os números variam um pouco entre execuções (em outra rodada, a acurácia ficou em 0,987 e o F1-macro em 0,927). A acurácia sozinha engana neste problema, então a métrica de decisão do treino é o F1-macro.

Observações:
- O modelo tem recall alto em todas as classes de arritmia, o que é desejável para triagem, mas a precisão de **S** e **F** é mais baixa, ou seja, há falsos positivos.
- Os erros mais comuns são S → N e F → N / F → V, coerentes com a ambiguidade morfológica dessas classes.
- A loss de validação começa a subir após ~10 épocas enquanto o F1 continua oscilando para cima, indicando overfitting leve.

## Estrutura do repositório

```
project-cardia/
├── src/
│   ├── preprocessing.py   # segmentação e mapeamento AAMI
│   ├── dataset.py         # Dataset, split e class weights
│   ├── model.py           # CNN + BiLSTM
│   ├── train.py           # treino e avaliação
│   ├── visualize.py       # matriz de confusão e curvas de treino
│   └── inference.py       # classificação de batimentos individuais
├── data/                  # (ignorado pelo Git) base MIT-BIH e dados processados
├── models/                # (ignorado pelo Git) checkpoints
└── README.md
```

## Como executar

O projeto foi desenvolvido no **Google Colab** com GPU (T4).

```python
# 1. Dependências
!pip install wfdb -q

# 2. Dados
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/mitdb')

# 3. Imports
import sys
sys.path.append('src')
from preprocessing import build_dataset, RECORD_IDS
from dataset import build_dataloaders
from model import CNNLSTMClassifier
from train import train_model, evaluate_model
from visualize import plot_confusion_matrix, plot_training_history

# 4. Pré-processamento
X, y = build_dataset('data/mitdb', RECORD_IDS)

# 5. Treino
train_loader, val_loader, test_loader, class_weights = build_dataloaders(X, y, batch_size=128)
model = CNNLSTMClassifier(n_classes=5)
model, history = train_model(
    model, train_loader, val_loader, class_weights,
    n_epochs=30, patience=5,
    checkpoint_path='models/best_model.pt'
)

# 6. Avaliação
labels, preds = evaluate_model(model, test_loader)
plot_confusion_matrix(labels, preds)
plot_training_history(history)
```

Inferência em um batimento:

```python
from inference import load_model, find_beats_by_symbol, predict_from_record, plot_prediction

model, device = load_model('models/best_model.pt')
indices = find_beats_by_symbol('data/mitdb/119', symbol='V')
window, pred, probs, true_symbol = predict_from_record(model, device, 'data/mitdb/119', beat_index=indices[0])
plot_prediction(window, pred, probs, true_symbol)
```

## Limitações conhecidas

- **Split por batimento, não por paciente:** batimentos do mesmo paciente aparecem em treino, validação e teste. Isso tende a inflar as métricas em relação a um cenário real com pacientes nunca vistos. Um split por paciente (ou o esquema inter-paciente DS1/DS2 da literatura) é o próximo passo para uma avaliação mais honesta.
- **Derivação única:** é usado sempre o primeiro canal do registro, que pode ser uma derivação diferente (MLII, V1, V5...) conforme o paciente.
- **Poucos exemplos da classe F** (802 no total), o que limita a precisão nessa classe.
- **Sem seed fixa:** os resultados variam entre execuções.

## Próximos passos

- Split por paciente para avaliação sem vazamento
- Fixar seeds e reportar média e desvio de várias execuções
- Mais regularização (dropout, weight decay) e data augmentation
- Comparar com Focal Loss e com outras arquiteturas (GRU, Transformer)
- Usar os dois canais do ECG como entrada

## Tecnologias

Python · PyTorch · NumPy · scikit-learn · wfdb · Matplotlib · Google Colab

## Referências

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol Mag, 2001.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 2000.
- AAMI EC57: padrão para classificação de batimentos em ECG.
