# 🖐️ Detector de Dedos Levantados (OpenCV + MediaPipe)

Projeto em Python para detectar mãos e contar dedos levantados em tempo real usando a câmera do computador. Foco em arquitetura simples, limpa e extensível.

## 🚀 Funcionalidades
- Abre a câmera do computador
- Detecta até 2 mãos simultaneamente (MediaPipe)
- Desenha landmarks e conexões das mãos
- Conta dedos levantados por mão e total
- Exibe contagens na tela em tempo real

## 🧰 Tecnologias
- OpenCV (captura e exibição de vídeo)
- MediaPipe (detecção e rastreamento de mãos)
- NumPy (utilidades)

## 📦 Requisitos
- Python 3.9+
- Windows, macOS ou Linux

Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📁 Estrutura
```
src/
  app.py
  fingers/
    __init__.py
    config.py
    types.py
    camera.py
    utils.py
    hand_detector.py
    finger_counter.py
    drawer.py
requirements.txt
README.md
```

## ▶️ Executar
No diretório do projeto:
```bash
python -m src.app
```

- Pressione "q" para encerrar a aplicação.
- Por padrão usa a câmera 0. Ajuste em `src/fingers/config.py`.

## 🧪 Ajustes úteis
- `MAX_NUM_HANDS`: máximo de mãos a detectar (2)
- Confiabilidade de detecção e rastreamento em `config.py`

## 📜 Licença
Uso educacional e livre. Ajuste conforme sua necessidade.
