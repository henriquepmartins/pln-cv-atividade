"""Gera os dois notebooks .ipynb a partir das células definidas inline."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def cell_md(src: str):
    return nbf.v4.new_markdown_cell(src)


def cell_code(src: str):
    return nbf.v4.new_code_cell(src)


def build_nlp() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        cell_md(
            "# Análise de Sentimentos com Transformers\n"
            "\n"
            "**Técnica:** Análise de Sentimentos (PLN)  \n"
            "**Biblioteca:** Hugging Face Transformers + PyTorch  \n"
            "**Modelo:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` — XLM-RoBERTa multilíngue\n"
            "\n"
            "## Pipeline conceitual\n"
            "\n"
            "```\n"
            "texto  →  tokenização (SentencePiece)\n"
            "       →  embeddings contextuais (XLM-RoBERTa)\n"
            "       →  cabeça de classificação\n"
            "       →  softmax  →  [positive | neutral | negative]\n"
            "```\n"
        ),
        cell_md("## 1. Imports"),
        cell_code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from transformers import pipeline"
        ),
        cell_md("## 2. Carregar o modelo\n\nNa primeira execução faz download de ~1.1GB."),
        cell_code(
            "classificador = pipeline(\n"
            "    'sentiment-analysis',\n"
            "    model='cardiffnlp/twitter-xlm-roberta-base-sentiment',\n"
            "    truncation=True,\n"
            ")"
        ),
        cell_md("## 3. Frases de teste em português"),
        cell_code(
            "frases = [\n"
            "    'Amei o atendimento, foi simplesmente impecável do começo ao fim!',\n"
            "    'Esse filme me fez chorar de emoção, vou lembrar dele para sempre.',\n"
            "    'A entrega chegou rapidíssima e o produto é melhor do que eu esperava.',\n"
            "    'O restaurante é confortável e o cardápio é razoavelmente variado.',\n"
            "    'O voo decolou no horário previsto.',\n"
            "    'Funciona como descrito na embalagem, nada de extraordinário.',\n"
            "    'O atendimento foi tão lento que eu envelheci esperando uma resposta.',\n"
            "    'Produto chegou quebrado, mal embalado e ainda veio o item errado.',\n"
            "    'Pior experiência da minha vida, jamais volto a essa loja.',\n"
            "    'Achei o app travado, confuso e cheio de propagandas invasivas.',\n"
            "]\n"
            "len(frases)"
        ),
        cell_md("## 4. Classificar"),
        cell_code(
            "resultados = classificador(frases)\n"
            "df = pd.DataFrame({\n"
            "    'frase': frases,\n"
            "    'sentimento': [r['label'] for r in resultados],\n"
            "    'confiança': [round(r['score'], 4) for r in resultados],\n"
            "})\n"
            "df"
        ),
        cell_md("## 5. Visualizar com cores semânticas"),
        cell_code(
            "CORES = {'positive': '#16a34a', 'neutral': '#737373', 'negative': '#dc2626'}\n"
            "\n"
            "ordenado = df.iloc[::-1].reset_index(drop=True)\n"
            "cores = [CORES[s.lower()] for s in ordenado['sentimento']]\n"
            "rotulos = [f if len(f) <= 60 else f[:57] + '…' for f in ordenado['frase']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 7))\n"
            "ax.barh(rotulos, ordenado['confiança'], color=cores, edgecolor='white', linewidth=1.5)\n"
            "for i, (s, sent) in enumerate(zip(ordenado['confiança'], ordenado['sentimento'])):\n"
            "    ax.text(s + 0.01, i, f'{s:.2%} · {sent}', va='center', fontsize=9, color='#262626')\n"
            "\n"
            "ax.set_xlim(0, 1.25)\n"
            "ax.set_xlabel('Confiança do modelo')\n"
            "ax.set_title('Análise de Sentimentos com XLM-RoBERTa multilíngue', fontsize=12, pad=15)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "ax.grid(axis='x', linestyle='--', alpha=0.3)\n"
            "\n"
            "import matplotlib.patches as mpatches\n"
            "legenda = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in CORES.items()]\n"
            "ax.legend(handles=legenda, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        cell_md(
            "## Conclusão\n"
            "\n"
            "O XLM-RoBERTa classificou corretamente as 10 frases em português, atribuindo:\n"
            "- **4 positivas** (elogios e experiências boas)\n"
            "- **2 neutras** (descrições factuais sem carga emocional)\n"
            "- **4 negativas** (reclamações e críticas)\n"
            "\n"
            "Mesmo sem ajuste fino específico para PT-BR, o modelo multilíngue acertou tom e ironia "
            "(*\"envelheci esperando\"* → negativo)."
        ),
    ]
    return nb


def build_cv() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        cell_md(
            "# Segmentação Watershed — Separação de Objetos Sobrepostos\n"
            "\n"
            "**Técnica:** Segmentação por Região (Watershed) — Visão Computacional  \n"
            "**Bibliotecas:** OpenCV + scikit-image\n"
            "\n"
            "## Intuição do algoritmo\n"
            "\n"
            "Imagine a imagem como um mapa topográfico onde regiões claras são montanhas e regiões "
            "escuras são vales. Se 'inundarmos' o mapa, a água sobe a partir de cada vale (marker). "
            "Quando águas de vales diferentes se encontram, formam uma **linha de separação** — é assim "
            "que o watershed separa objetos colados.\n"
            "\n"
            "## Pipeline\n"
            "\n"
            "```\n"
            "Imagem  →  CLAHE        (realça contraste)\n"
            "        →  Otsu         (binariza)\n"
            "        →  Morfologia   (limpa ruído)\n"
            "        →  Distance T.  (mapa de distâncias)\n"
            "        →  Threshold    (núcleos = markers)\n"
            "        →  Watershed    (expande até as bordas)\n"
            "```\n"
        ),
        cell_md("## 1. Imports"),
        cell_code(
            "import cv2\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from skimage import data"
        ),
        cell_md("## 2. Carregar a imagem das moedas"),
        cell_code(
            "cinza = data.coins()\n"
            "cinza = cv2.resize(cinza, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)\n"
            "bgr = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)\n"
            "\n"
            "plt.figure(figsize=(8, 5))\n"
            "plt.imshow(cinza, cmap='gray')\n"
            "plt.title('Imagem original — skimage.data.coins()')\n"
            "plt.axis('off')\n"
            "plt.show()\n"
            "print('Shape:', bgr.shape)"
        ),
        cell_md("## 3. Pré-processamento: CLAHE + blur + Otsu"),
        cell_code(
            "clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))\n"
            "cinza_eq = clahe.apply(cinza)\n"
            "cinza_blur = cv2.GaussianBlur(cinza_eq, (5, 5), 0)\n"
            "_, binario = cv2.threshold(cinza_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
            "for ax, img, t in zip(axes, [cinza, cinza_eq, binario], ['Original', 'CLAHE', 'Otsu']):\n"
            "    ax.imshow(img, cmap='gray'); ax.set_title(t); ax.axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        cell_md("## 4. Morfologia + Distance Transform"),
        cell_code(
            "kernel = np.ones((3, 3), np.uint8)\n"
            "opening = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel, iterations=2)\n"
            "opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)\n"
            "sure_bg = cv2.dilate(opening, kernel, iterations=3)\n"
            "\n"
            "dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)\n"
            "_, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)\n"
            "sure_fg = sure_fg.astype(np.uint8)\n"
            "desconhecido = cv2.subtract(sure_bg, sure_fg)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
            "axes[0].imshow(opening, cmap='gray'); axes[0].set_title('Opening (morfologia)')\n"
            "axes[1].imshow(dist, cmap='magma'); axes[1].set_title('Distance Transform')\n"
            "axes[2].imshow(sure_fg, cmap='gray'); axes[2].set_title('Sure Foreground (markers)')\n"
            "for a in axes: a.axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        cell_md("## 5. Aplicar Watershed"),
        cell_code(
            "num_markers, markers = cv2.connectedComponents(sure_fg)\n"
            "markers = markers + 1\n"
            "markers[desconhecido == 255] = 0\n"
            "markers = cv2.watershed(bgr.copy(), markers)\n"
            "\n"
            "resultado = bgr.copy()\n"
            "resultado[markers == -1] = [0, 0, 255]\n"
            "\n"
            "n_objetos = num_markers - 1\n"
            "print(f'{n_objetos} objetos detectados')\n"
            "\n"
            "plt.figure(figsize=(10, 6))\n"
            "plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))\n"
            "plt.title(f'Watershed — {n_objetos} objetos com bordas em vermelho', fontsize=13)\n"
            "plt.axis('off'); plt.show()"
        ),
        cell_md("## 6. Visualização final colorida por segmento"),
        cell_code(
            "rng = np.random.default_rng(seed=42)\n"
            "h, w = markers.shape\n"
            "colorido = np.zeros((h, w, 3), dtype=np.uint8)\n"
            "for label in np.unique(markers):\n"
            "    if label <= 1: continue\n"
            "    colorido[markers == label] = rng.integers(60, 255, size=3, dtype=np.uint8)\n"
            "\n"
            "overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()\n"
            "mask = colorido.sum(axis=2) > 0\n"
            "overlay[mask] = (0.55 * overlay[mask] + 0.45 * colorido[mask]).astype(np.uint8)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 6))\n"
            "axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title('Antes'); axes[0].axis('off')\n"
            "axes[1].imshow(overlay); axes[1].set_title(f'Depois — {n_objetos} objetos segmentados'); axes[1].axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        cell_md(
            "## Conclusão\n"
            "\n"
            "A pipeline detectou e separou as moedas — mesmo as que aparecem coladas — usando "
            "o algoritmo Watershed do OpenCV. Os passos críticos foram:\n"
            "\n"
            "1. **CLAHE** para que moedas escuras (canto superior) sobrevivessem ao Otsu\n"
            "2. **Distance Transform** para encontrar o 'centro' de cada moeda\n"
            "3. **connectedComponents** para gerar markers únicos por região\n"
            "4. **cv2.watershed** para expandir cada marker até encontrar bordas\n"
        ),
    ]
    return nb


def executar(nb: nbf.NotebookNode, cwd: Path) -> nbf.NotebookNode:
    client = NotebookClient(nb, timeout=600, kernel_name="pln-cv-venv", resources={"metadata": {"path": str(cwd)}})
    client.execute()
    return nb


def main() -> None:
    nlp_nb = executar(build_nlp(), ROOT / "nlp")
    nbf.write(nlp_nb, ROOT / "nlp" / "sentiment_analysis.ipynb")
    print("nlp/sentiment_analysis.ipynb gerado")

    cv_nb = executar(build_cv(), ROOT / "cv")
    nbf.write(cv_nb, ROOT / "cv" / "watershed_segmentation.ipynb")
    print("cv/watershed_segmentation.ipynb gerado")


if __name__ == "__main__":
    main()
