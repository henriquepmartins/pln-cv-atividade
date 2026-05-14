"""
Análise de Sentimentos com Transformers (XLM-RoBERTa multilíngue)

Técnica: Análise de Sentimentos (PLN)
Biblioteca: Hugging Face Transformers + PyTorch
Modelo: cardiffnlp/twitter-xlm-roberta-base-sentiment
        (XLM-RoBERTa multilíngue treinado em 8 idiomas, incluindo português)

Pipeline conceitual:
    texto → tokenização → embeddings BERT → cabeça de classificação
          → softmax → [positive | neutral | negative]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FRASES = [
    "Amei o atendimento, foi simplesmente impecável do começo ao fim!",
    "Esse filme me fez chorar de emoção, vou lembrar dele para sempre.",
    "A entrega chegou rapidíssima e o produto é melhor do que eu esperava.",
    "O restaurante é confortável e o cardápio é razoavelmente variado.",
    "O voo decolou no horário previsto.",
    "Funciona como descrito na embalagem, nada de extraordinário.",
    "O atendimento foi tão lento que eu envelheci esperando uma resposta.",
    "Produto chegou quebrado, mal embalado e ainda veio o item errado.",
    "Pior experiência da minha vida, jamais volto a essa loja.",
    "Achei o app travado, confuso e cheio de propagandas invasivas.",
]

CORES = {
    "positive": "#16a34a",
    "neutral": "#737373",
    "negative": "#dc2626",
}


def _cor(label: str) -> str:
    return CORES[label.lower()]


def main() -> None:
    print("Carregando modelo (primeira execução baixa ~1.1GB)…")
    classificador = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        truncation=True,
    )

    print(f"Classificando {len(FRASES)} frases…\n")
    resultados = classificador(FRASES)

    df = pd.DataFrame(
        {
            "frase": FRASES,
            "sentimento": [r["label"] for r in resultados],
            "confiança": [round(r["score"], 4) for r in resultados],
        }
    )
    print(df.to_string(index=False))

    csv_path = OUTPUT_DIR / "resultados.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nCSV salvo em {csv_path}")

    plotar(df)


def plotar(df: pd.DataFrame) -> None:
    df = df.iloc[::-1].reset_index(drop=True)
    cores = [_cor(s) for s in df["sentimento"]]
    rotulos = [
        f if len(f) <= 60 else f[:57] + "…" for f in df["frase"]
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(rotulos, df["confiança"], color=cores, edgecolor="white", linewidth=1.5)

    for i, (score, sent) in enumerate(zip(df["confiança"], df["sentimento"])):
        ax.text(
            score + 0.01,
            i,
            f"{score:.2%} · {sent}",
            va="center",
            fontsize=9,
            color="#262626",
        )

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Confiança do modelo")
    ax.set_title(
        "Análise de Sentimentos com XLM-RoBERTa multilíngue\n"
        "Modelo: cardiffnlp/twitter-xlm-roberta-base-sentiment",
        fontsize=12,
        pad=15,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    legenda = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=l.capitalize())
        for l, c in CORES.items()
    ]
    ax.legend(
        handles=legenda,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    png_path = OUTPUT_DIR / "resultados.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo em {png_path}")


if __name__ == "__main__":
    main()
