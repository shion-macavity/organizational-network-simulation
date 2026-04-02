
"""
セマンティック類似度評価（SSR）パッケージ

LLM（大規模言語モデル）のテキスト回答を、参照文とのセマンティック類似度（意味の近さ）を
用いて、リッカート尺度（アンケートなどで使われる「そう思う」「そう思わない」などの段階評価）の
確率分布に変換するためのパッケージです。

このパッケージは、以下の論文で説明されているSSRの手法を実装しています：
"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"
"""

from beartype.claw import beartype_this_package

from .compute import response_embeddings_to_pmf, scale_pmf
from .response_rater import ResponseRater

__version__ = "1.0.0"
__author__ = "Ben F. Maier, Ulf Aslak"

__all__ = [
    "ResponseRater",
    "response_embeddings_to_pmf",
    "scale_pmf",
]

beartype_this_package()
