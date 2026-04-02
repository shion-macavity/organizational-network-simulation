"""
確率質量関数（PMF: Probability Mass Functions）と埋め込み（embeddings: テキストなどの意味を数値ベクトルで表現したもの）を
計算・操作するためのユーティリティ（補助的な）関数群です。

このモジュール（機能のまとまり）は以下の機能を提供します：
- 異なる類似度指標（コサイン類似度、KS）間の変換
- 温度パラメータ（temperature parameters）を使用したPMFのスケーリング（調整）
- PMFの統計的モーメント（平均や分散など）の計算
- PMFスケーリングのための最適な温度パラメータの発見
- 回答の埋め込みからPMFへの変換

このモジュールは特に、リッカート尺度の回答とそれらの埋め込みを扱う際に有用であり、
根底にある確率分布を分析・変換するためのツールを提供します。
"""

import numpy as np


def scale_pmf(pmf, temperature, max_temp=np.inf):
    """
    温度スケーリング（temperature scaling）を用いてPMF（確率質量関数）を調整します。

    引数 (Parameters)
    ----------
    pmf : array_like (配列のような形式)
        入力する確率密度関数（ここではPMFを指す）
    temperature : float (浮動小数点数)
        スケーリング用の温度パラメータ（0からmax_tempまで）
    max_temp : float (浮動小数点数), optional (任意)
        温度の最大値。デフォルト（初期設定）はnp.inf（無限大）

    戻り値 (Returns)
    -------
    numpy.ndarray (Numpyライブラリの配列)
        スケーリングされ、全ての値の合計が1になるPMF

    注記 (Notes)
    -----
    - temperatureが0の場合、最も確率が高い位置が1で他が0のベクトル（one-hot vector）を返します
    - temperatureがmax_tempより大きい場合、max_tempをスケーリングに使用します
    - それ以外の場合は、指定されたtemperatureをスケーリングに使用します
    """
    if temperature == 0.0:
        if np.all(pmf == pmf[0]):
            return pmf
        else:
            new_pmf = np.zeros_like(pmf)
            new_pmf[np.argmax(pmf)] = 1.0
            return new_pmf
    elif temperature > max_temp:
        hist = pmf ** (1 / max_temp)
    else:
        hist = pmf ** (1 / temperature)
    return hist / hist.sum()


def response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon=0.0):

    """
    回答の埋め込み（response embeddings）とリッカート尺度の文の埋め込み（Likert sentence embeddings）をPMF（確率質量関数）に変換します。

    引数 (Parameters)
    ----------
    matrix_responses : array_like (配列のような形式)
        回答の埋め込みを行列（Matrix）にしたもの
    matrix_likert_sentences : array_like (配列のような形式)
        リッカート尺度の文の埋め込みを行列にしたもの
    epsilon : float (浮動小数点数), optional (任意)
        ゼロ除算（0で割ること）を防ぎ、スムージング（平滑化）を加えるための小さな正則化（regularization）パラメータ。
        デフォルト（初期設定）は0.0（正則化なし）。

    戻り値 (Returns)
    -------
    numpy.ndarray (Numpyライブラリの配列)
        回答の分布を表す確率密度関数（ここではPMFを指す）

    注記 (Notes)
    -----
    これはSSRの計算式を実装しています：
        p_{c,i}(r) = [γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r] /
                 [Σ_r γ(σ_{r,i}, t_c̃) - n_points * γ(σ_ℓ,i, t_c̃) + ε]
    ここで、γはコサイン類似度関数、δ_ℓ,rはクロネッカーのデルタ（特定の条件で1、それ以外で0になる関数）、
    n_pointsはリッカート尺度の段階の数です。
    """
    M_left = matrix_responses
    M_right = matrix_likert_sentences

    # Handle empty input case
    # 空の入力ケースを処理
    if M_left.shape[0] == 0:
        return np.empty((0, M_right.shape[1]))

    # Normalize the right matrix (Likert sentences)
    # 右側の行列（リッカート尺度の文）を正規化（normalize）する
    norm_right = np.linalg.norm(M_right, axis=0)
    M_right = M_right / norm_right[None, :]

    # Normalize the left matrix (responses)
    # 左側の行列（回答）を正規化する
    norm_left = np.linalg.norm(M_left, axis=1)
    M_left = M_left / norm_left[:, None]

    # Calculate cosine similarities: γ(σ_{r,i}, t_c̃)
    # コサイン類似度を計算する: γ(σ_{r,i}, t_c̃)
    cos = (1 + M_left.dot(M_right)) / 2

    # Find minimum similarity per row: γ(σ_ℓ,i, t_c̃)
    # 各行の最小類似度を見つける: γ(σ_ℓ,i, t_c̃)
    cos_min = cos.min(axis=1)[:, None]

    # Numerator: γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r
    # The ε δ_ℓ,r term adds epsilon only to exactly one minimum similarity position per row
    # 分子: γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r
    # ε δ_ℓ,r の項は、各行で最小類似度を持つ位置（一つだけ）にイプシロンを加える
    numerator = cos - cos_min
    if epsilon > 0:
        # Add epsilon to the first position that achieves minimum in each row (Kronecker delta effect)
        # 各行で最小値を達成する最初の位置にイプシロンを加える（クロネッカーのデルタの効果）
        min_indices = np.argmin(cos, axis=1)
        for i, min_idx in enumerate(min_indices):
            numerator[i, min_idx] += epsilon

    # Denominator: Σ_r γ(σ_{r,i}, t_c̃) - n_likert_points * γ(σ_ℓ,i, t_c̃) + ε
    # This is: sum of all similarities - n_likert_points * minimum similarity + epsilon
    # 分母: Σ_r γ(σ_{r,i}, t_c̃) - n_likert_points * γ(σ_ℓ,i, t_c̃) + ε
    # これは：全ての類似度の合計 - リッカート尺度の段階数 * 最小類似度 + イプシロン
    n_likert_points = cos.shape[1]
    denominator = cos.sum(axis=1)[:, None] - n_likert_points * cos_min + epsilon

    # Calculate final PMF
    # 最終的なPMFを計算する
    pmf = numerator / denominator

    return pmf
