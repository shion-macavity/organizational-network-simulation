"""
computeモジュール（計算機能のまとまり）のテストです。
数学的な特性と振る舞い（behavior）に焦点を当てています。
"""

import numpy as np
import pytest
from semantic_similarity_rating.compute import (
    scale_pmf,
    response_embeddings_to_pmf,
)

# テスト用の定数
EMBEDDING_DIM = 384  # 現実的な埋め込みの次元数
LIKERT_SIZE = 5


def assert_valid_pmf(pmf_array):
    """配列に有効な確率質量関数（PMF）が含まれていることを表明（assert）します。"""
    if pmf_array.ndim == 1:
        pmf_array = pmf_array.reshape(1, -1)

    for i, pmf in enumerate(pmf_array):
        assert np.isclose(pmf.sum(), 1.0, atol=1e-10), f"PMF {i} が1に合計されません"
        assert np.all(pmf >= 0), f"PMF {i} に負の確率が含まれています"


def create_test_embeddings(n_responses=3, n_dimensions=EMBEDDING_DIM, seed=42):
    """決定論的（何度やっても同じ結果になる）なテスト用の埋め込みを作成します。"""
    np.random.seed(seed)
    return np.random.randn(n_responses, n_dimensions)


def create_likert_embeddings(n_dimensions=EMBEDDING_DIM, n_points=LIKERT_SIZE, seed=42):
    """決定論的なリッカート尺度の参照埋め込みを作成します。"""
    np.random.seed(seed + 1)  # 多様性のために異なるシード（乱数の種）を使用
    return np.random.randn(n_dimensions, n_points)

class TestScalePMF:
    """確率分布の温度スケーリング（temperature scaling）をテストします。"""

    def test_temperature_identity(self):
        """温度（temperature）が1.0の場合、PMF（確率質量関数）は変更されないはずです。"""
        pmf = np.array([0.1, 0.2, 0.3, 0.4])
        scaled = scale_pmf(pmf, temperature=1.0)
        assert np.allclose(scaled, pmf)

    def test_temperature_extremes(self):
        """極端な温度での振る舞いをテストします。"""
        pmf = np.array([0.1, 0.2, 0.3, 0.4])

        # Near-zero temperature should create one-hot distribution
        # ゼロに近い温度は、one-hot分布（一つだけが1で他が0の分布）を作成するはずです
        sharp = scale_pmf(pmf, temperature=0.01)
        assert_valid_pmf(sharp)
        assert sharp[np.argmax(pmf)] > 0.99  # Highest prob element dominates # 最も確率の高い要素が支配的になる

        # High temperature should be more uniform
        # 高い温度は、より均一（uniform）になるはずです
        smooth = scale_pmf(pmf, temperature=10.0)
        assert_valid_pmf(smooth)

        # Higher temperature should increase entropy
        # 温度が高いほどエントロピー（不確実性）は増加するはずです
        sharp_entropy = -np.sum(sharp * np.log(sharp + 1e-12))
        smooth_entropy = -np.sum(smooth * np.log(smooth + 1e-12))
        assert smooth_entropy > sharp_entropy

    def test_temperature_capping(self):
        """温度はmax_temp（最大温度）で上限が設定されるはずです。"""
        pmf = np.array([0.1, 0.6, 0.3])

        capped = scale_pmf(pmf, temperature=100.0, max_temp=5.0)
        expected = scale_pmf(pmf, temperature=5.0)
        assert np.allclose(capped, expected)


class TestEmbeddingsToPMF:
    """Test core embedding-to-PMF conversion function."""
    """中核となる「埋め込みからPMFへ」の変換関数をテストします。"""
    

    def test_basic_functionality(self):
        """Should convert embeddings to valid PMFs with correct shape."""
        """埋め込みを、正しい形状（shape）の有効なPMF（確率質量関数）に変換するはずです。"""
        response_embs = create_test_embeddings(n_responses=3)
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (3, LIKERT_SIZE)
        assert_valid_pmf(result)

    def test_deterministic_behavior(self):
        """Identical inputs should produce identical outputs."""
        # Create identical response embeddings
        """同一の入力は、同一の出力を生成するはずです。"""
        # 同一の回答埋め込みを作成
        response_embs = np.tile(create_test_embeddings(n_responses=1), (2, 1))
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert np.allclose(result[0], result[1])
        assert_valid_pmf(result)

    def test_epsilon_regularization(self):
        """Epsilon should prevent zero probabilities and affect distribution."""
        """イプシロン（epsilon）は、確率がゼロになるのを防ぎ、分布に影響を与えるはずです。"""
        response_embs = create_test_embeddings(n_responses=2)
        likert_embs = create_likert_embeddings()

        no_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.0)
        with_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.1)

        assert_valid_pmf(no_eps)
        assert_valid_pmf(with_eps)
        assert not np.allclose(no_eps, with_eps)

        # With epsilon, all probabilities should be positive
        # イプシロンがある場合、全ての確率は正（positive）になるはずです
        assert np.all(with_eps > 0)

    def test_epsilon_effect_on_uniformity(self):
        """Higher epsilon should generally create more uniform distributions."""
        """イプシロンが大きいほど、一般的により均一な（uniform）分布を作成するはずです。"""
        response_embs = create_test_embeddings(n_responses=1)
        likert_embs = create_likert_embeddings()

        low_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.001)[
            0
        ]
        high_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.1)[
            0
        ]

        # Higher epsilon should increase entropy
        # イプシロンが大きいほどエントロピー（不確実性）は増加するはずです
        low_entropy = -np.sum(low_eps * np.log(low_eps + 1e-12))
        high_entropy = -np.sum(high_eps * np.log(high_eps + 1e-12))

        assert high_entropy >= low_entropy  # Should be more uniform # より均一になるはず

    def test_empty_input_handling(self):
        """Should handle empty response arrays gracefully."""
        """空の回答配列を適切に（gracefully）処理するはずです。"""
        empty_responses = np.empty((0, EMBEDDING_DIM))
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(empty_responses, likert_embs)

        assert result.shape == (0, LIKERT_SIZE)
        assert isinstance(result, np.ndarray)

    def test_similarity_ranking_preserved(self):
        """PMF should reflect embedding similarity ranking."""
        # Create response that's most similar to first Likert point
        """PMF（確率質量関数）は、埋め込みの類似度の順位（ranking）を反映するはずです。"""
        # 最初のリッカート尺度ポイントに最も類似する回答を作成
        likert_embs = create_likert_embeddings()
        response_embs = likert_embs[:, 0:1].T  # Transpose to make it (1, dim) # 転置（Transpose）して (1, dim) の形状にする

        result = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.01)

        # First Likert point should have highest probability 
        # 最初のリッカート尺度ポイントが最も高い確率を持つはずです
        assert np.argmax(result[0]) == 0


class TestEdgeCases:
    """Test edge cases and robustness."""
    """エッジケース（極端な事例）と堅牢性（robustness）をテストします。"""

    def test_single_response_realistic_dimension(self):
        """Should work with single response and realistic embeddings."""
        """単一の回答と現実的な埋め込みで動作するはずです。"""
        response_embs = np.array([[1.0, 0.5, -0.2]])  # Non-zero, realistic # ゼロではなく、現実的な値
        likert_embs = np.array(
            [
                [1.0, 0.5, 0.0, -0.5, -1.0],
                [0.8, 0.2, 0.1, -0.3, -0.8],
                [0.6, 0.1, 0.0, -0.1, -0.6],
            ]
        )

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (1, 5)
        assert_valid_pmf(result)

    def test_small_but_nonzero_embeddings(self):
        """Should handle small but non-zero embeddings without errors."""
        """小さいがゼロではない埋め込みを、エラーなしで処理するはずです。"""
        response_embs = np.full((2, EMBEDDING_DIM), 1e-6)  # Small but not zero # 小さいがゼロではない
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (2, LIKERT_SIZE)
        assert_valid_pmf(result)
        # Both responses should be identical since they're the same
        # 両方の回答は同じであるため、同一になるはずです
        assert np.allclose(result[0], result[1])

    def test_extreme_similarity_values(self):
        """Should handle very high and very low similarity values."""
        # Create response that's very similar to one Likert point
        """非常に高い、または非常に低い類似度の値を処理するはずです。"""
        # 一つのリッカート尺度ポイントに非常に類似した回答を作成
        likert_embs = create_likert_embeddings()
        response_embs = likert_embs[:, 2:3].T * 1000  # Scale up for extreme similarity # 極端な類似度のためにスケールアップ（拡大）する

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert_valid_pmf(result)
        # Should strongly prefer the similar poin
        # 類似したポイントを強く選好するはずです
        assert result[0, 2] > 0.8  # High probability for the similar point # 類似したポイントに対する高い確率


class TestNumericalStability:
    """Test numerical stability and precision."""
    """数値的安定性（numerical stability）と精度（precision）をテストします。"""

    def test_large_embeddings(self):
        """Should handle large embedding values without numerical issues."""
        """大きな値の埋め込みを、数値的な問題なしに処理するはずです。"""
        response_embs = create_test_embeddings() * 1000
        likert_embs = create_likert_embeddings() * 1000

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert_valid_pmf(result)
        assert np.all(np.isfinite(result))

    def test_very_small_epsilon(self):
        """Should handle very small epsilon values."""
        """非常に小さいイプシロン（epsilon）の値を処理するはずです。"""
        response_embs = create_test_embeddings(n_responses=1)
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=1e-10)

        assert_valid_pmf(result)
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
