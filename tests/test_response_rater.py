"""
ResponseRaterのテストです。
実装（implementation）の詳細ではなく、振る舞い（behavior）に焦点を当てています。
"""

import numpy as np
import polars as po
import pytest
from semantic_similarity_rating import ResponseRater

# テスト用の定数
EMBEDDING_DIM = 128
LIKERT_SCALE_SIZE = 5
SAMPLE_TEXTS = ["terrible", "poor", "neutral", "good", "excellent"]
TEST_RESPONSES = ["I love this product", "It's okay I guess", "Completely awful"]


def assert_valid_pmf(pmfs, expected_rows=None):
    """Assert that input is valid probability mass function(s)."""
    """入力が有効な確率質量関数（PMF）であることを表明（assert）します。"""
    if pmfs.ndim == 1:
        pmfs = pmfs.reshape(1, -1)

    if expected_rows is not None:
        assert pmfs.shape[0] == expected_rows, f"期待される行数は{expected_rows} です"

    assert pmfs.shape[1] == LIKERT_SCALE_SIZE, (
        f" {LIKERT_SCALE_SIZE} 個の列を持つはずです"
    )

    for i, row in enumerate(pmfs):
        assert np.isclose(row.sum(), 1.0, atol=1e-10), f"行 {i}の合計が1になりません"
        assert np.all(row >= 0), f"行 {i} に負の確率が含まれています"


def create_test_dataframe(include_embeddings=False, num_sets=1):
    """Create test DataFrame with realistic structure."""
    """現実的な構造を持つテスト用のデータフレーム（DataFrame）を作成します。"""
    np.random.seed(42)  # Deterministic for all tests　# 全てのテストで決定論的（結果が固定）になるように

    data = {
        "id": [],
        "int_response": [],
        "sentence": [],
    }

    if include_embeddings:
        data["embedding"] = []

    for set_id in range(1, num_sets + 1):
        data["id"].extend([f"set{set_id}"] * LIKERT_SCALE_SIZE)
        data["int_response"].extend(range(1, LIKERT_SCALE_SIZE + 1))
        data["sentence"].extend(SAMPLE_TEXTS)

        if include_embeddings:
            embeddings = [
                np.random.randn(EMBEDDING_DIM).tolist()
                for _ in range(LIKERT_SCALE_SIZE)
            ]
            data["embedding"].extend(embeddings)

    return po.DataFrame(data)


class TestResponseRaterCore:
    """Core functionality tests."""
    """中核となる機能のテストです。"""

    def test_text_mode_basic_functionality(self):
        """Text mode should produce valid PMFs for text inputs."""
        """テキストモードは、テキスト入力に対して有効なPMF（確率質量関数）を生成するはずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        pmfs = rater.get_response_pmfs("set1", TEST_RESPONSES)
        assert_valid_pmf(pmfs, expected_rows=len(TEST_RESPONSES))

    def test_embedding_mode_basic_functionality(self):
        """Embedding mode should produce valid PMFs for embedding inputs."""
        """埋め込みモードは、埋め込み入力に対して有効なPMF（確率質量関数）を生成するはずです。"""
        df = create_test_dataframe(include_embeddings=True)
        rater = ResponseRater(df)

        test_embeddings = np.random.randn(len(TEST_RESPONSES), EMBEDDING_DIM)
        pmfs = rater.get_response_pmfs("set1", test_embeddings)
        assert_valid_pmf(pmfs, expected_rows=len(TEST_RESPONSES))

    def test_survey_level_aggregation(self):
        """Survey PMF should aggregate individual responses correctly."""
        """調査（Survey）レベルのPMFは、個々の回答を正しく集約（aggregate）するはずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        individual_pmfs = rater.get_response_pmfs("set1", TEST_RESPONSES)
        survey_pmf = rater.get_survey_response_pmf(individual_pmfs)

        assert_valid_pmf(survey_pmf)

        # Convenience method should give same result
        # 便利な（ショートカット用の）メソッドも同じ結果を与えるはずです
        survey_pmf_conv = rater.get_survey_response_pmf_by_reference_set_id(
            "set1", TEST_RESPONSES
        )
        assert np.allclose(survey_pmf, survey_pmf_conv)


class TestResponseRaterBehavior:
    """Test behavioral properties and parameters."""
    """振る舞い（behavioral）に関する特性とパラメータ（parameters）をテストします。"""

    def test_temperature_affects_distribution_sharpness(self):
        """Higher temperature should create more uniform distributions."""
        """温度（temperature）が高いほど、より均一な（uniform）分布を作成するはずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        response = ["This is great"]
        sharp_pmf = rater.get_response_pmfs("set1", response, temperature=0.1)[0]
        smooth_pmf = rater.get_response_pmfs("set1", response, temperature=5.0)[0]

        # Higher temperature should increase entropy (more uniform)
        # 温度が高いほどエントロピー（不確実性）は増加する（より均一になる）はずです
        sharp_entropy = -np.sum(sharp_pmf * np.log(sharp_pmf + 1e-12))
        smooth_entropy = -np.sum(smooth_pmf * np.log(smooth_pmf + 1e-12))

        assert smooth_entropy > sharp_entropy, (
            "温度が高いほどエントロピーが増加するはずです"
        )

    def test_epsilon_affects_results(self):
        """イプシロン（epsilon）による正則化（regularization）は、出力を変更するはずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        response = ["Test response"]
        pmf_no_eps = rater.get_response_pmfs("set1", response, epsilon=0.0)
        pmf_with_eps = rater.get_response_pmfs("set1", response, epsilon=0.1)

        assert not np.allclose(pmf_no_eps, pmf_with_eps), (
            "イプシロンは結果を変更するはずです"
        )
        assert_valid_pmf(pmf_no_eps)
        assert_valid_pmf(pmf_with_eps)

    def test_mean_reference_aggregation(self):
        """Mean reference should work across multiple sets."""
        """（参照セットの）平均（mean）は、複数のセットにまたがって機能するはずです。"""
        df = create_test_dataframe(num_sets=2)
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        pmfs = rater.get_response_pmfs("mean", TEST_RESPONSES)
        assert_valid_pmf(pmfs, expected_rows=len(TEST_RESPONSES))


class TestResponseRaterEdgeCases:
    """Edge cases and error conditions."""
    """エッジケース（極端な事例）とエラー条件をテストします。"""

    def test_empty_inputs_handled_gracefully(self):
        """Empty response lists should be handled without errors."""
        """空の回答リストは、エラーなしで処理されるはずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        pmfs = rater.get_response_pmfs("set1", [])
        assert pmfs.shape == (0, LIKERT_SCALE_SIZE)

    def test_single_response_works(self):
        """Single responses should work in both modes."""
        # Text mode
        """単一の回答は、両方のモードで機能するはずです。"""
        # テキストモード
        df_text = create_test_dataframe()
        rater_text = ResponseRater(df_text, model_name="all-MiniLM-L6-v2")
        pmf = rater_text.get_response_pmfs("set1", ["single response"])
        assert_valid_pmf(pmf, expected_rows=1)

        # Embedding mode
        # 埋め込みモード
        df_embed = create_test_dataframe(include_embeddings=True)
        rater_embed = ResponseRater(df_embed)
        embedding = np.random.randn(1, EMBEDDING_DIM)
        pmf = rater_embed.get_response_pmfs("set1", embedding)
        assert_valid_pmf(pmf, expected_rows=1)

    def test_mode_validation(self):
        """Should reject wrong input types for each mode."""
        # Text mode should reject embeddings
        """各モードにおいて、間違った入力の型（タイプ）を拒否（reject）するはずです。"""
        # テキストモードは埋め込みを拒否するはずです
        df_text = create_test_dataframe()
        rater_text = ResponseRater(df_text, model_name="all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="Expected list of text strings"):
            rater_text.get_response_pmfs("set1", np.random.randn(2, EMBEDDING_DIM))

        # Embedding mode should reject text
        # 埋め込みモードはテキストを拒否するはずです
        df_embed = create_test_dataframe(include_embeddings=True)
        rater_embed = ResponseRater(df_embed)
        with pytest.raises(ValueError, match="Numpy配列の埋め込みを想定"):
            rater_embed.get_response_pmfs("set1", ["text response"])

    def test_invalid_reference_set(self):
        """Should raise clear error for nonexistent reference sets."""
        """存在しない参照セットに対して、明確なエラーを発生させる（raise）はずです。"""
        df = create_test_dataframe()
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        with pytest.raises(KeyError):
            rater.get_response_pmfs("nonexistent", ["test"])


class TestResponseRaterValidation:
    """Input validation tests."""
    """入力の検証（validation）に関するテストです。"""

    def test_incomplete_dataframe_rejected(self):
        """Should reject DataFrames missing required columns."""
        """必須の列（columns）が欠けているデータフレーム（DataFrames）を拒否するはずです。"""
        incomplete_df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                # Missing 'sentence' column
                # 'sentence' 列が欠けている
            }
        )

        with pytest.raises(ValueError, match="missing.*sentence"):
            ResponseRater(incomplete_df)

    def test_incomplete_likert_scale_rejected(self):
        """Should reject reference sets with incomplete Likert scales."""
        """不完全な（incomplete）リッカート尺度を持つ参照セットを拒否するはずです。"""
        incomplete_df = po.DataFrame(
            {
                "id": ["set1"] * 4,  # Only 4 responses instead of 5 # 5つではなく4つの回答しかない
                "int_response": [1, 2, 3, 4],
                "sentence": SAMPLE_TEXTS[:4],
            }
        )

        with pytest.raises(AssertionError):
            ResponseRater(incomplete_df)


class TestResponseRaterUtilities:
    """Test utility methods and properties."""
    """ユーティリティ（補助的な）メソッド（関数）とプロパティ（属性）をテストします。"""

    def test_reference_set_access(self):
        """Should provide access to reference sets and sentences."""
        """参照セットと参照文へのアクセスを提供するはずです。"""
        df = create_test_dataframe(num_sets=2)
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        available_sets = rater.available_reference_sets
        assert "set1" in available_sets
        assert "set2" in available_sets
        assert len(available_sets) == 2

        sentences = rater.get_reference_sentences("set1")
        assert sentences == SAMPLE_TEXTS

    def test_mode_detection(self):
        """Should correctly detect and report operating mode."""
        # Text mode
        """動作モード（operating mode）を正しく検出し、報告するはずです。"""
        # テキストモード
        df_text = create_test_dataframe()
        rater_text = ResponseRater(df_text, model_name="all-MiniLM-L6-v2")
        info = rater_text.model_info
        assert info["mode"] == "text"
        assert "embedding_dimension" in info

        # Embedding mode
        # 埋め込みモード
        df_embed = create_test_dataframe(include_embeddings=True)
        rater_embed = ResponseRater(df_embed)
        info = rater_embed.model_info
        assert info["mode"] == "embedding"
        assert info["embedding_dimension"] == EMBEDDING_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
