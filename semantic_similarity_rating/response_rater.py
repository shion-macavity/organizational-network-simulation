"""
自動的な埋め込み（embedding）計算を用いて、テキスト回答を参照文（reference sentences）と比較し、
評価・分析するためのモジュール（機能のまとまり）です。

このモジュールは以下の機能を提供します：
- sentence-transformers（テキスト埋め込みを計算するライブラリ）を使用して、テキスト文の埋め込みを自動的に計算する
- 参照文のデータ構造が正しいか検証（validate）する
- LLM（大規模言語モデル）のテキスト回答を確率分布に変換する
- 異なる参照セット（reference sets）を使用して、調査回答のPMF（確率質量関数）を計算する
- 回答を、平均の参照セットまたは特定の参照セットと比較する

このモジュールは特に、LLMからのリッカート尺度（段階評価）の回答を分析する際に有用であり、
セマンティック埋め込み（意味的な埋め込み）を用いて、それらのテキストを参照文のテキストと比較します。
"""

import numpy as np
import polars as po
from sentence_transformers import SentenceTransformer

from . import compute


def _assert_reference_sentence_dataframe_structure(df, embeddings_column=None):
    """
    参照文のデータフレーム（DataFrame: 表形式のデータ構造）の構造が正しいか検証します。

    引数 (Parameters)
    ----------
    df : polars.DataFrame (Polarsライブラリのデータフレーム)
        参照文と、任意で埋め込み（embeddings）を含むデータフレーム
    embeddings_column : str (文字列), optional (任意)
        埋め込みを含む列（column）の名前（もし提供されていれば）

    例外 (Raises)
    ------
    ValueError
        必須の列が存在しない場合
    AssertionError
        回答の構造が無効な場合
    """
    required_cols = ["id", "int_response", "sentence"]
    if embeddings_column:
        required_cols.append(embeddings_column)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected reference-sentence data frame to have columns {required_cols}, "
            f"but missing: {missing_cols}. Available columns: {df.columns}"
        )
    
    #missing_cols = [col for col in required_cols if col not in df.columns]
        #if missing_cols:
            #raise ValueError(
                #f"参照文のデータフレームには {required_cols} の列が必要ですが、"
                #f"次の列がありません: {missing_cols}。利用可能な列: {df.columns}"
            #)


    agg = df.group_by("id").agg(po.col("int_response")).sort("id")

    assert "mean" not in agg["id"]
    for i, int_resps in zip(agg["id"], agg["int_response"]):
        #assert len(int_resps) == 7
        assert all([i + 1 == r for i, r in enumerate(sorted(int_resps))])


class ResponseRater:
 
    """
    文字列を、参照文との類似度に関するPMF（確率質量関数）に変換します。

    実際には、文字列は調査の質問に対するLLM（大規模言語モデル）の回答であり、参照文は
    リッカート尺度（段階評価）の各段階をテキストで表現したものである場合があります。この場合、PMFは
    各文字列がリッカート尺度の各段階を表す確率を示します。

    これは2つのモードで動作します：

    1. **埋め込みモード (Embedding mode)**: `df_reference_sentences`に "embedding" 列が含まれている場合、
       ユーザーが提供したそれらの埋め込みを使用し、入力として埋め込み（のデータ）を期待します。
    2. **テキストモード (Text mode)**: "embedding" 列が提供されていない場合、`sentence-transformers` を使用して
       自動的に埋め込みを計算し、入力としてテキスト（のデータ）を期待します。

    使用例 (Examples)
    --------
    **テキストモード（自動的な埋め込み計算）:**

    >>> import polars as po
    >>> from semantic_similarity_rating import ResponseRater
    >>>
    >>> # 参照文のデータフレームを作成（埋め込み列なし）
    >>> df = po.DataFrame({
    ...     'id': ['set1'] * 5,
    ...     'int_response': [1, 2, 3, 4, 5],
    ...     'sentence': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
    ... })
    >>>
    >>> rater = ResponseRater(df)
    >>> llm_responses = ["I totally agree", "Not sure about this"]  # テキスト入力
    >>> pmfs = rater.get_response_pmfs('set1', llm_responses)

    **埋め込みモード（事前に計算された埋め込み）:**

    >>> import numpy as np
    >>> # 埋め込み付きの参照文を作成
    >>> df = po.DataFrame({
    ...     'id': ['set1'] * 5,
    ...     'int_response': [1, 2, 3, 4, 5],
    ...     'sentence': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],
    ...     'embedding': [np.random.rand(384).tolist() for _ in range(5)]
    ... })
    >>>
    >>> rater = ResponseRater(df)
    >>> llm_embeddings = np.random.rand(2, 384)  # 埋め込み入力
    >>> pmfs = rater.get_response_pmfs('set1', llm_embeddings)
    """

    def __init__(
        self,
        df_reference_sentences: po.DataFrame,
        embeddings_column: str = "embedding",
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
       

        """
        参照文（reference sentences）を使ってResponseRaterを初期化（initialize）します。

        引数 (Parameters)
        ----------
        df_reference_sentences : polars.DataFrame (Polarsライブラリのデータフレーム)
            参照文と、任意で事前に計算された埋め込み（embeddings）を含むデータフレーム
        embeddings_column : str (文字列), optional (任意)
            埋め込みを含む列の名前。デフォルト（初期設定）は 'embedding'。
            この列が存在する場合、Rater（評価器）は埋め込みモードで動作します。
        model_name : str (文字列), optional (任意)
            使用するsentence-transformerモデルの名前（テキストモードのみ）。デフォルトは 'all-MiniLM-L6-v2'
        device : str (文字列), optional (任意)
            モデルを実行するデバイス（'cpu'、'cuda'など）（テキストモードのみ）。デフォルトは None（自動検出）
        """
        
        df = df_reference_sentences

        # Check if we're in embedding mode or text mode
        # 埋め込みモードかテキストモードかを確認
        self.embedding_mode = embeddings_column in df.columns
        self.embeddings_column = embeddings_column if self.embedding_mode else None

        # Validate dataframe structure
        # データフレームの構造を検証
        _assert_reference_sentence_dataframe_structure(df, self.embeddings_column)

        # Initialize sentence transformer model only in text mode
        # テキストモードの場合のみ、sentence transformerモデルを初期化
        self.model = None
        if not self.embedding_mode:
            self.model = SentenceTransformer(model_name, device=device)

        # Initialize storage for reference matrices and sentences
        # 参照行列と参照文のためのストレージ（保管場所）を初期化
        self.reference_matrices = {}
        self.reference_sentences = {"mean": ["1", "2", "3", "4", "5"]}

        # Process each unique sentence set
        # 一意な（重複のない）文のセットごとに処理
        unique_sentence_set_ids = df["id"].unique().sort()
        for sentence_set in unique_sentence_set_ids:
            this_set = df.filter(po.col("id") == sentence_set).sort(by="int_response")
            sentences = this_set["sentence"].to_list()

            # Store the actual sentences for reference
            # 実際の文を参照用に保存
            self.reference_sentences[sentence_set] = sentences

            if self.embedding_mode:
                # Use pre-computed embeddings
                # 事前に計算された埋め込みを使用
                embeddings = np.array(this_set[self.embeddings_column].to_list())
                M = embeddings.T  # Transpose to match expected format # 期待される形式に合わせるために転置（T: Transpose）する
            else:
                # Compute embeddings for the reference sentences
                # 参照文の埋め込みを計算
                embeddings = self.model.encode(sentences)
                M = embeddings.T  # Transpose to match expected format # 期待される形式に合わせるために転置する

            self.reference_matrices[sentence_set] = M

    def get_response_pmfs(
        self, reference_set_id, llm_responses, temperature=1.0, epsilon=0.0
    ):
        

        """
        指定された参照セット（reference set）を使用して、文字列をPMF（確率質量関数）に変換します。

        引数 (Parameters)
        ----------
        reference_set_id : str (文字列)
            使用する参照セットのID。または 'mean' を指定すると全てのセットの平均を使用します
        llm_responses : list of str (文字列のリスト) または numpy.ndarray (Numpyの配列)
            - テキストモードの場合: LLMの回答テキストのリスト
            - 埋め込みモードの場合: LLMの回答の埋め込みの行列（形状: 回答数 x 埋め込みの次元数）
        temperature : float (浮動小数点数)
            温度Tでスケーリング（調整）されたPMFを取得します:
            ``p_new[i] ~ p_old[i]^(1/T)``。
        epsilon : float (浮動小数点数), optional (任意)
            ゼロ除算（0で割ること）を防ぎ、スムージング（平滑化）を加えるための小さな正則化パラメータ。
            デフォルト（初期設定）は0.0（正則化なし）。

        戻り値 (Returns)
        -------
        numpy.ndarray (Numpyライブラリの配列)
            各回答に対する確率質量関数

        例外 (Raises)
        ------
        ValueError
            入力の型（タイプ）がRater（評価器）のモード（テキスト vs 埋め込み）と一致しない場合
        """

        if self.embedding_mode:
            # Embedding mode: expect numpy array of embeddings
            # 埋め込みモード: Numpy配列の埋め込みを期待
            if not isinstance(llm_responses, np.ndarray):
                raise ValueError(
                    "ResponseRater is in embedding mode (dataframe contains 'embedding' column). "
                    "Expected numpy array of embeddings, got: "
                    "ResponseRaterは埋め込みモードです（データフレームに 'embedding' 列が含まれています）。"
                    "Numpy配列の埋め込みを想定していましたが、受け取ったのは: "

                    + str(type(llm_responses))
                )
            llm_response_matrix = llm_responses
        else:
            # Text mode: expect list of strings and compute embeddings
            # テキストモード: 文字列のリストを期待し、埋め込みを計算
            
            if not isinstance(llm_responses, (list, tuple)):
                raise ValueError(
                    "ResponseRater is in text mode (no 'embedding' column in dataframe). "
                    "Expected list of text strings, got: " + str(type(llm_responses))
                )
                #raise ValueError(
                #    "ResponseRaterはテキストモードです（データフレームに 'embedding' 列がありません）。"
                #    "テキスト文字列のリストを想定していましたが、受け取ったのは: " + str(type(llm_responses))
                #)

            llm_response_matrix = self.model.encode(llm_responses)

        if isinstance(reference_set_id, str) and reference_set_id.lower() == "mean":
            # Calculate PMFs using mean over all reference sets
            # 全ての参照セットの平均を用いてPMFを計算
            llm_response_pmfs = np.array(
                [
                    compute.response_embeddings_to_pmf(llm_response_matrix, M, epsilon)
                    for M in self.reference_matrices.values()
                ]
            ).mean(axis=0)
        else:
            # Calculate PMFs using specific reference set
            # 特定の参照セットを用いてPMFを計算
            M = self.reference_matrices[reference_set_id]
            llm_response_pmfs = compute.response_embeddings_to_pmf(
                llm_response_matrix, M, epsilon
            )

        if temperature != 1.0:
            llm_response_pmfs = np.array(
                [compute.scale_pmf(_pmf, temperature) for _pmf in llm_response_pmfs]
            )

        return llm_response_pmfs

    def get_survey_response_pmf(self, response_pmfs):
        """
        個々の回答のPMF（確率質量関数）を平均化することにより、調査全体の回答PMFを計算します。

        引数 (Parameters)
        ----------
        response_pmfs : numpy.ndarray (Numpyライブラリの配列)
            個々の回答PMFの行列

        戻り値 (Returns)
        -------
        numpy.ndarray (Numpyライブラリの配列)
            調査全体の回答を表す平均PMF
        """
        return response_pmfs.mean(axis=0)

    def get_survey_response_pmf_by_reference_set_id(
        self, reference_set_id, llm_responses, temperature=1.0, epsilon=0.0
    ):
        """
        特定の参照セット（reference set）を使用して、調査回答のPMF（確率質量関数）を取得します。

        引数 (Parameters)
        ----------
        reference_set_id : str (文字列)
            使用する参照セットのID
        llm_responses : list of str (文字列のリスト) または numpy.ndarray (Numpyの配列)
            - テキストモードの場合: LLMの回答テキストのリスト
            - 埋め込みモードの場合: LLMの回答の埋め込みの行列
        temperature : float (浮動小数点数), default (デフォルト) = 1.0
            温度Tでスケーリング（調整）されたPMFを取得します:
            ``p_new[i] ~ p_old[i]^(1/T)``。
        epsilon : float (浮動小数点数), optional (任意)
            ゼロ除算（0で割ること）を防ぎ、スムージング（平滑化）を加えるための小さな正則化パラメータ。
            デフォルト（初期設定）は0.0（正則化なし）。

        戻り値 (Returns)
        -------
        numpy.ndarray (Numpyライブラリの配列)
            調査全体の回答を表す平均PMF
        """
        return self.get_survey_response_pmf(
            self.get_response_pmfs(
                reference_set_id, llm_responses, temperature, epsilon
            )
        )

    def encode_texts(self, texts):
        """
        読み込まれたモデルを使用して、テキストのリストに対する埋め込み（embeddings）を計算します。

        注記: このメソッド（関数）はテキストモードでのみ利用可能です。

        引数 (Parameters)
        ----------
        texts : list of str (文字列のリスト)
            エンコード（埋め込みに変換）するテキストのリスト

        戻り値 (Returns)
        -------
        numpy.ndarray (Numpyライブラリの配列)
            埋め込みの行列（形状: テキスト数, 埋め込みの次元数）

        例外 (Raises)
        ------
        ValueError
            埋め込みモードで呼び出された場合（sentence transformerモデルが読み込まれていないため）
        """
        if self.embedding_mode:
            raise ValueError(
                "encode_texts() is not available in embedding mode. "
                "Embeddings should be pre-computed and provided directly."
            )
        return self.model.encode(texts)
    
        #if self.embedding_mode:
        #    raise ValueError(
        #        "encode_texts() は埋め込みモードでは利用できません。"
        #        "埋め込みは事前に計算し、直接提供する必要があります。"
        #    )
        #return self.model.encode(texts)

    def get_reference_sentences(self, reference_set_id):
        

        """
        特定の参照セット（reference set）に対応する参照文を取得します。

        引数 (Parameters)
        ----------
        reference_set_id : str (文字列)
            参照セットのID

        戻り値 (Returns)
        -------
        list of str (文字列のリスト)
            参照文のリスト
        """
        return self.reference_sentences[reference_set_id]

    @property
    def available_reference_sets(self):
        
        """
        利用可能な参照セットIDのリストを取得します。

        戻り値 (Returns)
        -------
        list of str (文字列のリスト)
            利用可能な参照セットIDのリスト
        """

        return list(self.reference_matrices.keys())

    @property
    def model_info(self):
        """
        ResponseRaterに関する情報を取得します。

        戻り値 (Returns)
        -------
        dict (辞書型データ)
            モデルとモードの情報を含む辞書
        """
        info = {
            "mode": "embedding" if self.embedding_mode else "text",
            "embedding_dimension": list(self.reference_matrices.values())[0].shape[0]
            if self.reference_matrices
            else "Unknown",
        }

        if not self.embedding_mode and self.model:
            info.update(
                {
                    "model_name": str(self.model),
                    "max_seq_length": getattr(self.model, "max_seq_length", "Unknown"),
                    "device": str(self.model.device),
                }
            )

        return info
