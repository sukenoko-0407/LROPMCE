# ECMPORL 要件定義書 v2

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v2.0
- 作成日: （日付は実装時に記載）
- 対象読者:
  - 本パッケージの設計・実装担当エンジニア
  - 創薬研究者（アルゴリズム仕様を理解したい読者）

---

## 2. システム概要

### 2.1 目的

ECMPORL は、低分子化合物の構造空間を **MCTS（Monte Carlo Tree Search）** により探索し、高報酬（複数の物性・動態指標を相乗平均した総合報酬）が期待される化合物およびその化学サブスペースを、できる限り網羅的に抽出する Python パッケージである。

### 2.2 特徴

- **UCT / PUCT** いずれの方策でも探索可能（引数で切替）。
- PUCT は AlphaGo Zero 型の式  
  \[
  \text{score}(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N_{\text{total}}+1}}{1+N_{sa}}
  \]
  をそのまま用いる。
- **統合モデル**（AlphaGoZero方式）: 状態価値 V(s) と行動方策 π(a|s) を同時出力する PyTorch モデルを利用。
- 開始コア構造から、事前定義された Fragment を `combine_smiles` + `hydrogen_replace` によって段階的に伸長。
- 分子全体の HAC / cnt_hetero / cnt_chiral / MW の上下限で探索空間を制御。
- **MCTSTree から学習データを抽出**し、一定間隔でモデルを学習。
- 探索後の Tree と学習済みモデルを保存し、後から高期待報酬ノード群を抽出する **推論専用モード** を提供。

---

## 3. 用語定義

- **状態 (State)**: ダミーアトムを含む SMILES で表現される部分構造（ダミーアトム以外に重水素を有することがある）。`MCTSNode` が保持する。
- **行動 (Action)**: Fragment テーブル中の Fragment を選択し、現在の状態と結合する操作。
- **コア構造**: 探索の起点となるダミーアトム付き SMILES（例: `*c1ccccc1`）。
- **Fragment**: ダミーアトム付き SMILES（例: `*CCN`）と、その物性情報（HAC, cnt_hetero, cnt_chiral, MW）を持つ単位構造。
- **Leaf**: ダミーアトムのない SMILES（報酬評価に用いる化合物）。Tree 上のノードとはみなさない、中間評価対象。
- **Alert 構造**: 望ましくない構造群。該当する状態は即座に terminal とする。
- **num_sub**: ある Node の先に存在し得る化合物数（化学サブスペースの大きさ）。ユーザ提供の関数で算出する。
- **Simulation（シミュレーション）**: MCTSの1回の反復。Selection → Expansion → Rollout → Backpropagation の4フェーズからなる。
- **Rollout（ロールアウト）**: 本システムでは、選択されたノードから深さ優先的に展開を継続し、Leaf評価可能なノードに到達するか、展開不能になるまで繰り返すフェーズ。標準MCTSのランダムプレイアウトとは異なる。

---

## 4. 前提条件・制約

### 4.1 利用環境

- Python 3.10 以上（目安）
- PyTorch（2.2を想定）
- RDKit が利用可能であること（SMILES 正規化・物性計算・構造判定に利用）
- Pandas（Fragment テーブル管理）

### 4.2 ユーザ提供コンポーネント

以下の機能・モデルは、ユーザが外部で準備することを前提とする。

1. **PyTorch モデル**（統合モデル：AlphaGoZero方式）
   - **統合モデル**: `Model(state_smiles: str) -> Tuple[Tensor[N], float]`
     - 状態（ダミーアトム付きSMILES）のみを入力とし、以下の2つを同時に出力する：
       - **Policy head**: 全Fragment（N個固定）に対する行動確率（softmax前のlogits）`Tensor[N]`
       - **Value head**: その状態の価値（スカラー）`float`
     - アーキテクチャ:
       ```
                           ┌─→ Policy head → p(a|s) : N次元ベクトル（全Fragmentへの確率）
       状態 s → 共通trunk ─┤
                           └─→ Value head  → v(s)   : 状態価値スカラー
       ```
   - **Policy head の使用方法**:
     - PUCTのprior P(s,a)として使用。
     - 無効なFragment（制約違反等）はマスキングしてsoftmax正規化。
     - 学習ターゲット: Tree上で実際に選択された行動の分布（子ノードの訪問回数比率）。
   - **Value head の使用方法**:
     - MCTSのバックアップで Q(s,a) を更新するために使用。
     - 学習ターゲット: Tree上のノードの `q_value`（平均報酬）。
   - **備考**:
     - Fragment数Nは固定（例: 300-400、最大2000程度まで対応可能）。
     - Fragmentセットを変更する場合は再学習が必要。

2. **報酬関数群**（1〜5 個）
   - `reward_i(smiles: str | list[str]) -> float | list[float]`
   - 各報酬は [0,1] の範囲で返す。
   - 合計報酬は、N 個の報酬の **相乗平均** として定義する。

3. **SMILES 操作関数**
   - `combine_smiles(core_like: str, frag_like: str) -> str`
     - いずれもダミーアトム付き SMILES を受け取り、２つを結合する。**ダミーアトムを含まない** SMILES を返す。FragmentのSMILESについては結合前に水素を全て重水素に変換する前処理もこの関数が実施する。
   - `hydrogen_replace(smiles_no_dummy: str) -> list[str]`
     - 重水素原子をダミーアトムに置換する全パターンを列挙。
     - 返り値は「ダミーアトムを 1 個含む SMILES」のリスト。
   - `remove_deuterium(smiles_with_deuterium: str) -> str`
     - 重水素原子をまず通常の水素に変換し、さらに水素除去したSMILESを返す。
     - Leaf SMILESを生成するために使う。

4. **Alert 判定関数**
   - Alert 判定関数は2種類存在する。
   - 1つは alert_elem_check であり、これはダミーアトムが入ったSMILESに対して適用する（すなわちFragment専用）。
   - もう1つは alert_mol_check であり、これはダミーアトムが入っていないSMILESに対して適用する（すなわちLeaf専用）
   - `is_alert_elem(smiles: str) -> 0 or 1 の int を出力する`
     - 0 の場合、その状態は（たとえFragmentであったとしても） is_terminal=True とする。
   - `is_alert_mol(smiles: str) -> 0 or 1 の int を出力する`
     - 0 の場合、そのLeafの報酬は0である。

5. **分子全体計測関数**
   - `measure_mol_props(smiles: str) -> dict`
     - 例: `{"HAC": int, "cnt_hetero": int, "cnt_chiral": int, "MW": float}`

6. **化学サブスペースサイズ算出関数**
   - `count_subspace(node_state_smiles: str) -> int`
   - そのノードの先に存在し得る全化合物数を返し、`num_sub` として持たせる。

7. **Fragment グラフ featurizer**
   - `fragment_featurizer(smiles: str) -> Any`
   - Fragment テーブル中の SMILES から、Policy_model 入力用のグラフ表現を作る。

### 4.3 探索スケール

- Fragment パターン数: 最大約 2000。
- 各状態における有効行動数（フィルタ後）は、最大数百から1000程度を想定。
- Tree ノード数は、実用上、数万〜数十万程度まで扱えるように実装することを目標とする（詳細は設計時に調整）。

---

## 5. パッケージ構成（クラス要件レベル）

パッケージ名: `ecmporl_02`

必須クラス:

- `MCTSNode`
- `MCTSTree`
- `Environment`
- `Agent`

モジュール構成（例示、確定は設計仕様書側で行う）:

- `ecmporl.core.mcts_node`
- `ecmporl.core.mcts_tree`
- `ecmporl.env.environment`
- `ecmporl.agent.agent`
- `ecmporl.io.serialization`
- `ecmporl.config` （探索パラメータ定義）
- `ecmporl.inference` （推論専用ユーティリティ）

---

## 6. 機能要件

### 6.1 MCTS 全体フロー

1. 初期状態:
   - コア構造（ダミーアトム付き SMILES）を入力し、根ノード `root` を作成。
2. シミュレーション（1 回）:
   1. Selection: UCT または PUCT により、root から子ノードを辿り、展開対象ノードを選択。
   2. Expansion:
      - 選択ノードから伸長可能な Fragment をフィルタリング（制約条件に基づく）し、1 つの Fragment を選択。
      - 選択された Fragment と現在の状態を `combine_smiles` で結合。
      - `hydrogen_replace` を用いて次状態候補リスト（ダミーアトム付きSMILES群）を生成。
      - 各次状態候補に対して V(s) による softmax サンプリングで 1 状態を選び、新ノードを生成・Tree に追加。
      - 新ノードについて以下を実施:
        - `is_alert_elem` による Alert 判定 → Alert の場合は `is_terminal = True`
        - `depth >= max_depth` の場合も `is_terminal = True`
        - `num_sub` 計算
        - `leaf_calc` の初期化:
          - `depth >= min_depth` かつ分子プロパティが下限を満たす場合: `leaf_calc = "ready"`
          - それ以外: `leaf_calc = "not_ready"`
   3. Rollout:
      - **深さ優先展開ループ**: 以下の処理を、停止条件を満たすまで繰り返す:
        a. 現在のノードの `leaf_calc` 状態をチェック:
           - `leaf_calc == "not_ready"`: Leaf評価をスキップ。
           - `leaf_calc == "ready"`: Leaf（ダミーアトムなしSMILES）を生成し、`is_alert_mol` でAlert判定。Alertでなければ報酬を評価。
           - `leaf_calc == "evaluated"`: Leaf評価をスキップ（既に評価済み）。

        b. 展開継続の判定:
           - 以下の**いずれか**を満たす場合、Rolloutを終了:
             - `is_terminal == True`（Alert判定またはmax_depth到達）
             - `depth >= max_depth`
             - 有効な行動（Fragment）が存在しない
           - 上記を満たさない場合、Expansionフェーズに戻り、さらに1ノード展開して深さ優先探索を継続。

        c. 行き詰まり時の処理:
           - `depth < min_depth` かつ展開不能の場合: 報酬 = 0 とする。
   4. Backpropagation:
      - Leaf で得た報酬（合計報酬）を root までの経路上のノードに逆伝播し、
        - 累計訪問回数 `N`
        - 累計報酬 `W`
        - 平均報酬 `Q = W / N`
        を更新。
      - Leaf評価を実行した場合は、評価を実行したノードの `leaf_calc` を `"evaluated"` に更新。

3. 上記シミュレーションを `max_simulation` 回繰り返す。

---

### 6.2 MCTSNode クラス要件

`MCTSNode` は以下の情報を保持する。

1. **状態情報**
   - `state_smiles`: ダミーアトムを含む SMILES（canonicalized）。
   - `depth`: ルートからの深さ（コアを depth=0 とする）。
   - `leaf_smiles`: ダミーアトムが水素原子で埋まったSMILES（重水素は含まないこととする）
   - `leaf_calc`: Leaf評価の状態を記録する。以下の3つの状態を持つ:
     - `"not_ready"`: まだLeaf評価の条件（`depth >= min_depth`など）を満たしていない。
     - `"ready"`: Leaf評価の条件を満たしているが、まだ評価を実行していない。
     - `"evaluated"`: すでにLeaf評価を実行済み。次回の訪問時はLeaf評価をスキップし、直接Expansionに進む。

2. **統計量**
   - `visit_count (N)`: 累計訪問回数。
   - `total_reward (W)`: 累計報酬。
   - `q_value (Q)`: `total_reward / visit_count`。本システムにおける最重要指標。
   - 必要に応じて、分散などの追加統計量を拡張可能とする。

3. **構造情報**
   - `parent`: 親ノードへの参照（root の場合は None）。
   - `incoming_action`: このノードへの遷移に使用されたFragment SMILES（rootの場合はNone）。Policy_modelの学習データ抽出に使用。
   - `children`: 子ノードの辞書またはリスト。キーは行動（Fragment SMILES）＋次状態SMILESなど、設計で定義。
   - `is_terminal`: bool。後続の展開が禁止されるかどうか。
   - `num_sub`: int。化学サブスペースの大きさ（提供関数で算出）。

4. **その他**
   - `metadata`: RDKit などで得られる補助情報のための任意フィールド（オプション）。
   - ノード ID として `state_smiles`（＋depth）を用いたユニーク性管理を行う想定。

**要件:**

- 別経路から同一状態に到達する場合は、既存ノードを再利用し、新規にノードを作らない。
  - 状態のユニーク性は「canonical SMILES + depth」で判定する。
- `MCTSNode` は、UCT / PUCT のスコア計算に必要な値（Q, N, 親の N_total）が取得できるインターフェイスを持つ。

---

### 6.3 MCTSTree クラス要件

1. **基本機能**
   - `root` ノードを保持。
   - 全ノードを管理するインデックス（例: `dict[state_smiles, MCTSNode]`）を持ち、状態→ノードへの高速アクセスを可能にする。

2. **検索・集計機能**
   - 条件付きでノードを抽出するメソッドを提供:
     - 例:
       - `total_reward >= R_min`
       - `num_sub >= S_min AND total_reward >= R_min`
       - `q_value >= Q_min`
     - 上記条件を AND/OR で組み合わせられるようなインターフェイスを用意する（詳細は設計で定義）。

3. **状態重複の取り扱い**
   - 同一状態（canonical SMILES＋depth）に対するノードは Tree 内で一意であり、別経路から到達した場合は同じノードの統計量を更新する。
   - トランスポジションテーブル的な役割を持つ。

4. **学習データ抽出支援**
   - V_model 学習用: 全ノードまたは条件を満たすノードから `(state_smiles, q_value)` ペアを抽出するメソッドを提供。
   - Policy_model 学習用: 子ノードを持つノードから `(parent_state_smiles, children_actions, children_visit_counts)` を抽出するメソッドを提供。

5. **シリアライズ支援**
   - Tree の保存／復元に必要な情報（ノード・エッジ・統計量）を出力・復元できるような API を備える。

---

### 6.4 Environment クラス要件

`Environment` は、化学的な制約・評価関数・Fragment 情報を管理する。

1. **Fragment 管理**
   - Fragment テーブル（Pandas DataFrame）を保持。
     - 必須列:
       - `smiles`（ダミーアトム付き SMILES）
       - `HAC`, `cnt_hetero`, `cnt_chiral`, `MW`
   - Fragment 数は最大 2000 程度。
   - 初期化時に、`fragment_featurizer` を用いて全 Fragment のグラフ表現を前計算し、`Policy_model` 入力に利用できる形でキャッシュする。

2. **分子全体プロパティ評価**
   - `calc_mol_props` を内部から呼び出し、分子全体の
     - HAC
     - cnt_hetero
     - cnt_chiral
     - MW
     を返すヘルパーメソッドを提供。

3. **制約設定**
   - 分子全体の各プロパティに対して **下限・上限** を指定可能とする。
     - 例: `HAC_min`, `HAC_max` など。
   - 各状態で有効な Fragment を列挙する際に、これら制約を用いて事前フィルタリングを行う:
     - 任意 Fragment を追加した結果が上限を超える場合、その Fragment はその状態での行動候補から除外。

4. **Alert 判定**
   - 2種類のAlert判定メソッドを提供:
     - `is_alert_elem(smiles)`: 状態（ダミーアトム付きSMILES）のAlert判定。新ノード生成時に呼び出し、Alertの場合は `is_terminal=True` とする。
     - `is_alert_mol(smiles)`: Leaf（ダミーアトムなしSMILES）のAlert判定。Leaf評価時に呼び出し、Alertの場合は報酬を0とする。

5. **サブスペースサイズ**
   - `count_subspace(state_smiles)` を呼び出し `num_sub` を算出するメソッドを提供。
   - `MCTSNode` 生成時に呼び出し、`num_sub` を設定する。

6. **報酬評価**
   - Leaf（ダミーアトム無し SMILES）を受け取り、ユーザ提供の複数報酬関数を呼び出し、その相乗平均を返す:
     - N 個の報酬 `r_1, ..., r_N` に対し:
       \[
       R_{\text{geom}} = \left(\prod_{i=1}^{N} r_i\right)^{1/N}
       \]
   - `min_depth` 未満で行き詰まった場合は、報酬 0 を返せるようにする（Agent 側から指定）。

---

### 6.5 Agent クラス要件

`Agent` は、MCTS 探索の制御、ML モデルの保持・学習を担う。

1. **主要引数・パラメータ**
   - `max_depth`: 探索の最大深さ。これ以上は展開しない。
   - `min_depth`: この深さに達していなければ、Leaf 評価を行わず展開を継続する。
   - `max_simulation`: 1 回の実行で行う MCTS シミュレーション回数。
   - `mcts_mode`: `"uct"` または `"puct"`。
   - `c_uct`: UCT の探索定数（任意）。
   - `c_puct`: PUCT の探索定数。
   - 学習関連:
     - `train_interval`: 何シミュレーションごとに学習を実行するか。
     - `batch_size`: 学習時のミニバッチサイズ。
     - `q_threshold_for_training`: 学習データ抽出時に使用する q_value の閾値（高報酬ノードを優先的にサンプリングするため）。
   - 温度スケジューラ関連:
     - `tau_initial`, `tau_final`
     - `tau_scheduler_type`: 例えば `"linear"`, `"exponential"` の 2 パターンをサポート。

2. **保持する要素**
   - `environment: Environment`
   - `tree: MCTSTree`
   - `model`: 統合モデル（PyTorch モデル、Policy head + Value head）
   - 学習に用いる Optimizer, Loss 関数などは設計時に定義。

3. **学習データの抽出（Treeベース）**
   - 学習データは MCTSTree から直接抽出する（別途メモリを持たない）。
   - `train_interval` シミュレーションごとに、現在の Tree からデータを抽出して学習を実行。
   - **統合モデルの学習データ**:
     - 各ノードから `(state_smiles, policy_target, value_target)` を抽出。
     - **Value target**: Tree上のノードの `q_value`（平均報酬）。
     - **Policy target**: 子ノードの訪問回数比率 `π_target(a) = N(s,a) / Σ_a' N(s,a')`
       - 子ノードを持つノードのみが対象。
       - `children_info` には各子ノードの `incoming_action`（Fragment index）と `visit_count` を含む。
     - `q_threshold_for_training` 以上のノードを優先的にサンプリング可能。
   - **Loss関数**（AlphaGoZero方式）:
     - `L = (v - z)² - π^T log(p) + c||θ||²`
     - v: Value head出力, z: value_target, π: policy_target, p: Policy head出力

4. **PUCT 用 P(s,a) の定義**
   - 統合モデルのPolicy headが出力するlogitsを使用。
   - 無効な行動（制約違反のFragment）をマスキング（-inf）した後、softmax で正規化し P(s,a) とする。

5. **V(s) による次状態選択**
   - `combine_smiles` → `hydrogen_replace` により複数の次状態候補 `{s'_i}` が生成された際、
     - 各候補に対して統合モデルのValue head出力 v(s'_i) を計算。
     - 温度付き softmax:
       \[
       p_i \propto \exp(V(s'_i)/\tau)
       \]
       に基づき 1 状態をサンプルし、次ノードとして採用。
   - 同じノードが再訪問され、その行動が再度選択されるたびに、V(s) による softmax を **再計算** する。

6. **温度 τ のスケジューリング**
   - Agent は τ を更新するスケジューラを持つ。
   - 少なくとも以下 2 種類をサポート:
     1. 線形減衰: シミュレーション回数またはエピソード数に応じて τ_initial → τ_final まで線形に減少。
     2. 指数減衰: τ = τ_initial * exp(-k * step) など。
   - スケジューラ種別は引数で切り替える。

7. **学習タイミング**
   - MCTS シミュレーション実行中に、`train_interval` ごとに学習を実行。
   - 学習時は現在の Tree から学習データを抽出し、バッチ学習を行う。

8. **モデルの保存・読み込み**
   - PyTorch の state_dict を用いて、統合モデルの重みを保存・読み込みできる API を提供。
   - 推論専用モードでは学習を行わず、読み込んだ重みで探索およびノード評価のみを行う。

---

### 6.6 UCT / PUCT 選択と Selection アルゴリズム

1. **UCT モード**
   - 各ノードで有効行動の中から、標準的な UCT スコアに基づいて子ノードを選択する。
   - UCT スコアの具体式（例）:
     \[
     \text{score}(s,a) = Q(s,a) + c_{\text{uct}} \cdot \sqrt{\frac{\log(N_{\text{total}}+1)}{1+N_{sa}}}
     \]
   - UCT モードでは、
     - 未訪問の行動／ノードを優先的に展開するロジックを明示的に持つこと。

2. **PUCT モード**
   - 上記の PUCT 式を用いる。
   - P(s,a) は Policy_model から得られる行動確率（logits を softmax 正規化）を用いる。
   - Q(s,a) は Tree 上の Q 値（累計報酬／訪問回数）を用いる。

---

### 6.7 Rollout と報酬評価のタイミング

1. **Rolloutフェーズの深さ優先展開**
   - Rolloutフェーズは、**1回のシミュレーション内で複数ノードを展開する深さ優先探索**である。
   - 以下の停止条件を満たすまで、Expansion → Leaf評価チェック → 継続判定 のループを繰り返す:
     - `is_terminal == True`
     - `depth >= max_depth`
     - 有効な行動（Fragment）が存在しない

2. **`leaf_calc` 状態の管理**
   - 各ノードは `leaf_calc` プロパティを持ち、Leaf評価の状態を管理する。
   - ノード生成時（Expansionフェーズ）:
     - `depth >= min_depth` かつ分子全体のプロパティが下限を満たす場合: `leaf_calc = "ready"`
     - それ以外: `leaf_calc = "not_ready"`
   - ノード訪問時（Rolloutフェーズのループ内）:
     - `leaf_calc == "not_ready"`: Leaf評価をスキップし、次のExpansionに進む。
     - `leaf_calc == "ready"`: Leaf（ダミーアトムなし完成分子）を生成し、`is_alert_mol` でAlert判定。Alertでなければ報酬を評価。
     - `leaf_calc == "evaluated"`: **Leaf評価をスキップ**し、次のExpansionに進む（重複評価を避ける）。
   - Backpropagationフェーズ:
     - Leaf評価を実行した場合は、評価を実行したノードの `leaf_calc` を `"evaluated"` に更新してから報酬を逆伝播する。

3. **報酬評価のタイミング**
   - Leaf評価は、あるノードに対して**最初に** `leaf_calc == "ready"` の状態で訪問したときに**1回だけ**実行される。
   - 同じノードが再度訪問された場合、`leaf_calc == "evaluated"` となっているため、Leaf評価はスキップされ、直接次のExpansionに進む。
   - **重要**: Leaf評価を行った後も、そのノードから展開可能であれば、同じシミュレーション内でさらに深い展開を継続する。

4. **ノードの展開可能性**
   - あるノードでLeaf評価を行った後も、そのノードは `is_terminal=False` である限り、上記の停止条件を満たさないならば展開を継続する。もちろん次回のシミュレーションで再度選択されたなら、さらに展開可能である。
   - つまり、「Leaf評価を行う」ことと「ノードの展開終了」は独立している。
   - 深さ優先探索により、同じシミュレーション内で `min_depth` から `max_depth` まで一気に展開することもある。

5. **行き詰まりの扱い**
   - `depth < min_depth` かつ以下のいずれかの場合:
     - 制約を満たす Fragment が存在しない。
     - `is_alert_elem` により `is_terminal=True` となった。
   - 上記の場合は、Leaf 評価を行わずに **報酬 0** として Backpropagation する。

---

### 6.8 状態重複（トランスポジション）への対応

- 同一の canonical SMILES（＋同じ depth）に到達する複数の経路が存在し得る。
- その場合は **別ノードを新規作成せず、既存ノードを使い回し**、訪問回数・報酬を統合する。
- depth が同じになることは Fragment が最小単位に分解されていることから保証される前提とする。

---

### 6.9 I/O とシリアライズ

1. **Tree の保存**
   - 探索完了後、`MCTSTree` の内容を容量効率の良い形式で保存する API を提供する。
   - 要件レベルではフォーマットは固定しないが、以下を満たすこと:
     - ノード数が多い場合でも、現実的なサイズに収まる（例: JSON よりもバイナリ形式が望ましい）。
     - 再ロード後もノード間の親子関係、統計量（N, W, Q, num_sub）を完全に再現できる。

2. **モデル重みの保存**
   - PUCT モードで学習した V_model, Policy_model を、ファイルパスを受け取って保存・読み込み可能にする。

3. **設定の保存（オプション）**
   - 探索条件（制約、max_depth, min_depth, max_simulation 等）を再現可能な形で保存・読込できると望ましい（実装は任意）。

---

### 6.10 推論専用モード（Inference）

1. **目的**
   - 学習済みモデルと Tree（あるいは新たに探索された Tree）を用いて、
     - 平均報酬 `Q = total_reward/visit_count` が高いノードを、指定した閾値以上のものを「できる限り漏れなく」抽出する。

2. **要件**
   - ユーザは以下を指定してノード集合を取得できる:
     - `q_value_threshold`（必須）
     - 任意の追加条件:
       - `num_sub >= S_min`
       - `total_reward >= R_min`
       - depth 範囲（例: `depth_min <= depth <= depth_max`）
   - 探索コストが増えても良いので、上記条件を満たすノードを網羅的に抽出することを目指す。
   - 一部の漏れ（ヒューリスティックのため）は許容されるが、実装上は極力漏れを減らすようにする。

3. **利用イメージ**
   - 学習済みモデル＋既存 Tree を読み込む。
   - 必要に応じて追加探索を行い Tree を拡充（任意）。
   - 高期待報酬ノードのリストを出力し、ユーザが SMILES, num_sub, Q などを参照できる。

---

## 7. 非機能要件

1. **拡張性**
   - 報酬関数や Alert 判定関数の追加・入れ替えが容易であること。
   - Fragment テーブル列の追加にも耐えられる設計とする。

2. **性能**
   - Fragment 数 ~1000、ノード数数万〜数十万規模の探索を想定。
   - シリアライズ／デシリアライズは現実的な時間で完了すること。

3. **テスト容易性**
   - 各クラス（Environment, Agent, MCTSTree, MCTSNode）を個別にユニットテスト可能な形で実装する。
   - combine_smiles, hydrogen_replace 等をダミー実装と差し替えてテストできること。

4. **ロギング・デバッグ**
   - シミュレーション進行状況（現在シミュレーション番号、Tree サイズ、平均報酬など）をロギングできる拡張ポイントを用意。
   - 重要イベント（Alert による terminal 判定、行き詰まりによる報酬 0 など）を識別可能にする。

---

## 8. 公開インタフェースの例（イメージ）

※最終的なシグネチャは設計仕様書で確定。

- `Environment`:
  - `Environment(fragment_df, combine_fn, hydrogen_replace_fn, reward_fns, alert_fn, calc_props_fn, count_subspace_fn, fragment_featurizer, constraints)`
- `Agent`:
  - `Agent(env, model, config)`  # model: 統合モデル（Policy head + Value head）
  - `Agent.run_mcts(core_smiles) -> MCTSTree`
  - `Agent.save_model(path)` / `Agent.load_model(path)`
- `MCTSTree`:
  - `tree.filter_nodes(q_min=None, total_reward_min=None, num_sub_min=None, depth_range=None) -> List[MCTSNode]`
  - `tree.extract_training_data(q_min=None) -> List[Tuple[str, Tensor[N], float]]`
    # 各ノードから (state_smiles, policy_target, value_target) を抽出
  - `tree.save(path)` / `MCTSTree.load(path)`

---

以上が ECMPORL_02 パッケージの要件定義書 v2。
