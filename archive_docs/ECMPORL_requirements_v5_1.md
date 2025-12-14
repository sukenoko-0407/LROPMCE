# ECMPORL 要件定義書 v5.1

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v5.1
- 作成日: 2025/12/12
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
- **離散行動空間（AlphaGoZero方式）**: 行動（Fragment選択）を離散的に扱い、各行動の確率分布を直接出力。
- **一体型モデル構成（AlphaGoZero方式）**: Policy HeadとValue Headを共通のバックボーンから分岐させた一体型ネットワーク。
- 開始コア構造から、事前定義された Fragment を `combine_smiles` + `hydrogen_replace` によって段階的に伸長。
- 分子全体の HAC / cnt_hetero / cnt_chiral / MW の上下限で探索空間を制御。
- **バッチ処理**: 報酬計算とモデル学習を効率的にバッチ処理。評価待ちノードはSelection対象外。
- **並列探索と木マージ**: 大規模並列Simulationで得られた複数の木構造をマージして統合。
- 探索後の Tree と学習済みモデルを保存し、後から高期待報酬ノード群を抽出する **推論専用モード** を提供。

---

## 3. 用語定義

- **状態 (State)**: ダミーアトムを含む SMILES で表現される部分構造（ダミーアトム以外に重水素を有することがある）。`MCTSNode` が保持する。
- **行動 (Action)**: Fragment テーブル中の Fragment を選択し、現在の状態と結合する操作。**離散行動空間**において、各Fragmentのインデックスとして表現される。
- **コア構造**: 探索の起点となるダミーアトム付き SMILES（例: `*c1ccccc1`）。
- **Fragment**: ダミーアトム付き SMILES（例: `*CCN`）と、その物性情報（HAC, cnt_hetero, cnt_chiral, MW）を持つ単位構造。
- **Leaf**: ダミーアトムのない SMILES（報酬評価に用いる化合物）。Tree 上のノードとはみなさない、中間評価対象。
- **Alert 構造**: 望ましくない構造群。該当する状態は即座に terminal とする。
- **num_sub**: ある Node の先に存在し得る化合物数（化学サブスペースの大きさ）。ユーザ提供の関数で算出する。
- **Simulation（シミュレーション）**: MCTSの1回の反復。Selection → Expansion（必要に応じて複数回）→ Leaf評価要求（`pending`化）→ （評価完了後に）Backpropagation の流れ。**1回のSimulationで複数回の展開が発生し得る**。
- **Legal Action**: ある状態において制約条件を満たす有効なFragment群。
- **バッチ評価**: 複数のLeafに対する報酬計算をまとめて実行する処理。評価待ちノードはSelection対象外となる。

---

## 4. 前提条件・制約

### 4.1 利用環境

- Python 3.10 以上（目安）
- PyTorch（2.2を想定）
- RDKit が利用可能であること（SMILES 正規化・物性計算・構造判定に利用）
- Pandas（Fragment テーブル管理）

### 4.2 ユーザ提供コンポーネント

以下の機能・モデルは、ユーザが外部で準備することを前提とする。

1. **PyTorch モデル**（AlphaGoZero方式・一体型構成）

   本システムでは、行動を**離散空間**で扱う。AlphaGoZeroと同様の一体型ネットワークを使用する。

   #### 1.1 PolicyValueNetwork（一体型）

   - **入力**: 状態（ダミーアトム付きSMILES）
   - **出力**:
     - `policy_logits`: 全Fragment（最大約2000個）に対するlogit値のベクトル
     - `value`: その状態における期待平均報酬（スカラー）
   - **役割**: 状態に基づいて、各行動の確率分布と状態価値を同時に出力する。
   - **シグネチャ**: `PolicyValueNetwork(state_smiles: str) -> Tuple[Tensor[num_fragments], float]`
   - **構成**:
     - **Backbone（共通部分）**: 状態のSMILESをエンコードする共通のネットワーク
     - **Policy Head**: Backboneの出力から各Fragmentに対するlogitを出力
     - **Value Head**: Backboneの出力から状態価値（スカラー）を出力

   #### 1.2 学習方法（AlphaGoZero方式）

   - **学習データ**: MCTSの探索結果から抽出
     - Policy: 各ノードの子ノード訪問回数分布 `π = N(s,a) / Σ_a N(s,a)`
     - Value: そのノードの `q_value`（平均報酬）をターゲット `z` とする
   - **Loss関数（AlphaGoZero準拠）**:

     ```
     L = L_value + L_policy
     L_value = (v - z)²           # Value Lossは二乗誤差
     L_policy = -Σ_a π(a) log p(a)  # Policy Lossはクロスエントロピー
     ```

     - `v`: Value Headの出力
     - `z`: MCTSから得られたvalue target（q_value）
     - `p(a)`: Policy Headの出力（softmax後の確率）
     - `π(a)`: MCTSの訪問回数から計算されたターゲット分布
   - **学習対象**: 評価済み（`leaf_calc == "evaluated"`）の子ノードのみ。未評価ノードは学習対象外

   #### 1.3 行動選択の仕組み（離散行動空間）

   ```text
   状態 s → PolicyValueNetwork → (policy_logits, value)

   Legal Actions (M個のFragment) のインデックスを取得

   Legal Actionsに対応するlogitのみを抽出:
     legal_logits = policy_logits[legal_action_indices]

   正規化された選択確率（マスク付きsoftmax）:
     P(a | s) = softmax(legal_logits)
   ```

   - **行動の選択**: Legal Action内の各Fragmentについて、Policy Headが出力するlogitを抽出し、softmaxで確率化。
   - **PUCTのprior P(s,a)**: 上記の正規化確率を使用。
   - **Illegal Actionのマスク**: Legal Actions以外のFragmentは選択確率計算から除外される。

2. **報酬関数群**（1〜5 個）

   - `reward_i(smiles: str | list[str]) -> float | list[float]`
   - 各報酬は [0,1] の範囲で返す。
   - 合計報酬は、N 個の報酬の **相乗平均** として定義する。
   - **バッチ処理対応**: 複数のSMILESをリストで受け取り、リストで返すことを推奨。
   - **疎結合設計**: 報酬関数群は頻繁に入れ替わるため、Environmentとの連携は疎結合とする。
     - Environmentは報酬関数のリストを外部から注入可能な形で保持する。
     - 報酬関数の追加・削除・入れ替えが、Environment内部のロジックに影響を与えないこと。
     - 報酬関数は統一インターフェイス（上記シグネチャ）に従えば、任意の実装を差し替え可能。

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
   - Returns: 1 = 構造がOK（Alert非該当）, 0 = Alert該当（NG）
   - 1つは alert_elem_check であり、これはダミーアトムが入ったSMILESに対して適用する（すなわちFragment専用）。
   - もう1つは alert_mol_check であり、これはダミーアトムが入っていないSMILESに対して適用する（すなわちLeaf専用）
   - `alert_ok_elem(smiles: str) -> 0 or 1 の int を出力する`
     - 0 の場合、その状態は（たとえFragmentであったとしても） is_terminal=True とする。
   - `alert_ok_mol(smiles: str) -> 0 or 1 の int を出力する`
     - 0 の場合、そのLeafの報酬は0である。

5. **分子全体計測関数**

   - `measure_mol_props(smiles: str) -> dict`
     - 例: `{"HAC": int, "cnt_hetero": int, "cnt_chiral": int, "MW": float}`

6. **化学サブスペースサイズ算出関数**

   - `count_subspace(node_state_smiles: str) -> int`
   - そのノードの先に存在し得る全化合物数を返し、`num_sub` として持たせる。

7. **Fragment グラフ featurizer**

   - `fragment_featurizer(smiles: str) -> Any`
   - Fragment テーブル中の SMILES から、Fragment Embedding Model 入力用のグラフ表現を作る。

### 4.3 探索スケール

- Fragment パターン数: 最大約 2000。
- 各状態における有効行動数（フィルタ後）は、最大数百から1000程度を想定。
- Tree ノード数は、実用上、数万〜数十万程度まで扱えるように実装することを目標とする（詳細は設計時に調整）。
- **バッチサイズ**: 報酬計算のバッチは128〜512程度を想定（引数で指定可能）。

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
- `ecmporl.parallel` （並列処理・木マージユーティリティ）

---

## 6. 機能要件

### 6.1 MCTS 全体フロー

**重要（Simulation定義）**: 1回のSimulationでは、**Leaf評価可能（`leaf_calc == "ready"`）になるか、または探索上の終端（terminal/dead-end）に到達するまで、同一Simulation内で複数回の展開（Expansion）を行ってよい**。  
報酬計算はバッチ処理であり、`ready` に到達した場合は **評価待ち（pending）としてSimulationを終了**する。

1. **初期状態**
   - コア構造（ダミーアトム付き SMILES）を入力し、根ノード `root` を作成する。

2. **Simulation（1回）**: 以下のループで実行する。

   ### Phase 1: Selection
   - UCT または PUCT により、現在ノードから子ノードを辿り、展開対象のノードを選択する。
   - **評価待ちノード（`leaf_calc == "pending"`）はSelection対象外**とする（重複評価の抑止）。
   - もし「選択可能な候補が全てpending等で除外され、候補が空」になった場合は、**flush を要求し、評価結果が返って候補が復活するまで待機**する（待機にはタイムアウトを設ける）。

   ### Phase 2: Expansion
   - 選択ノードから伸長可能な Fragment をフィルタリング（制約条件に基づく）し、Legal Actionを決定する。
   - Legal Action が空の場合は **dead-end** として扱い、**即時に報酬 0 でBackpropagate**し、そのSimulationを終了する（バッチ評価は行わない）。
   - Fragment（行動）を選択する:
     - **UCTモード**: 未選択のFragmentがあれば**ランダムに1つ選択**。全て選択済みならUCTスコアで選択。
     - **PUCTモード**: `policy_logits` を Legal Actionでマスクしてsoftmaxし、PUCTスコアにより選択（または温度付きサンプリング）。
   - `hydrogen_replace` を用いて次状態候補リスト（ダミーアトム付きSMILES群）を生成し、次状態を選ぶ:
     - **UCTモード**: 次状態候補が複数ある場合は**ランダムに1つ選択**。
     - **PUCTモード**: 全候補をまとめて Value Network に入力し、各候補のV(s')を計算して選択（温度付き softmax サンプリング可）。
   - 選択した次状態で新ノードを生成・Tree に追加し、新ノードについて以下を評価する（この順序を推奨）:

     1) **SMILES正規化（canonicalize/sanitize 等）**  
        - 失敗した場合は `is_terminal = True` とし、**報酬 0 を確定**して即時Backpropagateし、Simulationを終了する。

     2) **Alert判定（状態ノード）**  
        - `alert_ok_elem(state_smiles) == 0` の場合は `is_terminal = True` とし、**Leaf_calc が ready であっても報酬は 0 で確定**して即時Backpropagateし、Simulationを終了する。

     3) **最大深さ判定**  
        - `depth >= max_depth` の場合は `is_terminal = True` とする。  
        - この場合の報酬は **0固定ではなく、報酬関数に通した値**（バッチ評価可）とする（Leaf評価条件に未到達でも評価してよい）。

     4) **Leaf評価条件に基づく `leaf_calc` 初期化**（最大深さでない場合）
        - Leaf評価条件（例: `depth >= min_depth`、かつ分子プロパティ下限 `HAC >= HAC_min` / `MW >= MW_min` など）を満たす場合: `leaf_calc = "ready"`
        - それ以外（下限未到達で評価対象外）: `leaf_calc = "not_ready"`

   - **`leaf_calc` と終端フラグに応じた処理**:
     - **terminal（Alert失敗/SMILES失敗）で報酬0が確定**した場合: 即時Backpropagateして終了（バッチキューには追加しない）。
     - `is_terminal == True` かつ（最大深さ到達などで）**報酬関数で評価する**場合:
       - Leaf SMILES（ダミー原子を水素で置換したSMILES）を生成し、評価待ちキューに追加する。
       - ノードの `leaf_calc` を `"pending"` に更新し、このSimulationを終了する。
       - 経路上の各エッジに `inflight += 1` を付与する（virtual loss）。
     - `leaf_calc == "ready"` の場合:
       - Leaf SMILES を生成し、評価待ちキューに追加する。
       - ノードの `leaf_calc` を `"pending"` に更新し、このSimulationを終了する。
       - 経路上の各エッジに `inflight += 1` を付与する（virtual loss）。
     - `leaf_calc == "not_ready"` の場合:
       - 評価は行わず、**同一Simulation内で**この新ノードを起点として Selection→Expansion を継続する（1回のSimulationで複数回の展開が発生し得る）。

3. **バッチ評価フラッシュ**
   - 評価待ちキューが `eval_batch_size` に到達 / `max_wait_ms` を超過 / `batch_eval_interval` に到達 / **探索が行き詰まった（候補が空）** のいずれかでバッチ評価を実行する。
   - 評価完了後、結果を反映して Backpropagation を実施する（6.7参照）。

---

## 6.2 MCTSNode クラス要件

`MCTSNode` は**状態を記録する**ことを主目的とし、以下の情報を保持する。

1. **状態情報**

   - `state_smiles`: ダミーアトムを含む SMILES（canonicalized）。
   - `depth`: ルートからの深さ（コアを depth=0 とする）。
   - `leaf_smiles`: ダミーアトムが水素原子で埋まったSMILES（重水素は含まないこととする）
   - `leaf_calc`: Leaf評価の状態を記録する。以下の**4つ**の状態を持つ:
     - `"not_ready"`: まだLeaf評価の条件（`depth >= min_depth`など）を満たしていない。**Selectionで到達した場合は展開を継続**。
     - `"ready"`: Leaf評価の条件を満たしているが、まだキューに追加されていない。**Selectionで到達した場合は評価待ちキューに追加して停止**。
     - `"pending"`: 評価待ちキューに追加済み。**Selection対象外（スキップ）**。
     - `"evaluated"`: すでにLeaf評価を実行済み。**Selectionで到達した場合は展開を継続**（再評価はしない）。
     - 状態遷移図は以下のとおりである
       - not_ready → ready (min_depth到達 & 分子プロパティ下限（例: HAC_min/MW_min）を満たす)
       - ready → pending (評価待ちキューに追加)
       - pending → evaluated (バッチ評価完了)

2. **統計量**

   - `visit_count (N)`: 累計訪問回数。**初期値: 0**
   - `total_reward (W)`: 累計報酬。**初期値: 0.0**
   - `q_value (Q)`: 平均報酬。**visit_count > 0 の場合は `total_reward / visit_count`、visit_count == 0 の場合は 0.0**（除算エラー回避）。
   - 必要に応じて、分散などの追加統計量を拡張可能とする。

3. **構造情報**

   - `parent`: 親ノードへの参照（root の場合は None）。**グラフ構造の参照用**であり、Backpropagationには使用しない。最初に到達した経路の親を保持。
   - `incoming_action`: このノードへの遷移に使用されたFragment SMILES（rootの場合はNone）。Policy_modelの学習データ抽出に使用。
   - `children`: 行動（Fragment）ごとの子要素を保持する辞書。各要素は少なくとも `(child_node, inflight(s,a), prior_P(s,a) [puct時])` を含む **Edge統計** を保持できる構造とする（virtual loss のため `inflight` が必要）。- `is_terminal`: bool。後続の展開が禁止されるかどうか。
  - `is_terminal=True` となる条件:
    - `alert_ok_elem(state_smiles) == 0`（Alert該当）
    - `depth >= max_depth`（最大深さ到達）
    - SMILES正規化（canonicalize/sanitize 等）に失敗
  - 報酬規則:
    - Alert該当 / SMILES失敗: 報酬は 0 で確定し、即時Backpropagate
    - 最大深さ到達: 報酬は報酬関数の出力（バッチ評価可）を用いる
   - `num_sub`: int。化学サブスペースの大きさ（提供関数で算出）。

4. **その他**

   - `metadata`: RDKit などで得られる補助情報のための任意フィールド（オプション）。
   - ノード ID として `(state_smiles, depth)` を用いたユニーク性管理を行う。

**要件:**

- 別経路から同一状態に到達する場合は、既存ノードを再利用し、新規にノードを作らない。
  - 状態のユニーク性は「canonical SMILES + depth」で判定する。
  - **親ノードは最初に到達した経路のものを維持**し、後から別経路で到達しても更新しない。
- `MCTSNode` は、UCT / PUCT のスコア計算に必要な値（Q, N, 親の N_total）が取得できるインターフェイスを持つ。
- **MCTSNodeは状態記録に専念**し、木構造の操作（子ノード追加、選択、展開）は`MCTSTree`が担当する。

---

### 6.3 MCTSTree クラス要件

`MCTSTree` は、**木構造全体の制御および探索の実行**を担う中心的なクラスである。

1. **基本機能**

   - `root` ノードを保持。
   - 全ノードを管理するインデックス `dict[(state_smiles, depth), MCTSNode]` を持ち、状態→ノードへの高速アクセスを可能にする。
   - **評価待ちキュー**を保持する。

2. **木構造制御メソッド**

   #### 2.1 `traverse(stop_condition) -> MCTSNode`

   - rootから、停止条件を満たすまで木を辿る。
   - Selection（UCT/PUCT）に基づいて子ノードを選択しながら進む。
   - **`leaf_calc == "pending"` のノードはスキップ**する。
   - 停止条件（以下のいずれか）:
     - `leaf_calc == "ready"` のノードに到達（評価待ちキューに追加される）
     - `is_terminal == True`
     - `depth >= max_depth`
   - **重要**: `leaf_calc == "not_ready"` のノードに到達した場合は、たとえそのノードが未展開であっても停止**しない**。そのノードを起点として行動選択＋展開を継続する（1回のSimulationで複数回の展開が発生し得る）。
   - 戻り値: 展開対象となるノード、または評価対象となるノード。

   #### 2.2 `expand(node, action) -> MCTSNode`

   - 指定されたノードに対して、actionで指定されたFragmentによる展開を実行。
   - 新しい子ノードを生成し、木に追加。
   - 既存ノードの場合は再利用（トランスポジション対応）。ただし**親は最初の経路のものを維持**。
   - 戻り値: 生成または再利用された子ノード。

   #### 2.3 `add_child(parent_node, child_node, action)`

   - 親ノードに子ノードを追加する。
   - 子ノードの`parent`と`incoming_action`を設定（最初の到達時のみ）。

   #### 2.4 `select_child(node, mode) -> Tuple[MCTSNode, action]`

   - UCTまたはPUCTスコアに基づいて、最も有望な子ノード（または未展開行動）を選択。
   - `mode`: `"uct"` または `"puct"`

   #### 2.5 `evaluate(nodes: list[MCTSNode]) -> list[float]`

   - 複数のノードに対して報酬を**バッチ評価**する。
   - 各ノードのLeaf SMILESを生成し、報酬関数を呼び出す。
   - 戻り値: 各ノードに対応する報酬のリスト。

   #### 2.6 `backpropagate(path: list[MCTSNode], reward: float)`

   - **Simulation中に記録された経路（path）を遡る**。parentは参照しない。
   - 更新内容:
     - `visit_count (N)` += 1
     - `total_reward (W)` += reward
     - `q_value (Q)` = W / N
   - `leaf_calc` を `"evaluated"` に更新（該当する場合）。

   #### 2.7 `backpropagate_batch(path_reward_pairs: list[Tuple[list[MCTSNode], float]])`

   - 複数のSimulationに対するBackpropagationを**バッチ処理**する。
   - 各Simulationで記録された経路（path）と報酬のペアを受け取る。
   - 同一経路上のノードが複数回更新される場合を適切に処理。

   #### 2.8 `search(n_simulations: int, batch_eval_interval: int)`

   - 指定した回数のSimulationを実行するメインループ。
   - **処理フロー**:

   ```python
   pending_items = []  # 評価待ちキュー: (node, path) のリスト

   for i in range(n_simulations):
       # Simulation中に経路を記録
       path = []  # List[MCTSNode]
       
       # Selection + Expansion ループ（leaf_calc == "ready" になるまで継続）
       node = root
       path.append(node)
       
       while True:
           # Selection: pending以外のノードを辿る
           node = traverse_from(node, stop_condition)
           path.append(node)

           # 終端（terminal）/ 最大深さの扱い
           # - Alert失敗 / SMILES正規化失敗: is_terminal=True となり、報酬0で即時Backpropagate
           # - 最大深さ到達（depth >= max_depth）: is_terminal=True とし、報酬関数の出力を用いる（バッチ評価可）
           if node.is_terminal:
               if node.depth >= max_depth:
                   # Leaf評価条件に未到達でも評価してよい（仕様）
                   pending_items.append((node, path))
                   node.leaf_calc = "pending"
               else:
                   backpropagate(path, 0.0)
               break

           if node.leaf_calc == "ready":
               # 評価対象に到達（経路も一緒に保存）
               pending_items.append((node, path))
               node.leaf_calc = "pending"
               break

           # leaf_calc == "not_ready" または "evaluated" の場合: 展開を継続
           # ("evaluated" は既に評価済みなので再評価せず、さらに深い探索を続ける)
           action = select_action(node, mode)
           new_node = expand(node, action)
           path.append(new_node)
           node = new_node  # 新ノードを起点に継続

       # バッチ評価フラッシュ判定（例）
       # - eval_batch_size 到達
       # - max_wait_ms 超過
       # - batch_eval_interval 到達（回数ベース補助）
       # - Selection不能（全候補pending等）で flush要求された
       if should_flush(pending_items, sim_i, last_flush_time, flush_requested):
           nodes = [item[0] for item in pending_items]
           paths = [item[1] for item in pending_items]

           rewards = evaluate_batch(nodes)  # バッチ推論
           release_virtual_loss(paths)      # inflight を減算
           backpropagate_batch(zip(paths, rewards))

           for n, _ in pending_items:
               n.leaf_calc = "evaluated"
           pending_items = []
           last_flush_time = now()
           flush_requested = False

   # 残りのノードを処理
   if pending_items:
       nodes = [item[0] for item in pending_items]
       paths = [item[1] for item in pending_items]
       rewards = evaluate(nodes)
       backpropagate_batch(zip(paths, rewards))
       for n, _ in pending_items:
           n.leaf_calc = "evaluated"
   ```

3. **検索・集計機能**

   - 条件付きでノードを抽出するメソッドを提供:
     - 例:
       - `total_reward >= R_min`
       - `num_sub >= S_min AND total_reward >= R_min`
       - `q_value >= Q_min`
     - 上記条件を AND/OR で組み合わせられるようなインターフェイスを用意する（詳細は設計で定義）。

4. **状態重複の取り扱い（トランスポジション）**

   - 同一状態（canonical SMILES＋depth）に対するノードは Tree 内で一意。
   - 別経路から到達した場合は**既存ノードを再利用**し、統計量を共有する。
   - **親ノードはグラフ構造の参照用**であり、最初に到達した経路のものを維持。後からの到達では親を更新しない。
   - **Backpropagation時は、そのSimulationで実際に通った経路（path）を遡る**。MCTSNode.parentは使用しない。

5. **学習データ抽出支援**

   - Value Head 学習用: 全ノードまたは条件を満たすノードから `(state_smiles, q_value)` ペアを抽出するメソッドを提供。
   - Policy Head 学習用: 子ノードを持つノードから `(parent_state_smiles, children_visit_counts)` を抽出するメソッドを提供。訪問回数からターゲット分布 `π = N(s,a) / Σ_a N(s,a)` を計算。
     - **`leaf_calc == "evaluated"` の子ノードのみ**が学習対象。

6. **シリアライズ支援**

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
   - 初期化時に、`fragment_featurizer` を用いて全 Fragment のグラフ表現を前計算し、Policy Modelへの補助入力として利用可能にする（オプション）。
   - **Fragment埋め込みベクトル**: 初期化時に全FragmentのK次元座標を計算・保持。**探索中は固定**。

2. **分子全体プロパティ評価**

   - `calc_mol_props` を内部から呼び出し、分子全体の
     - HAC
     - cnt_hetero
     - cnt_chiral
     - MW
     を返すヘルパーメソッドを提供。

3. **制約設定（探索空間制御 + Leaf評価下限）**

   - 分子全体の各プロパティに対して **下限・上限** を指定可能とする（例: `HAC_min/max`, `MW_min/max`, `cnt_hetero_min/max`, `cnt_chiral_min/max` など）。
   - **Legal Actionの事前フィルタ**:
     - 任意 Fragment を追加した結果が上限を超える場合、その Fragment はその状態での行動候補から除外する。
   - **Leaf評価下限（ready判定）**:
     - Simulation中に「評価して意味のあるサイズ/性質」に達していない状態は報酬計算対象外とし `not_ready` とする。
     - 代表例として `HAC >= HAC_min` や `MW >= MW_min` を用いる（必要に応じて他プロパティも下限条件に含められる）。

4. **Alert 判定**

   - 2種類のAlert判定メソッドを提供:
     - `alert_ok_elem(smiles)`: 状態（ダミーアトム付きSMILES）のAlert判定。新ノード生成時に呼び出し、0が返る場合（Alert該当）は `is_terminal=True` とする。
     - `alert_ok_mol(smiles)`: Leaf（ダミーアトムなしSMILES）のAlert判定。Leaf評価時に呼び出し、0が返る場合（Alert該当）は報酬を0とする。

5. **サブスペースサイズ**

   - `count_subspace(state_smiles)` を呼び出し `num_sub` を算出するメソッドを提供。
   - `MCTSNode` 生成時に呼び出し、`num_sub` を設定する。

6. **報酬評価（バッチ対応）**

   - Leaf（ダミーアトム無し SMILES）を受け取り、ユーザ提供の複数報酬関数を呼び出し、その相乗平均を返す:
     - N 個の報酬 `r_1, ..., r_N` に対し:
       \[
       R_{\text{geom}} = \exp\left(\frac{1}{N}\sum_{i=1}^{N}\log(\varepsilon + r_i)\right) - \varepsilon
       \]
   - \(\varepsilon\) は数値安定化のための微小定数（例: \(10^{-12}\)）。\(r_i=0\) を許容しつつ、アンダーフロー/NaNを回避する。
   - **バッチ評価API**: 複数のLeaf SMILESをリストで受け取り、報酬リストを返す。
     - `evaluate_batch(smiles_list: list[str]) -> list[float]`

---

### 6.5 Agent クラス要件

`Agent` は、**MLモデルの保持・学習に専念**するクラスである。**`torch.nn.Module` を継承**する。

**重要**: 探索の実行は `MCTSTree.search()` が担当する。Agentは探索ロジックを持たない。

1. **主要引数・パラメータ**

   - `max_depth`: 探索の最大深さ。これ以上は展開しない。
   - `min_depth`: この深さに達していなければ、Leaf 評価を行わず展開を継続する。
   - `mcts_mode`: `"uct"` または `"puct"`。
   - `c_uct`: UCT の探索定数（任意）。
   - `c_puct`: PUCT の探索定数。
   - **バッチ処理関連**:
     - `eval_batch_size`: 報酬評価（推論）をまとめて実行する際の評価バッチサイズ（例: 128〜512）。
     - `max_wait_ms`: 評価待ちキューの滞留を防ぐための最大待ち時間（ms）。超過したらflushする。
     - `flush_wait_timeout_ms`: 「全候補pending等でSelection不能」時にflush後、結果を待つ最大時間（ms）。
     - `batch_eval_interval`: 何シミュレーションごとにバッチ評価をflush判定するか（回数ベースの補助トリガー）。
     - `train_interval`: 何シミュレーションごとに学習を実行するか。
     - `batch_size`: 学習時のミニバッチサイズ。
     - `q_threshold_for_training`: 学習データ抽出時に使用する q_value の閾値。
   - 温度スケジューラ関連:
     - `tau_initial`, `tau_final`
     - `tau_scheduler_type`: `"linear"` または `"exponential"`

2. **保持する要素（nn.Moduleのサブモジュール）**

   - `policy_value_network`: PolicyValueNetwork（一体型、Policy HeadとValue Headを含む）
   - **注意**: Fragment Embeddingは不要（離散行動空間のため）

3. **学習データの抽出と学習（AlphaGoZero方式）**

   - 学習データは MCTSTree から直接抽出する。
   - `train_interval` シミュレーションごとに、現在の Tree からデータを抽出して学習を実行。
   - **Value Headの学習データ**:
     - 各ノードから `(state_smiles, value_target)` を抽出。
     - **Value target `z`**: Tree上のノードの `q_value`（平均報酬）。
   - **Policy Headの学習データ**:
     - 子ノードを持つノードから `(parent_state_smiles, children_actions, children_visit_counts)` を抽出。
     - **`leaf_calc == "evaluated"` の子ノードのみ**が対象。未評価ノードは含まない。
     - **ターゲット分布 `π`**: 子ノードの訪問回数から計算 `π(a) = N(s,a) / Σ_a N(s,a)`
   - **Loss関数（AlphaGoZero準拠）**:
     - `L = L_value + L_policy`
     - Value Loss: `L_value = (v - z)²` （v: Value Head出力, z: q_value）
     - Policy Loss: `L_policy = -Σ_a π(a) log p(a)` （クロスエントロピー）
       - `p(a)`: Policy Head出力（softmax後の確率）
       - `π(a)`: 訪問回数から計算されたターゲット分布

4. **PUCT 用 P(s,a) の定義（離散行動空間）**

   - PolicyValueNetworkが出力するpolicy_logitsから、Legal Actionsに対応するlogitを抽出。
   - softmaxで正規化して P(s,a) とする。
   - 計算式:

   ```text
   legal_logits = policy_logits[legal_action_indices]
   P(s, a) = softmax(legal_logits)
   ```

5. **V(s) による次状態選択**

   - `combine_smiles` → `hydrogen_replace` により複数の次状態候補 `{s'_i}` が生成された際、
     - 各候補に対して Value Head 出力 v(s'_i) を計算。
     - 温度付き softmax:
       \[
       p_i \propto \exp(V(s'_i)/\tau)
       \]
       に基づき 1 状態をサンプルし、次ノードとして採用。

6. **温度 τ のスケジューリング**

   - Agent は τ を更新するスケジューラを持つ。
   - 少なくとも以下 2 種類をサポート:
     1. 線形減衰: シミュレーション回数に応じて τ_initial → τ_final まで線形に減少。
     2. 指数減衰: τ = τ_initial *exp(-k* step)
   - スケジューラ種別は引数で切り替える。

7. **モデルの保存・読み込み**

   - PyTorch の state_dict を用いて、PolicyValueNetwork の重みを保存・読み込みできる API を提供。
   - 推論専用モードでは学習を行わず、読み込んだ重みで探索およびノード評価のみを行う。

---

### 6.6 UCT / PUCT 選択と Selection アルゴリズム（pending除外 + virtual loss 対応）

#### 6.6.1 共通ルール（UCT/PUCT共通）

- **pending除外**: `leaf_calc == "pending"` の子ノード（評価待ち）は Selection の候補から除外する。
- **virtual loss（in-flightペナルティ）**:
  - ready leaf を評価待ちキューへ投入して `pending` 化した時点で、その Simulation の経路上の各エッジ (s,a) に対して `inflight(s,a) += 1` を付与する。
  - 評価結果が返って Backpropagation するタイミングで、同じ経路上の各エッジに対して `inflight(s,a) -= 1` を行い、確定値（visit/価値）を更新する。
  - Selection では `inflight` を用いて **実効訪問回数**と**実効Q値**を計算し、同一経路への殺到を抑制する。
- **候補が空になった場合（全候補がpending等）**:
  - バッチ評価を **flush要求**し、評価結果が返って候補が復活するまで **待機**する（タイムアウト付き）。
  - 待機後も候補が空の場合、その Simulation は終了して次へ進む（ログ出力）。

#### 6.6.2 virtual loss を組み込んだ実効量

- `N(s,a)`: 確定した訪問回数（通常の visit_count）
- `W(s,a)`: 確定した累計報酬（通常の total_reward）
- `inflight(s,a)`: 評価待ち（未確定バックアップ）数
- `vloss`: virtual loss 係数（推奨: 1.0）

実効量を以下で定義する:

\[
N_{eff}(s,a) = N(s,a) + inflight(s,a)
\]

\[
Q_{eff}(s,a) = \frac{W(s,a) - vloss \cdot inflight(s,a)}{\max(1, N_{eff}(s,a))}
\]

#### 6.6.3 UCT モード

- 未展開（未選択）行動があれば **ランダムに1つ**選択して展開。
- 全行動が展開済みなら、virtual loss を含む `Q_eff`, `N_eff` を使って UCT スコアで選択する:

\[
\text{score}(s,a) = Q_{eff}(s,a) + c_{uct} \cdot \sqrt{\frac{\log(N_{parent}+1)}{1+N_{eff}(s,a)}}
\]

#### 6.6.4 PUCT モード（離散行動空間・AlphaGoZero方式）

- PUCT 式（virtual loss 対応版）:

\[
\text{score}(s,a) = Q_{eff}(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N_{parent}+1}}{1+N_{eff}(s,a)}
\]

- `P(s,a)` は PolicyValueNetwork の **policy_logits（Fragmentごとのlogit）** を Legal Action でマスクし、softmax（温度τ適用可）で正規化した確率を用いる。
- `leaf_calc == "pending"` の候補は除外する（重複評価の抑止）。


### 6.7 報酬評価とBackpropagation

1. **報酬評価のタイミング（バッチ処理・pendingでSimulation終了）**

   - Leaf評価は**即時実行ではなく、評価待ちキューに蓄積**される。
   - Expansionの結果、ノードが `leaf_calc == "ready"` に到達した場合:
     - Leaf SMILES（ダミー原子を水素で置換したSMILES）を生成し、評価待ちキューに追加する。
     - ノードの `leaf_calc` を `"pending"` に更新する。
     - **この時点で当該Simulationは終了**する（評価結果は後続のバッチ評価で確定し、確定後にBackpropagationする）。
   - **重複評価の禁止**: 同一ノードがすでに `leaf_calc == "pending"` の場合、同じLeafをキューへ再投入しない。
   - **バッチ評価のフラッシュ条件**（いずれかを満たしたら実行）:
     - `eval_batch_size` 件に到達した
     - `max_wait_ms` を超過した（時間ベース）
     - `batch_eval_interval`（Simulation数）に到達した（回数ベース）
     - **探索が行き詰まった**（Selectionにおいて「選択可能な候補が全てpending等で除外され、候補が空」になった）  
       → この場合は **flush を要求し、評価結果が返って候補が復活するまで待機**（待機にはタイムアウトを設ける）

   - 評価待ちノード（`leaf_calc == "pending"`）は **Selection対象外** とする。
   - pending中の探索停滞を避けるため、**virtual loss（in-flightペナルティ）** を導入する（6.6参照）。

2. **終端（terminal）・行き詰まり（dead-end）の扱い**

   **終端（terminal: `is_terminal=True`）になる条件と報酬規則**

   - **Alert失敗**: `alert_ok_elem(state_smiles) == 0` の場合、`is_terminal=True` とし、**Leaf評価条件や `leaf_calc` に関わらず報酬は 0 で確定**する。  
     → **即時に報酬 0 で Backpropagate**（バッチキューには追加しない）。
   - **SMILES正規化失敗**: canonicalize/sanitize 等に失敗した場合、`is_terminal=True` とし、**報酬は 0 で確定**する。  
     → **即時に報酬 0 で Backpropagate**（バッチキューには追加しない）。
   - **最大深さ到達**: `depth >= max_depth` の場合、`is_terminal=True` とする。  
     この場合、報酬は **0固定ではなく、報酬関数に通した値** を用いる（バッチ評価可）。Leaf評価条件に未到達でも評価してよい。  
     → `leaf_calc = "pending"` として **評価待ちキューに投入**し、評価確定後にBackpropagateする（virtual lossを適用）。

   **行き詰まり（dead-end）**

   - `leaf_calc == "not_ready"`（Leaf評価条件未到達）で、かつ **制約を満たす Fragment が存在しない（合法手が空）** 場合は dead-end とみなし、  
     → **即時に報酬 0 で Backpropagate** する（バッチキューには追加しない）。


3. **Backpropagationの更新対象**

   - Backpropagate時は、**そのSimulationで実際に通った経路（path）を遡る**。MCTSNode.parentは使用しない。
   - 各Simulationで経路を記録し、Backpropagation時にはその記録された経路を使用する。

---

### 6.8 状態重複（トランスポジション）への対応

- 同一の canonical SMILES（＋同じ depth）に到達する複数の経路が存在し得る。
- その場合は **別ノードを新規作成せず、既存ノードを使い回し**、訪問回数・報酬を統合する。
- **親ノードは最初に到達した経路のものを維持**。後から別経路で到達しても親は更新しない。
- depth が同じになることは Fragment が最小単位に分解されていることから保証される前提とする。

---

### 6.9 I/O とシリアライズ

1. **Tree の保存**

   - 探索完了後、`MCTSTree` の内容を容量効率の良い形式で保存する API を提供する。
   - 要件レベルではフォーマットは固定しないが、以下を満たすこと:
     - ノード数が多い場合でも、現実的なサイズに収まる（例: JSON よりもバイナリ形式が望ましい）。
     - 再ロード後もノード間の親子関係、統計量（N, W, Q, num_sub）を完全に再現できる。

2. **モデル重みの保存**

   - PUCT モードで学習した Policy Network, Value Network を、ファイルパスを受け取って保存・読み込み可能にする。

3. **設定の保存（オプション）**

   - 探索条件（制約、max_depth, min_depth, max_simulation 等）を再現可能な形で保存・読込できると望ましい（実装は任意）。

---

### 6.10 バッチ処理（報酬計算・モデル学習）

本システムでは、計算コストの高い処理（①報酬計算、②PyTorchモデル学習）を効率化するため、**バッチ処理**を導入する。

#### 6.10.1 設計思想

- **Time-consumingな処理**: 報酬計算（ユーザ提供のML推論）とモデル学習は時間がかかる。
- **バッチ化の方針**: これらの処理を `eval_batch_size` などでまとめて実行し、GPUの並列性を活かす。
- **pendingでSimulation終了**: `leaf_calc == "ready"` に到達したら評価待ちキューへ投入し `pending` 化して、そのSimulationは終了する（評価結果は後で確定）。
- **重複評価の禁止**: すでに `pending` のノードは再投入しない。
- **探索停滞の回避**:
  - pendingノードはSelection対象外
  - virtual loss（6.6）で同一経路への殺到を抑制
  - 全候補pending等でSelection不能になった場合は flush要求→待機（タイムアウト付き）

#### 6.10.2 評価待ちキュー

- `MCTSTree` が**評価待ちキュー**を保持する。
- Expansion時、`leaf_calc == "ready"` のノードに到達した場合:
  - Leaf SMILES（ダミー原子を水素で置換したSMILES）を生成し、ノード（または状態キー）をキューに追加。
  - `leaf_calc` を `"pending"` に更新し、**このSimulationを終了**する。
  - 経路上の各エッジ (s,a) に `inflight += 1` を付与する（virtual loss）。
- `leaf_calc == "pending"` のノードに再到達した場合:
  - **キューへ再投入しない**（重複評価を禁止）。
  - 必要に応じて、当該ノードに「Backprop待ちの経路情報（watchers）」を登録し、評価完了後にまとめてBackpropする。
- `leaf_calc == "pending"` のノードはSelectionでスキップされる（6.6）。

#### 6.10.3 バッチ評価の流れ

```text
1. キュー内のノードからLeaf SMILESリストを生成
2. Environment.evaluate_batch(smiles_list) を呼び出し
3. 各報酬関数をバッチで実行（GPUで並列処理）
4. log空間で相乗平均（数値安定化）を計算して報酬リストを返す
5. virtual loss を解除（経路上の inflight を減算）
6. バッチBackpropagationを実行（確定報酬を反映）
7. 各ノードの leaf_calc を "evaluated" に更新
```

#### 6.10.4 バッチBackpropagationの処理

バッチ評価後、複数のノードに対するBackpropagationを効率的に処理する。

```python
def backpropagate_batch(path_reward_pairs: list[Tuple[list[MCTSNode], float]]):
    # 各ノードへの更新を集計
    updates = defaultdict(lambda: {"reward_sum": 0.0, "visit_delta": 0})

    for path, reward in path_reward_pairs:
        # Simulation中に記録された経路（path）を遡る（parentは使用しない）
        for node in reversed(path):
            key = (node.state_smiles, node.depth)
            updates[key]["reward_sum"] += reward
            updates[key]["visit_delta"] += 1

    # 一括更新
    for key, delta in updates.items():
        node = get_node_by_key(key)
        node.visit_count += delta["visit_delta"]
        node.total_reward += delta["reward_sum"]
        node.q_value = node.total_reward / node.visit_count
```

#### 6.10.5 学習のバッチ処理

- `train_interval` シミュレーション後に、Treeから学習データを抽出してバッチ学習。
- バッチ評価と学習のタイミングを連携させることで効率化。
- 推奨フロー:

```text
バッチ評価 → Backpropagation → Tree更新 → 学習データ抽出 → バッチ学習
```

#### 6.10.6 パラメータ

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| `eval_batch_size` | 報酬評価（推論）をまとめて実行する評価バッチサイズ | 128〜512 |
| `max_wait_ms` | 評価待ちキューの最大滞留時間（時間ベースflush） | 50〜2000 |
| `flush_wait_timeout_ms` | 「全候補pending等でSelection不能」時にflush後、結果を待つ最大時間 | 100〜5000 |
| `batch_eval_interval` | 回数ベースのflush補助トリガー（Simulation間隔） | 128〜512 |
| `train_interval` | モデル学習を実行するSimulation間隔 | 256〜1024 |
| `batch_size` | 学習時のミニバッチサイズ | 64〜256 |
| `vloss` | virtual loss 係数（in-flightペナルティ） | 1.0 |

---

### 6.11 並列探索と木構造マージ

大規模並列でSimulationを実行し、得られた複数の木構造を統合する機能を提供する。

#### 6.11.1 目的

- **並列化による高速化**: 複数のプロセス/スレッドで独立にSimulationを実行。
- **木の統合**: 並列で得られた小さな木を合体させ、大きな木を形成。
- **探索効率の向上**: 統合された木を用いることで、既に探索済みの領域を再利用し、MCTS探索の効率を上げる。

#### 6.11.2 並列探索とモデル学習の全体フロー（4フェーズ）

```text
Phase 1: 並列探索（学習なし）
┌─────────────────────────────────────────┐
│  Worker 1 → Tree 1  (探索のみ)          │
│  Worker 2 → Tree 2  (探索のみ)          │
│  Worker N → Tree N  (探索のみ)          │
└─────────────────────────────────────────┘
           ↓
Phase 2: Tree マージ
           ↓
Phase 3: 学習データ抽出 & モデル学習
           ↓
Phase 4: 学習済みモデルで追加探索
```

**各フェーズの詳細**:

**Phase 1: 並列探索（学習なし）**

- 初期状態（コア構造）から複数のWorkerがそれぞれ独立に木を構築。
- 各Workerは `batch_eval_interval` に基づいてバッチ評価を実行。
- **この段階ではモデル学習を行わない**（UCTモードまたは初期Policy/Valueを使用）。

**Phase 2: Tree マージ**

- 全Workerの木をマージして1つの大きな木を形成。
- 統計量（visit_count, total_reward）を統合。

**Phase 3: 学習データ抽出 & モデル学習**

- マージされた木から学習データを抽出:
  - Policy Head用: 各ノードの訪問回数分布 `π = N(s,a) / Σ_a N(s,a)`
  - Value Head用: 各ノードの `q_value`
- AlphaGoZero方式でPolicyValueNetworkを学習。

**Phase 4: 学習済みモデルで追加探索**

- 学習済みモデルを使用してPUCTモードで追加探索を実行。
- 必要に応じてPhase 1-4を繰り返す。

#### 6.11.3 並列Simulationの実行モデル

```text
                    ┌─→ Worker 1 → Tree 1 ─┐
                    │                       │
Initial Tree → Fork ├─→ Worker 2 → Tree 2 ─┼─→ Merge → Merged Tree
                    │                       │
                    └─→ Worker N → Tree N ─┘
```

- 各Workerは独立に探索を実行（Phase 1）。
- 探索完了後、全Workerの木をマージ（Phase 2）。

#### 6.11.4 木マージ関数

**`merge_trees(trees: list[MCTSTree]) -> MCTSTree`**

複数の木構造を統合する関数。

**マージルール**:

1. **ノードの同一性判定**: `(canonical SMILES, depth)` が同一であれば同一ノードとみなす。

2. **統計量のマージ**:
   - `visit_count` = Σ visit_count_i （全木の訪問回数の合計）
   - `total_reward` = Σ total_reward_i （全木の累積報酬の合計）
   - `q_value` = total_reward / visit_count （再計算）

3. **子ノードのマージ**:
   - 各木に存在する子ノードを統合。
   - 同一の子ノードがある場合は統計量をマージ。
   - 異なる子ノードがある場合は両方を保持。

4. **親子関係**:
   - マージ後は**最初に処理した木の親子関係を優先**。

#### 6.11.5 増分マージ

**`merge_into(base_tree: MCTSTree, other_tree: MCTSTree) -> None`**

- `other_tree` の内容を `base_tree` にマージする（破壊的操作）。
- 探索中に他のWorkerの結果を取り込むために使用。

#### 6.11.6 マージ後の探索継続

- マージ後の木は通常の木と同様に探索を継続可能。
- マージにより訪問回数が増加した領域は、UCT/PUCTにより探索優先度が調整される。

#### 6.11.7 木の保存と分散処理

- 各Workerが生成した木は `MCTSTree.save()` で個別に保存可能。
- 後から複数の保存済み木を読み込んでマージすることも可能。

---

### 6.12 推論専用モード（Inference）

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

3. **利用イメージ**

   - 学習済みモデル＋既存 Tree を読み込む。
   - 必要に応じて追加探索を行い Tree を拡充（任意）。
   - 高期待報酬ノードのリストを出力し、ユーザが SMILES, num_sub, Q などを参照できる。

---

## 7. 非機能要件

1. **拡張性**

   - 報酬関数や Alert 判定関数の追加・入れ替えが容易であること。
   - Fragment テーブル列の追加にも耐えられる設計とする。
   - **モデル構成の拡張性**: PolicyValueNetworkのBackbone/Head構造を柔軟に差し替え可能。

2. **性能**

   - Fragment 数 ~1000、ノード数数万〜数十万規模の探索を想定。
   - シリアライズ／デシリアライズは現実的な時間で完了すること。
   - **バッチ処理の効果**: バッチサイズ128〜512でGPU利用効率を最適化。
   - **並列処理**: 複数Workerでの並列Simulationとマージが効率的に動作すること。

3. **テスト容易性**

   - 各クラス（Environment, Agent, MCTSTree, MCTSNode）を個別にユニットテスト可能な形で実装する。
   - combine_smiles, hydrogen_replace 等をダミー実装と差し替えてテストできること。
   - **バッチ処理のテスト**: バッチサイズ1でも正しく動作することを確認。

4. **ロギング・デバッグ**

   - シミュレーション進行状況（現在シミュレーション番号、Tree サイズ、平均報酬など）をロギングできる拡張ポイントを用意。
   - 重要イベント（Alert による terminal 判定、行き詰まりによる報酬 0 など）を識別可能にする。
   - **バッチ処理のログ**: バッチ評価の実行タイミング、処理件数を記録。

---

## 8. 公開インタフェースの例（イメージ）

※最終的なシグネチャは設計仕様書で確定。

- `Environment`:
  - `Environment(fragment_df, combine_fn, hydrogen_replace_fn, reward_fns, alert_elem_fn, alert_mol_fn, calc_props_fn, count_subspace_fn, constraints)`
  - `Environment.evaluate_batch(smiles_list: list[str]) -> list[float]`
  - `Environment.get_legal_actions(state_smiles: str) -> list[str]`

- `Agent(nn.Module)`:
  - `Agent(policy_value_network, config)`
  - `Agent.compute_action_probs(state_smiles: str, legal_actions: list[str]) -> Tensor[M]`
  - `Agent.compute_state_value(state_smiles: str) -> float`
  - `Agent.train_step(tree: MCTSTree) -> dict`  # Loss (L_value + L_policy)等を返す
  - `Agent.save(path)` / `Agent.load(path)`

- `MCTSTree`:
  - `MCTSTree(root_smiles: str, environment: Environment, agent: Agent, config)`
  - `tree.search(n_simulations: int, batch_eval_interval: int)`
  - `tree.traverse(stop_condition) -> MCTSNode`
  - `tree.expand(node, action) -> MCTSNode`
  - `tree.evaluate(nodes: list[MCTSNode]) -> list[float]`
  - `tree.backpropagate(path: list[MCTSNode], reward: float)`
  - `tree.backpropagate_batch(path_reward_pairs: list[Tuple[list[MCTSNode], float]])`
  - `tree.filter_nodes(q_min=None, total_reward_min=None, num_sub_min=None, depth_range=None) -> List[MCTSNode]`
  - `tree.extract_training_data(q_min=None) -> Dict`
  - `tree.save(path)` / `MCTSTree.load(path)`
  - `merge_trees(trees: list[MCTSTree]) -> MCTSTree`
  - `tree.merge_into(other_tree: MCTSTree) -> None`

---

以上が ECMPORL_02 パッケージの要件定義書 v5.1。
