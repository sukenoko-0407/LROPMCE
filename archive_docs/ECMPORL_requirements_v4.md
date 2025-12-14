# ECMPORL 要件定義書 v4.0

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v4.0
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
- **Simulation（シミュレーション）**: MCTSの1回の反復。Selection → Expansion → （バッチ評価後）Backpropagation の流れ。**1回のSimulationで1ノードを展開**する。
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

**重要**: 1回のSimulationで**1ノードのみ展開**する。バッチ処理のため、報酬計算は遅延実行される。

1. **初期状態**:
   - コア構造（ダミーアトム付き SMILES）を入力し、根ノード `root` を作成。

2. **Simulation（1回）**: 以下の3フェーズからなる。

   **Phase 1: Selection**
   - UCT または PUCT により、root から子ノードを辿り、展開対象ノードを選択。
   - **評価待ちノード（`leaf_calc == "pending"`）はSelection対象外**とする。

   **Phase 2: Expansion**
   - 選択ノードから伸長可能な Fragment をフィルタリング（制約条件に基づく）し、Legal Actionを決定。
   - **UCTモード**: 未選択のFragmentがあれば**ランダムに1つ選択**。全て選択済みならUCTスコアで選択。
   - **PUCTモード**: Policy Networkから(mean, std_log)を取得し、各Legal Actionの確率を計算して選択。
   - 選択された Fragment と現在の状態を `combine_smiles` で結合。
   - `hydrogen_replace` を用いて次状態候補リスト（ダミーアトム付きSMILES群）を生成。
   - **次状態選択**:
     - **UCTモード**: 次状態候補が複数ある場合は**ランダムに1つ選択**。
     - **PUCTモード**: 全候補を**まとめて** Value Network に入力し、各候補のV(s')を計算。温度付き softmax サンプリングで 1 状態を選択。
   - 選択した次状態で新ノードを生成・Tree に追加。
   - 新ノードについて以下を実施:
     - `alert_ok_elem` による Alert 判定 → 0が返る場合（Alert該当）は `is_terminal = True`
     - `depth >= max_depth` の場合も `is_terminal = True`
     - `num_sub` 計算
     - `leaf_calc` の初期化:
       - `depth >= min_depth` かつ分子プロパティが下限を満たす場合: `leaf_calc = "ready"`
       - それ以外: `leaf_calc = "not_ready"`
   - **`leaf_calc` に応じた処理**:
     - `leaf_calc == "ready"`: **評価待ちキューに追加**し、`leaf_calc = "pending"` に更新。
     - `leaf_calc == "not_ready"`: 評価は行わず、**このノードを起点として行動選択＋展開を継続**する（1回のSimulationで複数回の展開が発生し得る）。

   **Phase 3: （バッチ評価後）Backpropagation**
   - バッチ評価トリガー時に実行（後述の6.10参照）。
   - Leaf で得た報酬（合計報酬）を root までの経路上のノードに逆伝播。

3. 上記シミュレーションを `max_simulation` 回繰り返す。

4. **バッチ評価トリガー**: `batch_eval_interval` 回のSimulation後にバッチ評価とBackpropagationを実行。

---

### 6.2 MCTSNode クラス要件

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
       - not_ready → ready (min_depth到達 & 下限制約満たす)
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
   - `children`: 子ノードの辞書またはリスト。キーは行動（Fragment SMILES）＋次状態SMILESなど、設計で定義。
   - `is_terminal`: bool。後続の展開が禁止されるかどうか。
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

           if node.is_terminal or node.depth >= max_depth:
               # 行き詰まり: 即時報酬0でBackpropagate（経路を遡る）
               if node.depth < min_depth:
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

       # バッチ評価トリガー
       if len(pending_items) >= batch_eval_interval:
           nodes = [item[0] for item in pending_items]
           paths = [item[1] for item in pending_items]
           rewards = evaluate(nodes)
           backpropagate_batch(zip(paths, rewards))
           for n, _ in pending_items:
               n.leaf_calc = "evaluated"
           pending_items = []

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

3. **制約設定**

   - 分子全体の各プロパティに対して **下限・上限** を指定可能とする。
     - 例: `HAC_min`, `HAC_max` など。
   - 各状態で有効な Fragment を列挙する際に、これら制約を用いて事前フィルタリングを行う:
     - 任意 Fragment を追加した結果が上限を超える場合、その Fragment はその状態での行動候補から除外。

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
       R_{\text{geom}} = \left(\prod_{i=1}^{N} r_i\right)^{1/N}
       \]
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
     - `batch_eval_interval`: 何シミュレーションごとにバッチ評価を実行するか（128〜512を想定）。
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

### 6.6 UCT / PUCT 選択と Selection アルゴリズム

1. **UCT モード**

   - **未選択行動の優先**: 有効行動の中で、まだ子ノードが存在しないFragmentがあれば、**ランダムに1つ選択**して展開。
   - **全行動が選択済みの場合**: UCTスコアに基づいて子ノードを選択。
   - UCT スコアの具体式:
     \[
     \text{score}(s,a) = Q(s,a) + c_{\text{uct}} \cdot \sqrt{\frac{\log(N_{\text{parent}}+1)}{1+N_{child}}}
     \]
   - **注意**: `visit_count == 0` の場合、`Q = 0` として計算（除算エラー回避済み）。

   ```python
   def select_action_uct(node, legal_actions, c_uct):
       # 1. 未選択の行動があれば優先（ランダムに1つ選択）
       untried = [a for a in legal_actions if a not in node.children]
       if untried:
           return random.choice(untried)

       # 2. 全て選択済みなら UCT スコアで選択
       best_score = -inf
       best_action = None
       N_parent = node.visit_count
       for action, child in node.children.items():
           if child.leaf_calc == "pending":
               continue  # 評価待ちノードはスキップ
           Q = child.q_value
           N_child = child.visit_count
           score = Q + c_uct * sqrt(log(N_parent + 1) / (1 + N_child))
           if score > best_score:
               best_score = score
               best_action = action
       return best_action
   ```

2. **PUCT モード（離散行動空間対応・AlphaGoZero方式）**

   - PUCT 式:
     \[
     \text{score}(s,a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N_{\text{parent}}+1}}{1+N_{child}}
     \]
   - P(s,a) は PolicyValueNetworkのPolicy Headが出力するlogitをsoftmaxで正規化した確率を用いる。
   - Q(s,a) は Tree 上の Q 値（累計報酬／訪問回数）を用いる。`visit_count == 0` の場合は `Q = 0`。
   - **評価待ちノード（`leaf_calc == "pending"`）はスキップ**する。

---

### 6.7 報酬評価とBackpropagation

1. **報酬評価のタイミング（バッチ処理）**

   - Leaf評価は**即時実行ではなく、評価待ちキューに蓄積**される。
   - `batch_eval_interval` で指定された数のSimulationが完了した時点でバッチ評価を実行。
   - 評価待ちノードは `leaf_calc = "pending"` となり、**Selection対象外**。

2. **行き詰まりの扱い**

   - `depth < min_depth` かつ以下のいずれかの場合:
     - 制約を満たす Fragment が存在しない。
     - `alert_ok_elem` により `is_terminal=True` となった（0が返った場合）。
   - 上記の場合は、**即時に報酬 0 で Backpropagate** する（バッチキューには追加しない）。

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

- **Time-consumingな処理**: 報酬計算とモデル学習はMLモデルを通すため時間がかかる。
- **バッチ化の方針**: これらの処理を一定数（128〜512程度）まとめて実行し、GPUの並列性を活かす。
- **評価待ちノードのSelection除外**: バッチ評価完了までそのノードはSelection対象外とし、同一ノードへの繰り返し選択を防ぐ。

#### 6.10.2 評価待ちキュー

- `MCTSTree` が**評価待ちキュー**を保持する。
- Expansion時、`leaf_calc == "ready"` のノードに到達した場合:
  - Leaf SMILESを生成し、ノードをキューに追加。
  - `leaf_calc` を `"pending"` に更新。
  - **即時評価は行わない**。
- `leaf_calc == "pending"` のノードはSelectionでスキップされる。

#### 6.10.3 バッチ評価の流れ

```text
1. キュー内のノードからLeaf SMILESリストを生成
2. Environment.evaluate_batch(smiles_list) を呼び出し
3. 各報酬関数をバッチで実行（GPUで並列処理）
4. 相乗平均を計算して報酬リストを返す
5. バッチBackpropagationを実行
6. 各ノードの leaf_calc を "evaluated" に更新
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
| `batch_eval_interval` | バッチ評価を実行するSimulation間隔 | 128〜512 |
| `train_interval` | モデル学習を実行するSimulation間隔 | 256〜1024 |
| `batch_size` | 学習時のミニバッチサイズ | 64〜256 |

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

以上が ECMPORL_02 パッケージの要件定義書 v4.0。
