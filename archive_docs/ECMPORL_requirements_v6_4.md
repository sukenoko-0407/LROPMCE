# ECMPORL 要件定義書 v6.4

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v6.4
- 作成日: 2025-12-12
- 位置づけ: 本書はリポジトリ構築の起点となる最重要ドキュメントであり、設計仕様書・実装は本書に準拠する。

### 1.1 v6.4での変更点（v6.3→v6.4）

- **depth_actionの変化タイミングを明記**: Action適用時に+1、Leaf→Branch遷移では不変と定義（セクション4.1）
- **Leaf→Branch選択のValue推論バッチ化を必須化**: 候補全体を1回の推論で処理する仕様を追加（セクション13.2.1）
- **並列Worker間の状態キャッシュ共有を追加**: 評価済みLeafのキャッシュ共有によるスループット向上オプションを定義（セクション16.3）

### 1.2 v6.3での変更点（v6.2→v6.3）

- **設定パラメータ一覧を追加**: 散在していたパラメータを一覧化（セクション19）
- **PolicyValueネットワーク仕様を追加**: グラフ入力・アーキテクチャ・出力仕様を明記（セクション20）
- **推論専用モード仕様を追加**: 高評価ノード抽出APIを定義（セクション21）
- **Fragmentライブラリ管理を明確化**: 既存CSVファイルの利用を明記（セクション7.2.2）
- **flush_intervalを削除**: flush条件をflush_thresholdのみに簡略化（セクション10.2）

### 1.3 v6.2での変更点（v6.1→v6.2）

- **MCTS統計の保持先を明確化**: 統計は原則 BranchNode に保持（状態統計 N(s),W(s) と行動統計 N(s,a),W(s,a),inflight(s,a) を BranchNode 内に集約）
- **探索空間制御の適用順序を変更**: 上限制約（HAC/MW/hetero/chiral）は BranchNode 側の legal action フィルタで保証し、LeafNode が上限超過しないことを不変条件として明記
- **legal action フィルタを必須化**: Branch の現在プロパティ＋fragment 増分プロパティ（Δ）テーブルに基づき、上限制約を満たす fragment のみを legal action とする

---

## 2. 概要

ECMPORL は MCTS により低分子化合物探索を行う Python パッケージである。

- コア構造（開始状態）から、Fragment（離散行動）を反復的に結合することで分子を成長させる
- 探索は **UCT / PUCT** の切替に対応する
- PUCT は AlphaGoZero 方式の **PolicyValue 一体型 Network** を前提とする
- **Leaf（ダミー無し分子）** を評価（報酬）対象とし、評価はバッチ化して高速化する
- **並列探索**（複数Worker）と **木マージ** によりスケールする
- 探索後に、Tree と学習済みモデルを保存し、推論専用モードで高期待値ノードを抽出可能とする

---

## 3. 用語定義

- **BranchNode（出発ノード）**: ダミーアトムを1つ含む canonical SMILES を状態とし、次のFragment（行動）選択を行うノード
- **LeafNode（着地ノード）**: ダミーアトムを含まない canonical SMILES を状態とし、報酬評価対象となるノード
- **Action**: BranchNode における Fragment 選択（離散行動、次元 K）
- **Leaf→Branch 遷移**: 1つの Leaf から `hydrogen_replace` により生成される複数の Branch 候補のうち、次に進む Branch を選ぶ処理
- **Simulation**: 1回の探索試行。開始から「Leaf を pending キューに投入（または終端確定）する」までを 1 simulation とする
- **pending**: Leaf の評価待ち状態（バッチ評価で後から value が確定する）
- **inflight / virtual loss**: pending の存在により同一枝への集中を抑えるための未確定カウンタ

---

## 4. 状態・一意性（トランスポジション）

状態は canonical SMILES と depth（後述）の組で一意に定義し、トランスポジション表で共有する。

- `leaf_key = (canonical_leaf_smiles, depth_action)`
- `branch_key = (canonical_branch_smiles, depth_action)`

> 注: depth は「Action を何回適用したか」を表す **depth_action** とする。

### 4.1 depth_action の更新タイミング（必須）

`depth_action` の更新は以下のタイミングで行う。

- **Action適用時（combine_smiles）**: `depth_action += 1`
  - BranchNode から Action を選択し、`combine_smiles` で LeafNode を生成した時点で depth が1増加する
  - 生成された LeafNode の `depth_action` は親 BranchNode の `depth_action + 1` となる

- **Leaf→Branch遷移時**: `depth_action` は **変化しない**
  - `hydrogen_replace` で生成される BranchNode 候補は、元の LeafNode と同じ `depth_action` を持つ
  - これは「同一深度内での分岐位置選択」を意味する

**例**: 初期 BranchNode（depth=0）から探索を開始した場合

```text
Branch(depth=0) --[Action]--> Leaf(depth=1) --[hydrogen_replace]--> Branch(depth=1) --[Action]--> Leaf(depth=2) ...
```

---

## 5. Node 定義（クラス分離）

### 5.1 LeafNode（着地）

LeafNode は必ず `combine_smiles` で生成される（行動適用の着地点）。

**必須フィールド**
- `leaf_smiles: str`（ダミー無し canonical）
- `depth_action: int`
- `leaf_calc: Literal["not_ready","ready","pending","done"]`
- `is_terminal: bool`
- `value: float | None`（done 時に確定値）
- `children_branches: list[branch_key]`（hydrogen_replace から作られる候補）
- `parent_ref: Any`（参照用、探索ロジックでは利用しない）

### 5.2 BranchNode（出発）

BranchNode は `hydrogen_replace(leaf_smiles)` の結果から生成される。

**必須フィールド**
- `branch_smiles: str`（ダミー1個を含む canonical）
- `depth_action: int`
- `is_terminal: bool`
- `parent_ref: Any`（参照用）

### 5.3 MCTS統計（BranchNodeに保持）

MCTS の統計は **原則として BranchNode に保持**する（概念としては「状態 s の統計」および「s における各行動 a の統計」）。
LeafNode は統計を持たず、**当該 Leaf の確定報酬（leaf_value）**を保持する。

- **状態統計（BranchNode）**
  - `N(s)`: 確定訪問回数
  - `W(s)`: 確定累計報酬
  - `Q(s) = W(s) / max(1, N(s))`: 平均報酬（保持しても計算してもよい）
- **行動統計（BranchNode 内の action_stats）**
  - `N(s,a)`: 確定訪問回数
  - `W(s,a)`: 確定累計報酬
  - `inflight(s,a)`: 評価待ち（未確定）数（virtual loss 用）
  - `Q_eff(s,a), N_eff(s,a)`: virtual loss を加味した実効量（定義は本書の式に準拠）

補足:
- 「エッジ統計」という意味では `N(s,a), W(s,a), inflight(s,a)` を用いるが、**実装上は BranchNode が保持する**（独立した Edge オブジェクトは必須ではない）。

---

## 6. 主要関数（環境・SMILES操作・Alert）

### 6.1 SMILES操作（必須）

- `combine_smiles(branch_smiles_with_dummy: str, frag_smiles_no_dummy: str) -> str`
  - ダミー位置へ fragment を結合し、**ダミー無し（Leaf）**のSMILESを返す（正規化は内部で実施してよい）
- `hydrogen_replace(leaf_smiles_no_dummy: str) -> list[str]`
  - Leaf に対して、次の付加位置（ダミー）を導入した候補（**ダミー1個付き**）を返す
  - 空リストを返し得る（Leaf-only / dead-end）

### 6.2 Alert 判定（必須・2系統）

Alert 判定は2種類存在し、基準も適用対象も異なるため両方必須とする。

- `alert_ok_elem(smiles_with_dummy: str) -> int`（1=OK, 0=NG）
  - **BranchNode生成時**（または到達時）に適用する
  - 0 の場合、その BranchNode は `is_terminal=True` とする
  - 実装上は「その Branch を生成しない（候補から除外）」でもよい
- `alert_ok_mol(leaf_smiles_no_dummy: str) -> int`（1=OK, 0=NG）
  - LeafNode に適用する
  - 0 の場合、その Leaf の **最終報酬は 0** と確定し、他の報酬関数は呼ばなくてよい

### 6.3 分子全体プロパティ計測（必須）

- `measure_mol_props(leaf_smiles_no_dummy: str) -> dict`
  - 例: `{"HAC": int, "MW": float, "cnt_hetero": int, "cnt_chiral": int}`

---

## 7. 探索空間制御（分子全体制約）

探索空間制御のため、分子全体プロパティに対して以下の制約を設定可能とする。

- HAC: `HAC_min`, `HAC_max`（下限・上限）
- MW: `MW_min`, `MW_max`（下限・上限）
- cnt_hetero（N,O,S,P）: `hetero_max`（上限）
- cnt_chiral: `chiral_max`（上限）

### 7.1 制約の適用タイミング（上限制約は Branch 側で保証）

分子全体制約は **上限制約（max）** と **下限制約（min）** を分けて扱う。

- **上限制約（HAC_max / MW_max / hetero_max / chiral_max）**  
  - BranchNode で legal action を生成する際に **必ずフィルタ**し、上限制約を満たす fragment のみを legal action とする（詳細は 7.2）。  
  - したがって、legal action のみを選択して `combine_smiles` を適用する限り、**LeafNode が上限を超過することは起こらない**（不変条件）。
  - 防御的実装として、LeafNode 生成後に上限超過が検出された場合は「到達不能状態」とみなし、`is_terminal=True`・報酬0・ログ記録としてよい。

- **下限制約（HAC_min / MW_min）**  
  - 下限は「探索を継続するか否か」ではなく、**Leaf の ready 判定（評価対象かどうか）**にのみ影響させる。
  - `HAC < HAC_min` または `MW < MW_min` の場合、その Leaf は原則 `leaf_calc="not_ready"` とし、`hydrogen_replace` により Branch 候補を生成して探索を継続する。
  - ただし `depth >= max_depth` の場合は、下限未到達でも評価対象として `leaf_calc="ready"` としてよい（探索打ち切りのため）。

LeafNode 生成時には `measure_mol_props(leaf_smiles)` を実行し、LeafNode に保持する（7.2 の Branch 側フィルタでも同種のプロパティが必要）。

### 7.2 Legal Action フィルタ（必須）

BranchNode における legal action（fragment）候補は、**分子全体の上限制約を満たすものだけ**に制限する。
行動選択は legal action に対してのみ行われるため、これにより **LeafNode が上限超過することを構造的に防ぐ**。

#### 7.2.1 BranchNode 側の現在プロパティ
BranchNode は、現在の分子状態に対する分子全体プロパティを保持または計算できること。
（計数対象は「ダミー原子を除いた実分子部分」とし、RDKit等で一貫した定義を採用する。）

- `mol_props_branch = measure_mol_props(branch_equivalent_leaf_smiles)`  
  - `branch_equivalent_leaf_smiles` は「BranchNode のダミーを除去（/置換）して得られる、現在の実分子のSMILES」を指す（実装詳細は任意だが定義を一貫させる）。

#### 7.2.2 Fragment プロパティテーブル（必須）
fragment（行動）ごとに、上限制約フィルタに用いるプロパティを **テーブルとして事前保持**する。すでにCSVファイルが用意されているので、それをDataFrameとして読み込む。カラムはaction_id, frag_smiles, hac, mw, cnt_hetero, cnt_chiralが存在する。

- `frag_props[action_id] = {ΔHAC, ΔMW, Δhetero, Δchiral}`  
  - ここで `Δ*` は **当該 fragment を結合したときに分子全体へ加算される増分**として定義する。

#### 7.2.3 フィルタ規則（上限制約の保証）
BranchNode における各 fragment 候補 `a` に対し、以下を満たすもののみを legal とする。上記7.2.2のDataFrameの全体に対してFilterを適用することでLegal Actionの選定が可能である。

- `ΔHAC(a) <= HAC_max - mol_props_branch.HAC`
- `ΔMW(a)  <= MW_max - mol_props_branch.MW`
- `Δhetero(a) <= hetero_max - mol_props_branch.cnt_hetero`
- `Δchiral(a) <= chiral_max - mol_props_branch.cnt_chiral`

このフィルタにより、legal action のみを選択して `combine_smiles` を実施する限り、生成される LeafNode の分子全体プロパティは上限制約を超えない（7.1 の不変条件）。

---

## 8. Simulation 定義とフロー（最重要）

### 8.1 Simulation の定義（一本化）

- **1 Simulation は `leaf_calc == "ready"` の Leaf を `pending` キューに投入するまで**（または終端確定まで）を指す
- 途中で `leaf_calc == "not_ready"` の Leaf が出た場合、**同一 Simulation 内で Branch を選択して次の Action へ進む**（複数回展開してよい）

### 8.2 基本フロー（PUCT/UCT共通）

1. **Selection（BranchNode）**: UCT/PUCT で Action を選択
2. **Expansion（Action適用）**:
   - `combine_smiles` で LeafNode を生成
   - `alert_ok_mol` / 下限制約（ready判定）/ max_depth / 正規化失敗 を判定（上限制約は 7.2 の legal action フィルタで保証）
3. **Leaf 処理**:
   - 終端確定（報酬0確定など）なら即時Backpropして Simulation 終了
   - `leaf_calc` を決定（ready / not_ready / pending / done）
   - ready の場合: pending へ投入し Simulation 終了
   - not_ready の場合: `hydrogen_replace` で Branch 候補を生成し、Leaf→Branch 選択を実施して次へ
4. **Batch 評価（flush）**: pending が一定条件で flush され、評価結果が確定
5. **Backpropagate**: 確定値に基づき、path に沿って統計量（N,W）を更新

---

## 9. Leaf の評価状態 `leaf_calc`

`leaf_calc` は LeafNode に付与する。

- `not_ready`: まだ評価対象ではなく、探索を継続する（Leaf→Branch 遷移が可能な場合）
- `ready`: 評価対象（pending キューへ投入可能）
- `pending`: 評価待ち（同一 leaf_key は再投入しない）
- `done`: 評価済み（value が確定）

### 9.1 ready 判定（max_depth でない場合）

以下を満たす場合に `leaf_calc="ready"` とする（例示。設計で固定する）。

- `depth_action >= min_depth`
- `HAC >= HAC_min` かつ `MW >= MW_min`
- （上限制約は 7.2 の legal action フィルタで保証される）

それ以外は `leaf_calc="not_ready"`。

### 9.2 max_depth 判定

- `depth_action >= max_depth` の場合は `is_terminal=True` とする
- この場合の報酬は **0固定ではなく、報酬関数に通した値（バッチ可）**とする  
  （ready 条件に未到達でも評価してよい）

---

## 10. pending / バッチ評価

### 10.1 pending_items（dictで一意化）

- `pending_items: dict[leaf_key, PendingEntry]`
- `PendingEntry` は少なくとも以下を持つ:
  - `leaf: LeafNode`
  - `waiters: list[PathToken]`（評価結果の反映が必要な経路情報）
  - `enqueued_at: float`

> 同一 `leaf_key` は **評価を1回に一意化**し、再到達時は `waiters` に追加するのみ。

### 10.2 flush 条件

- `len(pending_items) >= flush_threshold`
- Selection の結果「候補が全て pending で空」になった場合は **flush要求し待機**（タイムアウト付き）

### 10.3 pending Leaf の Selection 除外

BranchNode の Action 選択において、
既に展開済みの Action が指す子 Leaf が `leaf_calc=="pending"` の場合、その Action は候補から除外（スコア = -∞）する。

---

## 11. virtual loss（inflight）と実効量

- `N(s,a)`: 確定 visit
- `W(s,a)`: 確定 total reward
- `inflight(s,a)`: pending 起因の未確定数
- `vloss`: virtual loss 係数（推奨: 1.0、設定可能）

実効量:

- `N_eff(s,a) = N(s,a) + inflight(s,a)`
- `Q_eff(s,a) = (W(s,a) - vloss * inflight(s,a)) / max(1, N_eff(s,a))`

pending に投入した時点で、Simulation が通過した path 上の各エッジに対して `inflight += 1` を行う。
評価確定時に対応する `inflight -= 1` を行い、その後に Backpropagate を適用する。

---

## 12. UCT / PUCT（BranchNode の Action 選択）

### 12.1 UCT

- 未展開（未訪問）行動があれば **ランダムに1つ**選択して展開
- 全行動が展開済みなら UCT スコア最大を選択

（virtual loss は `Q_eff/N_eff` を用いる）

### 12.2 PUCT（AlphaGoZero）

PUCT スコア:

`score = Q_eff(s,a) + c_puct * P(s,a) * sqrt(N_total + 1) / (1 + N_eff(s,a))`

#### 12.2.1 legal action マスク + softmax（温度）

- policy_logits は常に固定次元 K
- 状態ごとに legal action が変わるため、illegal 行動は確率0にする必要がある

手順（必須）:

1. `legal_indices` を取得
2. `legal_logits = policy_logits[legal_indices] / tau_policy`（温度 `tau_policy`）
3. `P_legal = softmax(legal_logits)`  
4. `P(s,a)` は legal のみで定義し、illegal の prior は 0 とする

---

## 13. Leaf→Branch 選択（A案: 次Branchは1つに決める）

Leaf `L` に対して `hydrogen_replace(L.leaf_smiles)` が返す候補集合を `B(L)` とする。

- 候補生成後、各候補 Branch について `alert_ok_elem` を適用し、NG は除外する
- 除外後に候補が空なら Leaf-only として終端（報酬は max_depth 等の規則に従う）

Leaf→Branch 遷移もエッジ統計（N/W/inflight）を持つ。

### 13.1 UCT モード（未訪問優先＝ランダム）

- 未訪問 Branch があれば、未訪問集合からランダムに1つ選ぶ
- そうでなければ UCT スコア最大を選ぶ

### 13.2 PUCT モード（未訪問優先＝prior最大）

Leaf→Branch の prior は、Branch 候補それぞれに対して PVネットの **value head** を用いて定義する。

- 各 `b ∈ B(L)` について `V(b)` を推論（候補数は平均10程度を想定）
- `P_leaf(L,b) = softmax(V(b) / tau_branch)`（温度 `tau_branch`）

選択規則:

- 未訪問 Branch があれば、未訪問集合の中で `P_leaf` 最大を選ぶ
- そうでなければ、PUCT スコア最大を選ぶ（`P` に `P_leaf` を用いる）

#### 13.2.1 Value推論のバッチ化（必須）

Leaf→Branch 選択時の Value 推論は、**候補全体を1回のバッチ推論で処理**することを必須とする。
候補ごとに個別推論を行うと計算コストが線形に増加し、ボトルネックとなるため。

**実装要件**:

```python
def compute_branch_priors(leaf: LeafNode, branches: list[BranchNode], model: PolicyValueNet) -> dict[branch_key, float]:
    """
    Branch候補全体を1回のバッチ推論でValue評価し、priorを計算する。

    Args:
        leaf: 元のLeafNode
        branches: hydrogen_replaceで生成されたBranch候補リスト
        model: PolicyValueネットワーク

    Returns:
        branch_key -> P_leaf(L,b) のマッピング
    """
    # 1. 候補SMILESをバッチ化
    batch_smiles = [b.branch_smiles for b in branches]

    # 2. 1回のforward passで全候補のValueを取得
    values = model.batch_value(batch_smiles)  # shape: (len(branches),)

    # 3. softmax で prior を計算
    P_leaf = softmax(values / tau_branch)

    return {b.branch_key: p for b, p in zip(branches, P_leaf)}
```

**性能要件**:

- `batch_value` は内部で PyTorch Geometric の `Batch.from_data_list` を使用し、グラフをバッチ化すること
- GPU使用時は候補グラフを一度にGPUへ転送し、複数回の転送オーバーヘッドを回避すること

---

## 14. 報酬（Leaf評価）

### 14.1 報酬関数インタフェース

- `reward_funcs: list[Callable[[leaf_smiles_no_dummy], float]]`
- 各関数は原則として `[0,1]` のスカラーを返す（違反時の扱いは 14.3）

### 14.2 相乗平均（N個の報酬を統合）

報酬ベクトル `r_i (i=1..N)` を用いて最終報酬 `R` を定義する。

- **log空間で計算し、εで log(0) を回避**する（必須）

`ε = 1e-12`（設定可能）

`R = exp( (1/N) * Σ_i log(max(ε, r_i)) )`

### 14.3 異常系ポリシー（必須）

- 例外（Exception）: 当該 `r_i = 0` とみなし、ログに記録
- `NaN / inf`: 当該 `r_i = 0` とみなし、ログに記録
- 範囲外（`r_i < 0` または `r_i > 1`）: `clip(r_i,0,1)` し、ログに記録
- タイムアウト: 当該 `r_i = 0` とみなし、ログに記録
- バッチ評価で 1件が異常でも、バッチ全体を落とさず継続する

### 14.4 Alert と報酬の優先順位

- `alert_ok_mol == 0` の Leaf は **最終報酬0に確定**し、報酬関数は呼ばなくてよい

---

## 15. 学習（AlphaGoZero方式）

PUCT モードでは PolicyValue 一体型 Network を制御し、探索結果から学習データを抽出して学習可能であること。

### 15.1 学習データ

- Policy target: `π(a) = N(s,a) / Σ_a N(s,a)`（BranchNode の子行動分布）
- Value target: `z`（例: そのノードの `q_value` = W/N、またはバックアップにより確定した平均値）

### 15.2 Loss（必須）

- `L = L_value + L_policy`
- `L_value = (v - z)^2`
- `L_policy = - Σ_a π(a) * log p(a)`（legal のみ）

### 15.3 学習支援関数（要件）

少なくとも以下の関数群を提供する。

- `extract_training_samples(tree) -> Dataset`
  - (state_smiles_with_dummy, π, z) を抽出
- `train_policy_value_model(dataset, model, optimizer, train_config) -> model`
- `save_checkpoint(model, path)`
- `load_checkpoint(path) -> model`

---

## 16. 並列探索と木マージ

### 16.1 並列探索

- `run_worker(config, seed) -> MCTSTree`
  - Worker ごとに独立に探索を実行し、Tree を返す

### 16.2 木マージ（必須）

- `merge_trees(trees: list[MCTSTree]) -> MCTSTree`

**マージルール（必須）**
- ノード同一性: key（leaf_key / branch_key）が一致すれば同一ノード
- 統計量: `N` と `W` は加算で統合する
- `inflight` は「探索実行後の確定木」では通常 0 を期待する（マージ対象に残る場合は加算してよい）
- `leaf_calc` の統合優先度:
  - `done` > `pending` > `ready` > `not_ready`
- 親参照（parent_ref）は **参照用**であり、マージ時は「最初に採用されたものを保持」でよい
  （探索・Backprop の正当性は **path** により担保する）

### 16.3 評価キャッシュ共有（オプション）

複数Workerが独立に探索する場合、同一状態を重複評価する無駄が発生し得る。
スループット向上のため、**評価済みLeafのキャッシュを Worker 間で共有**するオプションを提供する。

#### 16.3.1 キャッシュ構造

```python
# 共有キャッシュ（プロセス間共有またはRedis等の外部ストア）
evaluation_cache: dict[leaf_key, CachedEvaluation]

@dataclass
class CachedEvaluation:
    value: float           # 確定した報酬値
    mol_props: dict        # 分子プロパティ
    evaluated_at: float    # 評価時刻（キャッシュ有効期限管理用）
```

#### 16.3.2 キャッシュ利用フロー

1. **LeafNode生成時**: `leaf_key` でキャッシュを参照
2. **キャッシュヒット時**: 評価をスキップし、キャッシュ値を即座に使用（`leaf_calc="done"`）
3. **キャッシュミス時**: 通常通り pending キューへ投入
4. **評価完了時**: キャッシュに結果を登録

#### 16.3.3 実装上の注意

- **スレッドセーフ**: 複数Workerからの同時アクセスに対応すること
- **メモリ管理**: キャッシュサイズ上限（`cache_max_size`）を設定可能とし、LRU等で古いエントリを削除
- **一貫性**: キャッシュヒット時も統計量（N, W）の更新は正しく行うこと

#### 16.3.4 設定パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| enable_eval_cache | bool | False | 評価キャッシュ共有の有効化 |
| cache_max_size | int | 100000 | キャッシュエントリ数上限 |
| cache_backend | str | "memory" | キャッシュバックエンド（"memory" / "redis"） |

> 注: 初期実装では `enable_eval_cache=False`（無効）をデフォルトとし、性能問題が顕在化した場合に有効化を検討する。

---

## 17. ロギング / テスト（要件）

- ロギング（最低限）:
  - flush回数、pending滞留時間、dead-end（Leaf-only）比率
  - Alert NG 件数（elem / mol 別）
  - 異常系（例外/NaN/clip/timeout）件数
- テスト（最低限）:
  - Leaf-only（hydrogen_replace空）
  - pending 再到達（同一 leaf_key の待ち合わせ）
  - NaN報酬・例外報酬・clip
  - Alert elem / mol 失敗
  - max_depth 終端（報酬0固定でない）

---

## 18. 受け入れ基準（抜粋）

- LeafNode / BranchNode の責務分離が実装に反映されていること
- `alert_ok_elem` と `alert_ok_mol` が両方実装され、適用箇所が仕様通りであること
- 相乗平均に ε が導入され、log(0) 事故が起きないこと
- `vloss` の設定が可能で、デフォルト推奨値 1.0 を持つこと
- `pending_items` が dict 化され、同一 leaf_key の評価が 1 回に一意化されること
- illegal 行動が prior 0 となるよう、legal mask + softmax（温度）が実装されること
- 学習データ抽出・学習ループ・チェックポイントの関数群が提供されること
- 並列探索と木マージ（merge_trees）が提供されること
- HAC/MW の下限/上限、hetero/chiral 上限が探索空間制御として機能すること

---

## 19. 設定パラメータ一覧

設定パラメータの一覧は下表のとおり。一番右列の値は参考値であり、実装では引数として指定できるようにする。  

| パラメータ | 型 | デフォルト/推奨 | 説明 | 参考値 |
|-----------|-----|----------------|------|------|
| min_depth | int | 3 | 最小探索深度（ready判定） | 9.1 |
| max_depth | int | 10-12 | 最大探索深度（終端判定） | 9.2 |
| c_puct | float | 1.0 | PUCT探索定数 | 12.2 |
| tau_policy | float | 1.0 | Policy softmax温度 | 12.2.1 |
| tau_branch | float | 1.0 | Leaf→Branch選択温度 | 13.2 |
| vloss | float | 1.0 | Virtual loss係数 | 11 |
| flush_threshold | int | 128-512 | Flushトリガー件数 | 10.2 |
| HAC_min | int | - | 重原子数下限 | 7 |
| HAC_max | int | - | 重原子数上限 | 7 |
| MW_min | float | - | 分子量下限 | 7 |
| MW_max | float | - | 分子量上限 | 7 |
| hetero_max | int | - | ヘテロ原子数上限 | 7 |
| chiral_max | int | - | キラル中心数上限 | 7 |
| ε (epsilon) | float | 1e-12 | 相乗平均のlog(0)回避 | 14.2 |
| K | int | 1000-2000 | アクション空間サイズ（Fragment数） | 12.2.1 |
| max_workers | int | 32 | 最大並列Worker数 | 16.1 |
| enable_eval_cache | bool | False | 評価キャッシュ共有の有効化 | 16.3.4 |
| cache_max_size | int | 100000 | キャッシュエントリ数上限 | 16.3.4 |
| cache_backend | str | "memory" | キャッシュバックエンド | 16.3.4 |

---

## 20. PolicyValueネットワーク仕様

### 20.1 入力（状態表現）

SMILESからグラフ表現（PyTorch Geometric形式）に変換して入力とする。

**ノード特徴量（8種）**

- atom_type, atom_degree, atom_implicit_valence, atom_hybridization
- atom_formal_charge, atom_is_aromatic, atom_is_in_ring, atom_chiral_tag

**エッジ特徴量（4種）**

- bond_type, bond_is_conjugated, bond_is_in_ring, bond_stereo

変換関数: `get_graph_data(smiles) -> Data`（参照実装: example_code/featurizer.py）

### 20.2 アーキテクチャ

- **GNNバックボーン**: ChemProp または AttentiveFP（他のGNNも可）
- **hidden dimension**: 128 または 256
- **層数/fru_num**: 2-3
- **グラフ集約**: AttentiveFPの標準方式（Attention Pooling）

### 20.3 出力

- **policy_logits**: shape (K,) - Fragmentアクション空間（K=1000-2000）
- **value**: shape (1,) - [0,1]のスカラー

### 20.4 設計方針

- PolicyとValueは共有バックボーンから分岐する一体型構造（AlphaGoZero方式）
- ダミー原子（*）を含むBranchNodeのSMILESを入力とする
- 報酬関数は疎結合で入れ替え可能とする

---

## 21. 推論専用モード

### 21.1 目的

学習済みTreeから高評価ノードを抽出し、有望な分子候補を取得する。

### 21.2 API

- `extract_top_leaves(tree, **criteria) -> list[LeafResult]`

### 21.3 抽出条件（criteria）

以下の条件を組み合わせて指定可能とする（全てオプション）:

- `top_k: int` - 上位k件を抽出
- `value_threshold: float` - value >= threshold のノードを抽出
- `visit_threshold: int` - N(s) >= threshold のノードを抽出

### 21.4 出力形式（LeafResult）

- `leaf_smiles: str`
- `value: float`
- `visit_count: int`
- `depth: int`
- `mol_props: dict` (HAC, MW, cnt_hetero, cnt_chiral)
