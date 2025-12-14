# ECMPORL 要件定義書 v6.1

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v6.1
- 作成日: 2025-12-12
- 位置づけ: 本書はリポジトリ構築の起点となる最重要ドキュメントであり、設計仕様書・実装は本書に準拠する。

### 1.1 v6.1での変更点（v6.0→v6.1）

v6.0 の **LeafNode / BranchNode 分離** 等の方針を維持しつつ、v5.1 で重要だった視点を以下の通り復活・明確化する。

- Alert 判定を **alert_ok_elem（Fragment/ダミー付き）** と **alert_ok_mol（Leaf/ダミー無し）** の2系統で規定
- 報酬の相乗平均における **ε（log(0)回避）** を明記
- virtual loss 係数 **vloss（推奨 1.0）** を明記
- **並列探索**を支援する関数群、および **木マージ（merge_trees）** を要件に追加
- PUCT の prior 計算における **legal action マスク + softmax（温度）** を明確化
- AlphaGoZero 方式（PolicyValue一体型）に基づく **学習のための関数群**（データ抽出・損失・学習ループ）を要件化
- 探索空間制御のための分子全体制約を明確化:
  - **HAC/MW: 下限・上限**
  - **cnt_hetero/cnt_chiral: 上限**

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

---

## 5. Node 定義（クラス分離）

### 5.1 LeafNode（着地）

LeafNode は必ず `combine_smiles` で生成される（行動適用の着地点）。

**必須フィールド（例）**
- `leaf_smiles: str`（ダミー無しSMILES）
- `depth_action: int`
- `leaf_calc: Literal["not_ready","ready","pending","done"]`
- `is_terminal: bool`
- `value: float | None`（done 時に確定値）
- `children_branches: list[branch_key]`（hydrogen_replace から作られる候補）
- `parent_ref: Any`（参照用、探索ロジックでは利用しない）

### 5.2 BranchNode（出発）

BranchNode は `hydrogen_replace(leaf_smiles)` の結果から生成される。

**必須フィールド（例）**
- `branch_smiles: str`（ダミー1個を含むSMILES）
- `depth_action: int`
- `is_terminal: bool`
- `parent_ref: Any`（参照用）

### 5.3 エッジ統計（共通）

MCTS の統計はエッジ（遷移）に保持する（Node 内に格納してもよいが、概念としてはエッジ）。

- `N(s,a)`: 確定訪問回数
- `W(s,a)`: 確定累計報酬
- `inflight(s,a)`: 評価待ち（未確定）数

---

## 6. 主要関数（環境・SMILES操作・Alert）

### 6.1 SMILES操作（必須）

- `combine_smiles(branch_smiles_with_dummy: str, frag_smiles_no_dummy: str) -> str`
  - ダミー位置へ fragment を結合し、**ダミー無し（Leaf）**のSMILESを返す
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

### 7.1 制約違反の扱い

LeafNode 生成時に `measure_mol_props` を実行し、以下を適用する。

- 上限超過（HAC>HAC_max, MW>MW_max, hetero>hetero_max, chiral>chiral_max）:
  - `is_terminal=True` とし **報酬0確定**で即時Backprop、Simulation終了
- 下限未到達（HAC<HAC_min, MW<MW_min）:
  - **探索は継続可能**（leaf_calc の ready 判定にのみ影響）
  - ただし `depth >= max_depth` の場合は評価対象としてよい（後述）

### 7.2 Legal Action の事前フィルタ（任意だが推奨）

BranchNode の legal action（fragment）候補に対して、
「結合後に上限を確実に超えると分かる fragment」を事前に除外してよい。
（近似・保守的でよい。除外し過ぎないこと。）

---

## 8. Simulation 定義とフロー（最重要）

### 8.1 Simulation の定義（一本化）

- **1 Simulation は `leaf_calc == "ready"` の Leaf を `pending` キューに投入するまで**（または終端確定まで）を指す
- 途中で `leaf_calc == "not_ready"` の Leaf が出た場合、**同一 Simulation 内で Branch を選択して次の Action へ進む**（複数回展開してよい）

### 8.2 基本フロー（PUCT/UCT共通）

1. **Selection（BranchNode）**: UCT/PUCT で Action を選択
2. **Expansion（Action適用）**:
   - `combine_smiles` で LeafNode を生成
   - `alert_ok_mol` / 制約 / max_depth / 正規化失敗 を判定
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
- （上限違反は 7.1 により既に終端）

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

### 10.2 flush 条件（例）

- `len(pending_items) >= flush_threshold`
- `flush_interval` 経過
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
