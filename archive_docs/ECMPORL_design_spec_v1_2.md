# ECMPORL 設計仕様書 v1.2

- 対象要件: **ECMPORL 要件定義書 v6.5（2025-12-12）**
- 作成日: 2025-12-13
- 目的: 要件定義書に基づき、実装可能な粒度で「モジュール構成・データ構造・アルゴリズム・I/F・永続化・テスト方針」を定義する。

---

## 1. 全体方針

### 1.1 ゴール
ECMPORL は、**BranchNode（ダミー付きSMILES）を状態**として MCTS を行い、Action（Fragment）で Leaf（ダミー無しSMILES）を生成し、Leaf を報酬評価する探索エンジンである。

- UCT / PUCT を切替可能
- Leaf 評価は **pending キュー + バッチ評価（flush）**でスループットを稼ぐ
- pending があるため、**virtual loss（inflight）**で同一枝集中を抑制する
- 並列 worker で木を作り **merge_trees** で統合する

### 1.2 非ゴール（本仕様の範囲外）
- `combine_smiles` / `hydrogen_replace` の化学的正当性の詳細（RDKit 実装は別途）
- Alert ルールの具体（SMARTS等）
- どの報酬関数を採用するかの科学的議論（I/F のみ定義）

---

## 2. リポジトリ / パッケージ構成

### 2.1 ディレクトリ案

```
ecmporl/
  __init__.py
  config.py
  types.py
  nodes.py
  tree.py
  transposition.py

  smiles/
    __init__.py
    ops.py              # combine_smiles, hydrogen_replace, canonicalize
    props.py            # measure_mol_props
    alerts.py           # alert_ok_elem, alert_ok_mol（ユーザ実装を差し替え可能）

  fragments/
    __init__.py
    library.py          # CSV読込、frag_props, K, action_id整合性
    legal.py            # legal action フィルタ（Δテーブル×上限制約）

  mcts/
    __init__.py
    engine.py           # run_search（単一プロセス）
    simulation.py       # 1 simulation の実装（8章の一本化）
    selection.py        # UCT/PUCT
    leaf_to_branch.py   # 13章（Leaf→Branch選択）
    backprop.py         # 確定値の反映
    pending.py          # pending_items, flush, evaluator 呼び出し
    path.py             # PathToken / waiters

  eval/
    __init__.py
    reward.py           # 相乗平均、異常系ポリシー
    evaluator.py        # leaf batch evaluator（同期/並列）

  model/
    __init__.py
    featurizer.py       # get_graph_data（PyG Data）
    pvnet.py            # PolicyValueNet（Torch/PyG）
    inference.py        # batch_policy_value / batch_value

  train/
    __init__.py
    dataset.py          # extract_training_samples
    trainer.py          # train_policy_value_model
    checkpoint.py       # save/load

  parallel/
    __init__.py
    worker.py           # run_worker
    merge.py            # merge_trees
    cache.py            # evaluation_cache（オプション）

  infer/
    __init__.py
    extract.py          # extract_top_leaves

  utils/
    __init__.py
    log.py
    timer.py
    rng.py

cli/
  ecmporl_cli.py        # search/train/infer サブコマンド
tests/
  test_pending.py
  test_vloss.py
  test_merge.py
  test_leaf_only.py
  ...
```

### 2.2 依存関係（推奨）
- 必須: `numpy`, `pandas`
- 化学: `rdkit`
- 学習: `torch`, `torch-geometric`（extra: `ecmporl[torch]`）
- 形式: `pydantic` は任意（dataclass で十分なら不要）

---

## 3. 設定（Config）設計

### 3.1 Configクラス
`dataclasses.dataclass` で以下を定義し、YAML/JSON からロード可能にする。

- `SearchConfig`
  - `algorithm: Literal["uct","puct"]`
  - `min_depth, max_depth`
  - `c_puct, tau_policy, tau_branch`
  - `vloss`
  - `flush_threshold`
  - `max_simulations`（or time budget）
  - `seed`
  - `value_target_mode: Literal["A","A_prime"]`
- `ConstraintConfig`
  - `HAC_min, HAC_max, MW_min, MW_max, hetero_max, chiral_max`
- `FragmentLibraryConfig`
  - `frag_csv_path`, `K_expected: Optional[int]`
- `ParallelConfig`
  - `max_workers`
  - `enable_eval_cache`, `cache_backend`, `cache_max_size`
- `RewardConfig`
  - `epsilon`
  - `reward_timeout_sec`
  - `reward_parallelism`（thread/process）
- `ModelConfig`
  - `checkpoint_path`, `checkpoint_id`
  - `hidden_dim`, `num_layers`, `gnn_type`

### 3.2 既定値
既定値は要件の推奨に合わせる（`vloss=1.0`, `tau_policy=1.0`, `tau_branch=1.0` 等）。  
※ 要件の「参考値」は探索用途の例であり、設計側では「デフォルト/推奨」を採用。

---

## 4. データモデル

### 4.1 Key 型
- `LeafKey = tuple[str, int]`  # (canonical_leaf_smiles, depth_action)
- `BranchKey = tuple[str, int]` # (canonical_branch_smiles, depth_action)

### 4.2 LeafNode
`LeafNode` は統計を持たず、状態・評価状態・確定報酬を保持する。

```python
@dataclass
class LeafNode:
    leaf_smiles: str
    depth_action: int
    leaf_calc: Literal["not_ready","ready","pending","done"]
    is_terminal: bool
    value: float | None
    mol_props: dict[str, float|int]  # HAC/MW/cnt_hetero/cnt_chiral（生成時に測定）
    children_branches: list[BranchKey]  # hydrogen_replace 結果（not_ready時に埋める）
    # 以下は参照用
    parent_ref: Any | None = None
```

### 4.3 BranchNode と統計
`BranchNode` が **状態統計 + 行動統計**を保持する。

```python
@dataclass
class ActionStats:
    N: int = 0
    W: float = 0.0
    inflight: int = 0
    child_leaf: LeafKey | None = None  # 展開済みなら指す

@dataclass
class BranchNode:
    branch_smiles: str
    depth_action: int
    is_terminal: bool
    mol_props_branch: dict[str, float|int]  # branch_equivalent_leaf_smiles から測定
    legal_actions: np.ndarray  # int array of legal action_id
    priors_legal: np.ndarray | None  # float array aligned with legal_actions（PUCT時）
    N: int = 0
    W: float = 0.0
    action_stats: dict[int, ActionStats] = field(default_factory=dict)
    parent_ref: Any | None = None
```

#### 4.3.1 Q_eff / N_eff の計算
- `N_eff = N + inflight`
- `Q_eff = (W - vloss * inflight) / max(1, N_eff)`

計算は都度関数で算出（保持してもよいが、更新整合性を優先して都度が無難）。

---

## 5. Tree / トランスポジション

### 5.1 MCTSTree
トランスポジション表を中心に管理する。

```python
@dataclass
class MCTSTree:
    checkpoint_id: str
    root: BranchKey
    branches: dict[BranchKey, BranchNode]
    leaves: dict[LeafKey, LeafNode]
    # 学習データ抽出用（A mode）にシミュレーション記録を残す場合
    sim_records: list["SimulationRecord"] = field(default_factory=list)
```

### 5.2 一意性
- `canonical_smiles` は RDKit 等で正規化した文字列（ダミー有無に応じて別関数で）
- `depth_action` を key に含める

---

## 6. Fragment ライブラリと legal action

### 6.1 FragmentLibrary
CSV を DataFrame として読み込み、以下を提供する。

- `K`（action数）
- `frag_smiles[action_id]`
- `delta_props[action_id] = (dHAC, dMW, dHetero, dChiral)`  # 増分

`action_id` は **0..K-1 の連番（欠番なし）**を前提にする。欠番・重複を検出した場合は `ValueError` とし、入力CSVの修正を要求する。

### 6.2 legal action フィルタ
`BranchNode` 生成/到達時に、branch 側の現在プロパティ `mol_props_branch` と `delta_props` を用いて上限制約を満たす action のみを `legal_actions` として保持する。

---

## 7. Simulation（探索ループ）設計

### 7.1 Simulation の状態機械
要件の一本化定義に従う。

- 1 simulation は **ready の Leaf を pending に投入するまで**（または終端確定まで）
- not_ready が出たら **同一 simulation 内で Leaf→Branch を選んで継続**

### 7.2 PathToken
pending に投入した Leaf の評価結果が確定したとき、どの経路に backprop するかを保持する。

```python
@dataclass(frozen=True)
class PathToken:
    edges: tuple[tuple[BranchKey, int], ...]  # (branch_key, action_id) の列
    leaf_key: LeafKey
```

### 7.3 Simulation 実装（疑似コード）

```
path = []
b = root_branch
while True:
  a = select_action(b)         # UCT or PUCT（pending除外）
  leaf = expand(b, a)          # combine_smiles -> LeafNode
  path.append((b.key, a))

  if leaf.is_terminal and leaf.value is not None:
      backprop_immediate(path, leaf.value)
      break

  if leaf.leaf_calc == "ready":
      enqueue_pending(leaf, PathToken(path, leaf.key))
      break

  branches = generate_branches(leaf)  # hydrogen_replace + alert_ok_elem
  if branches empty:
      value = finalize_leaf_only(leaf)
      backprop_immediate(path, value)
      break

  b = select_next_branch(leaf, branches)  # 13章
  continue
```

---

## 8. Expansion（Leaf生成）と leaf_calc 決定

### 8.1 例外/失敗の扱い（設計固定）
以下は終端として扱い、`value=0` を確定させる。

- SMILES 正規化失敗（canonicalizeで例外）
- `alert_ok_mol == 0`（最終報酬0）
- leaf_only かつ `depth_action < min_depth`（展開不能かつ min_depth 未到達）

`max_depth` は **報酬0固定ではなく**、通常の reward 経路（バッチ可）へ流す。

### 8.2 ready / not_ready 判定
要件例を設計で固定する（可変にする場合は関数化）。

- `depth_action >= min_depth` かつ `HAC >= HAC_min` かつ `MW >= MW_min` なら ready
- それ以外は not_ready
- `depth_action >= max_depth` は `is_terminal=True` かつ ready 扱い（評価するため）

---

## 9. pending / flush / evaluator

### 9.1 pending_items
要件通り dict で一意化する。

```python
@dataclass
class PendingEntry:
    leaf: LeafNode
    waiters: list[PathToken]
    enqueued_at: float
```

- 同一 `leaf_key` に再到達した場合は `waiters.append(...)` のみ

### 9.2 pending投入時の virtual loss
投入時に `path.edges` を走査して `inflight += 1`。確定時に対称に `inflight -= 1`。

### 9.3 flush 条件
- `len(pending_items) >= flush_threshold`
- Selection が「候補が全て pending で空」になった場合は flush 要求して待機（タイムアウト）

### 9.4 evaluator I/F
報酬計算は batch 単位で呼び出す。

```python
class LeafBatchEvaluator(Protocol):
    def evaluate(self, leaf_smiles_list: list[str]) -> list[float]:
        ...
```

実装は 2 系統用意:
- `SyncEvaluator`: 逐次で reward_funcs を回す
- `ProcessPoolEvaluator`: timeout付きで並列化（1件失敗でも継続）

### 9.5 確定時の処理
確定値 `R` を得たら:

1. Leaf を `leaf_calc="done"`, `value=R` に更新
2. `inflight -= 1`（経路上）
3. `backprop(path, R)` で `N/W` を更新

---

## 10. Selection（UCT / PUCT）

### 10.1 共通
- 対象は `legal_actions` のみ
- 展開済み action が指す child leaf が `pending` の場合、候補から除外（score=-inf）

### 10.2 UCT（標準：PUCT寄せ）

- **未展開行動があれば**：未展開集合から 1 つ選択する  
  - **採用**: `seed` 固定の擬似乱数で **一様サンプリング**（再現性 + 探索バイアス低減）  
  - 参考: デバッグで完全決定性が必要な場合のみ `action_id` 最小を選ぶ `tie_break_mode="min_action_id"` を用意してよい

- **全て展開済みなら**：以下の UCT スコアを最大化する（pending child は除外）
  - 定義（確定統計 / completed のみ）  
    - `Q(s,a) = W(s,a) / max(1, N(s,a))`  （PUCTのQと同じ: 平均価値）  
    - `N_total(s) = sum_{a∈legal} N(s,a)` （PUCTと同じ定義）  
    - `score(s,a) = Q(s,a) + c_uct * sqrt( ln(N_total(s) + 1) / (1 + N(s,a)) )`
  - `ln` は自然対数（底 *e*）。未展開は前段で処理するため、ここでは `N(s,a) >= 1` を前提としてよい

- **virtual loss を考慮する場合（推奨）**：selection 時のみ `eff` 統計を使う（PUCTと同じ思想）
  - `N_eff(s,a) = N(s,a) + inflight(s,a)`  
  - `W_eff(s,a) = W(s,a) - vloss * inflight(s,a)`（vloss>0 を前提）  
  - `Q_eff = W_eff / max(1, N_eff)`  
  - `N_total_eff(s) = sum_{a∈legal} N_eff(s,a)`  
  - `score_eff(s,a) = Q_eff + c_uct * sqrt( ln(N_total_eff + 1) / (1 + N_eff) )`
  - backprop による確定更新は **completed 統計のみ**を更新する（inflight は flush 完了で解消）

- **係数 `c_uct` のデフォルト**（初期実装の推奨）  
  - **`c_uct_default = 1.0`** とする（PUCTの `c_puct` と揃えて差分を減らす）  
  - 報酬スケールが将来変わる可能性があるため、最終的には `SearchConfig.c_uct` で調整可能にする



### 10.3 PUCT
- score は要件の式を使用
- N_total は `sum(N(s,a) for a in legal)`（確定visitのみ）
- 未展開 Q_eff 初期値 = 0.0
- tie-break: action_id 最小（決定的）

#### 10.3.1 prior の生成
PVネットの `policy_logits`（shape K）から

1. `legal_indices` 抽出
2. `policy_logits[legal]/tau_policy`
3. `softmax -> P_legal`
4. `BranchNode.priors_legal` に保持（legal_actions と同じ順）

---

## 11. Leaf→Branch 選択（A案）

### 11.1 Branch候補生成
`hydrogen_replace(leaf_smiles)` の結果を canonicalize し、各候補に `alert_ok_elem` を適用。NGは除外。

候補が空なら **leaf_only 終端**（これ以上展開できない）。

- `depth_action < min_depth` の場合：報酬評価は行わず `value=0` を確定（`leaf_calc="skipped"` 等の理由コードを付与）
- `depth_action >= min_depth` の場合：通常の leaf 判定（ready / not_ready）に従う。ready なら評価キューへ、not_ready なら `value=0` を確定


### 11.2 UCTモード
未訪問 Branch があれば未訪問集合からランダム。

### 11.3 PUCTモード（value head prior）
- 候補 Branch の `V(b)` を **バッチ推論**で取得
- `P_leaf = softmax(V(b)/tau_branch)` を prior とする
- 未訪問があれば prior 最大、それ以外は PUCT 最大（`P=P_leaf`）

---

## 12. Backpropagate

### 12.1 更新対象
- `BranchNode.N/W`（状態統計）
- `ActionStats.N/W`（行動統計）

更新は「評価確定時のみ」行う（pending投入時は inflight のみ増える）。

### 12.2 実装
`backprop(path: PathToken, R: float)`:

```
for (b_key, a) in path.edges:
  b = tree.branches[b_key]
  b.N += 1; b.W += R
  s = b.action_stats[a]
  s.N += 1; s.W += R
```

---

## 13. 学習データ抽出 / 学習ループ

### 13.1 目的
探索結果 Tree から (state, π, z) を作り、PolicyValueNet を教師ありで更新する。

### 13.2 π（policy target）
各 BranchNode s について:

- `π(a) = N(s,a) / Σ_a N(s,a)`（legal のみ）

### 13.3 z（value target）
`value_target_mode` で分岐。

- `"A"`: per-simulation MC  
  - SimulationRecord を保存し、確定報酬 R が付いたら、その simulation が通過した状態に対して `z=R`
- `"A_prime"`: per-node mean MC  
  - 各状態 s で `z = W(s)/N(s)`（同一 s からは原則1サンプル）
  - `sample_weight = N(s)` を付与可能

### 13.4 Dataset 形式
- `state_smiles_with_dummy: str`（BranchNode の SMILES）
- `pi: np.ndarray`（legal に対応する確率列）
- `z: float`
- `legal_actions: np.ndarray`
- `sample_weight: float`（任意）

### 13.5 Loss
- `L = (v - z)^2 + -Σ π(a) log p(a)`（legal のみ）
- sample_weight を適用可能

---

## 14. 並列探索 / 木マージ

### 14.1 run_worker
`run_worker(config, seed) -> MCTSTree` は単一プロセス探索を行い、終了時に flush を完了させ inflight をゼロに近づける。

### 14.2 merge_trees
ルール:

- key 一致で同一ノード
- N/W は加算
- leaf_calc は `done > pending > ready > not_ready`
- inflight は通常 0 を期待（残っていれば加算）
- checkpoint_id の一致を検証し、異なるなら警告（運用上はマージ禁止）

---

## 15. 推論専用モード（extract）

推論（探索結果の利用）では **良い Leaf を列挙する**というより、**良い Branch（= 遷移 (s,a)）を見つける**ことが重要、という方針に合わせる。

`extract_top_leaves(tree, top_k=None, value_threshold=None, visit_threshold=None)`

- 対象: `leaf_calc="done"` の **子 Leaf** を持つ **Branch遷移 (s,a)**（= 親BranchNode + action_id）
  - つまり返す単位は「Leaf」ではなく「Leafへ到達した Branch（エッジ）」である
- `visit_count` の定義（確定統計 / completed のみ）  
  - **`visit_count = N(s,a)`**（その Branch遷移が選択された回数）  
  - virtual loss は *推論ランキングの visit_count には含めない*（= inflight は無視）
- `value` は、当該遷移で到達した **子Leafの value（Leaf_calc done の報酬）**を用いる
- フィルタ  
  - `value_threshold`: `value >= threshold`  
  - `visit_threshold`: `visit_count >= threshold`
- 出力（例）:  
  - `BranchResult(parent_smiles, action_id, frag_smiles, child_smiles, value, visit_count, depth, mol_props)`
    - `parent_smiles`: 親Branchの状態
    - `frag_smiles`: 当該 action の fragment
    - `child_smiles`: 遷移先（Leaf）の状態
    - `depth`: child の depth（`depth_action`）
    - `mol_props`: child の分子全体プロパティ

備考:
- transposition / merge が有効な場合、同一 child_smiles が複数の親から到達し得る。その場合でも本関数は **(parent, action)** 単位で結果を返す（Branch探索を主目的とするため）。


---

## 16. 永続化（推奨）

### 16.1 Tree 保存
- `save_tree(path)` / `load_tree(path)`
- 形式は以下のどちらか:
  - `pickle`（簡単だが互換性に注意）
  - `json + npz`（互換性高、実装やや手間）
- 推奨: `tree.json`（メタ） + `arrays.npz`（N/W/inflight 等）で分離

### 16.2 Checkpoint
- `torch.save(state_dict, path)`
- `checkpoint_id` はファイルの hash または path を採用

---

## 17. ロギング / テスト

### 17.1 ロギング指標
- flush回数、pending滞留時間
- Leaf-only比率
- Alert NG（elem/mol）
- 例外/NaN/clip/timeout 件数

### 17.2 テストケース（最低限）
- Leaf-only（hydrogen_replace 空）
- pending 再到達（waiters に追加され評価1回）
- NaN / 例外 / clip
- Alert elem/mol
- max_depth 終端（報酬0固定でない）
- PUCT tie-break 決定性（action_id 最小）
- pending selection 除外（score=-inf）

---

## 18. 未確定点 / 要確認（質問候補）

実装着手前に決めておくと手戻りが減る論点のうち、**以下は確定**。

- 1) **UCT の具体式**：標準UCT（PUCT寄せ）を採用（10.2 節）。  
  - `N_total(s)=sum_{a∈legal} N(s,a)`（PUCTと同じ）  
  - 未展開は `seed` 固定の一様ランダムで優先選択  
  - `c_uct_default = 1.0`
- 2) **leaf_only（Branch候補が空）**：`depth_action < min_depth` なら報酬評価せず `value=0` 確定。それ以外は ready 判定に従い、ready のみ評価、not_ready は `value=0`。
- 3) **visit_count の定義（推論専用モードの出力）**：Branch重視のため **`visit_count = N(s,a)`**（15章）。
- 4) Fragment CSV の `action_id`：**0..K-1 連番（欠番なし）**。

**残る論点（要決定）**：

- 5) PVネットのバックボーン（ChemProp / AttentiveFP のどちらを初期実装にするか）

---
