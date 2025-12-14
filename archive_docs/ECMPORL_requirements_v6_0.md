# ECMPORL 要件定義書 v6.0

## 1. 文書情報

- 名称: ECMPORL 要件定義書
- バージョン: v6.0
- 作成日: 2025-12-12
- 位置づけ: 本書はリポジトリ構築の起点となる最重要ドキュメントであり、設計仕様書・実装は本書に準拠する。

### 1.1 v6.0での主要変更点（v5.1→v6.0）

- Nodeを **LeafNode（着地）** と **BranchNode（出発）** の2種類に分離
- `pending_items` を `leaf_key` で一意化した **dict** に変更（重複評価防止を構造で担保）
- 報酬関数の **異常系ポリシー（例外/NaN/範囲外/タイムアウト）** を追加
- Leaf→複数Branch の選択規則を **UCT/PUCTで差別化**
  - UCT: 未訪問優先（未訪問集合からランダム）
  - PUCT: 未訪問優先（未訪問集合の中で prior 最大）
- 「ある行動により **Leafしか発生しない**（Branch候補が空）」ケースを仕様として明記（自然終端として処理）
- `hydrogen_replace` 候補数は平均10程度を前提とし、上限制御は不要

---

## 2. システム概要

### 2.1 目的

- fragment-growing による化合物探索を、MCTS（UCT/PUCT）で自動化する
- ユーザ定義の複数報酬関数（各[0,1]）の **相乗平均** を最大化する探索を実現する
- バッチ評価・トランスポジション・virtual loss により、性能と頑健性を両立する

### 2.2 特徴

- **UCT / PUCT 切替**（PUCTは AlphaGo Zero 準拠）
- **Leaf（分子）評価はバッチ評価**（pending→flush）
- **pending回避**（pending状態のLeafに到達する action はSelection候補から除外）
- **virtual loss（inflight）** による探索集中の緩和
- **トランスポジション**（状態を canonical SMILES + depth で一意化し共有）
- 探索木の **保存/ロード**、推論専用モードでの再利用

### 2.3 モード

- 探索モード:
  - `algo=uct` または `algo=puct`
  - `pv_model` が無い場合でも探索可能（PUCTでは一様prior等のfallbackを許容）
- 推論専用モード:
  - 既存の探索木をロードし、追加探索または上位候補抽出のみを行う

---

## 3. 用語定義

- **action**: BranchNode（出発）における離散行動（フラグメント選択等）
- **Leaf**: ダミーアトム無し分子（canonical SMILES）を表す着地状態
- **Branch/frag**: ダミーアトム1個あり状態（canonical SMILES）を表す出発状態
- **Simulation**: ルートから開始し、評価待ち（pending）投入または終端に到達するまでの1回の探索走行
- **pending**: Leafの評価待ち状態（バッチ評価待ち）
- **flush**: pendingキューをまとめて評価し、Backpropする処理
- **virtual loss / inflight**: pending中の枝に探索が集中しないための一時ペナルティ

---

## 4. 前提条件・制約

### 4.1 利用環境

- Python 3.11 以上（設計で固定してよい）
- RDKit 等の化学情報学ライブラリは利用可（実装で選定）

### 4.2 ユーザ提供コンポーネント

- Fragmentテーブル（action候補）
- 報酬関数群（統一IF）
- Alert判定関数
- （任意）PVモデル（Policy/Value、あるいはValueのみ）

### 4.3 探索スケール（目安）

- SARテーブル規模: 30〜1000化合物規模の発想を支える探索を想定
- ただし本パッケージは生成探索であり、探索ステップ数・木サイズはユーザ設定に依存する

---

## 5. パッケージ構成（クラス要件レベル）

- `LeafNode`（着地ノード）
- `BranchNode`（出発ノード）
- `MCTSTree`（トランスポジション表、統計量管理、シリアライズ）
- `Environment`（合法手、SMILES操作、alert、報酬評価のハーネス）
- `Agent`（探索ループ、Simulation/flush/backpropのオーケストレーション）
- `PVModel`（任意、PUCTのprior/Value、Leaf→Branch prior用のValue）

---

## 6. データモデル要件

### 6.1 Node種別

#### 6.1.1 LeafNode（着地）

- `leaf_smiles: str`（ダミー無し canonical SMILES）
- `depth: int`（action適用回数。Branch→Leafで +1）
- `leaf_calc: Literal["not_ready","ready","pending","done"]`
  - not_ready: 評価対象外（下限未到達など）。Branch候補があれば探索継続
  - ready: 評価対象（pendingへ）
  - pending: 評価待ち
  - done: 評価完了
- `is_terminal: bool`
  - Alert失敗 / SMILES失敗 / depth>=max_depth / Branch候補なし（dead-end）など
- `reward: float | None`（doneで0〜1）
- `children_branch_keys: list[str] | None`（hydrogen_replace候補のBranchキー、キャッシュ）

#### 6.1.2 BranchNode（出発）

- `branch_smiles: str`（ダミー1個あり canonical SMILES）
- `depth: int`（Leafと同じ。Leaf→Branchで増えない）
- `legal_actions: list[action]`
- `children_leaf: dict[action_id, leaf_key]`

### 6.2 状態キー（一意性）

状態一意性を **canonical SMILES + depth** で定義する（この前提は固定）。

- `leaf_key = "LEAF:{canonical_leaf_smiles}:{depth}"`
- `branch_key = "BRANCH:{canonical_branch_smiles}:{depth}"`

Leaf/Branchで別種のため、同一SMILESでもキーは別となる。

### 6.3 エッジ統計量（共通）

- `N`（訪問回数）
- `W`（累積報酬）
- `Q = W / N`
- `inflight`（virtual loss用）

実効量:

- `N_eff = N + inflight`
- `Q_eff = (W - inflight * v_loss) / max(1, N_eff)`

---

## 7. 機能要件（MCTS）

### 7.1 Simulation定義（最重要）

1 Simulation は root BranchNode から開始し、同一Simulation内で複数回の展開（Branch→Leaf→Branch→...）を行ってよい。

Simulationが終了する条件:

- LeafNode を `ready` として pending投入し `pending` 化した時点で終了
- Alert失敗 / SMILES正規化失敗 / dead-end 等で終端し、報酬0が確定して即時Backpropした時点で終了
- `depth >= max_depth` により終端し、Leafが `ready→pending` 投入された時点で終了

### 7.2 ルート

- 入力されたダミー付きコアSMILESから `BranchNode(root)` を生成（depth=0）

### 7.3 Selection（BranchNode上のaction選択）

- UCT/PUCT により `legal_actions` から action を選択
- action適用後に到達する LeafNode が `pending` の場合、その action は選択候補から除外（-∞相当）
- 候補が空になった場合:
  - flushを要求し、評価結果が返り候補が復活するまで待機（タイムアウトあり）

#### 7.3.1 UCT

- 未訪問（`N_eff==0`）actionがあれば、その集合から **ランダムに1つ**
- すべて訪問済みなら UCTスコア最大

#### 7.3.2 PUCT（AlphaGo Zero準拠）

- Policy net から `P(s,a)`（softmax）を得る
- `score = Q_eff + c_puct * P(s,a) * sqrt(N_total + 1) / (1 + N_eff(s,a))`
- 未訪問があれば、その中で **P(s,a)最大**（UCTとの差別化）

### 7.4 Expansion（Branch→Leaf）

1. `combine_smiles(branch_smiles, action.frag_smiles) -> leaf_smiles_no_dummy`
2. canonicalize/normalize
   - 失敗: reward=0で即時Backprop（終端）
3. LeafNode をトランスポジション表から取得/作成（leaf_key）
4. `alert_ok_mol(leaf_smiles)==0`:
   - reward=0で即時Backprop（終端）
5. `leaf_calc` 初期化:
   - `depth >= max_depth` なら `ready`（境界では評価）
   - それ以外で評価条件を満たすなら `ready`
   - それ以外は `not_ready`

### 7.5 Leaf処理

#### 7.5.1 `leaf_calc=="ready"`（評価へ）

- Leafを pending投入し `pending` 化
- `pending_items` に登録（leaf_keyで一意化）
- 経路上のエッジに `inflight += 1`
- Simulation終了

#### 7.5.2 `leaf_calc=="not_ready"`（Branch生成）

- `hydrogen_replace(leaf_smiles)` で Branch候補 `B(L)` を得る（空リスト許容）
- `B(L)==[]` の場合:
  - **Leaf-only（dead-end）** とし、reward=0で即時Backprop（終端）
- 空でなければ:
  - Leafに候補をキャッシュし、次に進むBranchを **Leaf→Branch選択規則**で1つ決定して探索継続

### 7.6 Leaf→Branch 選択規則（A案、UCT/PUCTで差別化）

Leafから複数Branch候補が得られた場合、次に進むBranchは **常に1つ**とする（argmaxの枠組み）。

Leaf→Branch エッジ `(L→b)` にも `N/W/Q/inflight` を持たせる。

#### 7.6.1 UCT（Leaf→Branch）

- 未訪問（`N_eff==0`）Branchがあれば、その集合から **ランダムに1つ**
- すべて訪問済みなら UCTスコア最大（`c_uct_branch` を用意可）

#### 7.6.2 PUCT（Leaf→Branch）

- Leaf→Branch prior `P_leaf(L,b)` を定義して用いる
- prior定義:
  - Branch候補 `b` に対し Value net で `V(b)` を得る
  - `P_leaf(L,b) = softmax( V(b) / tau_branch )`
- 未訪問があれば、その中で **P_leaf 最大**
- すべて訪問済みなら
  - `score = Q_eff + c_puct_branch * P_leaf(L,b) * sqrt(N_total + 1) / (1 + N_eff(L→b))`

### 7.7 pending_items（dict化）

- `pending_items: dict[leaf_key, PendingEntry]`
- `PendingEntry` は最低限 `{leaf_node, paths}` を保持（watchers方式）
  - watchersを採用しない場合でも、inflightの解放と複数path反映に相当する管理を必須とする
- 同一 leaf_key に再到達した場合:
  - 評価キューへ再投入しない（dictで一意化）
  - `paths` へ追加（または同等の待機管理）
  - inflightを加算したなら、評価完了時に必ず減算する（リーク禁止）

### 7.8 flush（バッチ評価）

- flush条件（パラメータ化）:
  - `len(pending_items) >= batch_size`
  - `oldest_pending_age >= flush_interval_sec`
  - Selectionで候補が空になった場合の強制flush
- `evaluate_batch([leaf_smiles...]) -> [reward...]`
- 各LeafNodeを `done` に更新し、該当する全pathへBackpropを実行する

### 7.9 Backpropagate

- Backpropは **実際に通過した path（Simulationが保持する経路）**に従う
- Nodeの `parent` は参照用（可視化・デバッグ）であり、Backpropでは使用しない
- 評価完了時の更新順序:
  1) `inflight -= 1`
  2) `N += 1`, `W += reward`（必要なら `Q` 再計算）

### 7.10 トランスポジション

- Leaf/Branchともに state_key（canonical SMILES + depth）で一意化して共有する
- 共有ノードであっても、複数pathから統計更新され得る（MCTSとして許容）

---

## 8. 報酬計算要件

### 8.1 基本

- `reward_i(smiles: str | list[str]) -> float | list[float]`
- 各 `r_i` は [0,1]
- 最終報酬 `R` は相乗平均（数値安定のためlog空間計算を推奨）

### 8.2 異常系ポリシー（必須）

- Exception: 当該 `r_i=0`、ログ記録
- NaN/inf: 当該 `r_i=0`、ログ記録
- 範囲外: `clip(r_i,0,1)`、ログ記録
- タイムアウト: 当該 `r_i=0`、ログ記録
- バッチ中に異常が混在しても、バッチ全体を失敗させない（要素単位で0扱い）
- `alert_ok_mol==0` のLeafは最終報酬0で確定し、報酬関数呼び出しを省略してよい

---

## 9. SMILES 操作要件

- `combine_smiles(core_like: str, frag_like: str) -> str`
  - 入力: ダミー付きSMILES同士
  - 出力: ダミー無しSMILES（Leaf）
- `hydrogen_replace(smiles_no_dummy: str) -> list[str]`
  - 入力: ダミー無しSMILES（Leaf）
  - 出力: ダミー1個ありSMILES候補（Branch）。空リストを許容（Leaf-only）
- `remove_deuterium(smiles: str) -> str`
  - 結合で導入した同位体表記の復元等

---

## 10. I/O とシリアライズ要件

- 探索木の保存/ロードを提供する
- 保存対象（最低限）:
  - Node（Leaf/Branch）の状態（smiles/depth/leaf_calc/reward等）
  - エッジ統計（N/W、inflightは0で保存してよい）
  - トランスポジション表（キー→ノード参照）
- 互換性:
  - 同一バージョン内でのロード再現性を保証
  - 形式はJSON/MessagePack等、実装で選定

---

## 11. 非機能要件（性能・頑健性）

- pendingの重複評価が発生しないこと（leaf_key一意化）
- 例外が探索全体を停止させないこと（異常系ポリシー）
- ロギング:
  - flush回数、pending滞留時間、dead-end比率、例外/clip回数を記録できること
- テスト:
  - Leaf-only、pending再到達、NaN報酬、Alert失敗、max_depth終端を単体テストでカバー

---

## 12. 受け入れ基準（抜粋）

- Nodeが Leaf/Branch に分離され、責務（着地/出発）が明確であること
- pending_items が dict化され、同一 leaf_key の評価が1回に一意化されること
- Leaf-only（hydrogen_replace空）を終端として扱い、クラッシュしないこと
- UCT/PUCTで Leaf→Branch の未訪問優先規則が仕様通りに動作すること
- 報酬関数が例外/NaN/範囲外/タイムアウトを起こしても探索が継続すること
