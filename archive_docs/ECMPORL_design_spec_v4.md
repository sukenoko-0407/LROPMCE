# ECMPORL 設計仕様書 v4.0

## 1. 文書情報

- 名称: ECMPORL 設計仕様書
- バージョン: v4.0
- 作成日: 2025/12/12
- 対応要件定義書: ECMPORL_requirements_v4.md
- 対象読者:
  - 本パッケージの実装担当エンジニア

---

## 2. パッケージ構成

### 2.1 ディレクトリ構造

```
ecmporl_02/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py               # 型定義・Enum・Protocol
│   ├── mcts_node.py           # MCTSNodeクラス
│   └── mcts_tree.py           # MCTSTreeクラス
├── env/
│   ├── __init__.py
│   └── environment.py         # Environmentクラス
├── agent/
│   ├── __init__.py
│   └── agent.py               # Agentクラス
├── io/
│   ├── __init__.py
│   └── serialization.py       # シリアライズユーティリティ
├── parallel/
│   ├── __init__.py
│   └── tree_merge.py          # 木マージユーティリティ
├── config.py                  # 設定クラス
├── inference.py               # 推論専用ユーティリティ
└── logging_utils.py           # ロギングユーティリティ
```

### 2.2 依存関係図

```
                    ┌─────────────┐
                    │   Agent     │
                    └──────┬──────┘
                           │
                           ▼
┌──────────────┐    ┌─────────────┐    ┌─────────────────┐
│ Environment  │◄───│  MCTSTree   │───►│    MCTSNode     │
└──────────────┘    └─────────────┘    └─────────────────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌─────────────┐
│ Reward Funcs │    │ Serializer  │
└──────────────┘    └─────────────┘
```

---

## 3. 型定義 (`core/types.py`)

### 3.1 Enum 定義

```python
from enum import Enum, auto

class MCTSMode(Enum):
    """MCTS探索モード"""
    UCT = auto()
    PUCT = auto()

class LeafCalcStatus(Enum):
    """Leaf評価の状態"""
    NOT_READY = "not_ready"    # min_depth未満、または下限制約未達
    READY = "ready"            # 評価可能だが未評価
    PENDING = "pending"        # 評価待ちキューに追加済み（Selection対象外）
    EVALUATED = "evaluated"    # 評価済み

class TauSchedulerType(Enum):
    """温度スケジューラタイプ"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
```

### 3.2 TypedDict 定義

```python
from typing import TypedDict, Optional

class MolProps(TypedDict):
    """分子プロパティ"""
    HAC: int
    cnt_hetero: int
    cnt_chiral: int
    MW: float

class Constraints(TypedDict):
    """探索制約条件"""
    HAC_min: int
    HAC_max: int
    cnt_hetero_min: int
    cnt_hetero_max: int
    cnt_chiral_min: int
    cnt_chiral_max: int
    MW_min: float
    MW_max: float

class NodeStats(TypedDict):
    """ノード統計量"""
    visit_count: int        # N
    total_reward: float     # W
    q_value: float          # Q = W / N

class TrainingDataPolicy(TypedDict):
    """Policy学習用データ"""
    state_smiles: str
    action_indices: list[int]       # 子ノードのFragment index
    visit_distribution: list[float] # π(a) = N(s,a) / Σ N(s,a)

class TrainingDataValue(TypedDict):
    """Value学習用データ"""
    state_smiles: str
    value_target: float     # z = q_value
```

### 3.3 Protocol 定義（ユーザ提供関数インターフェース）

```python
from typing import Protocol, runtime_checkable
import torch

@runtime_checkable
class CombineSmilesFunc(Protocol):
    """SMILES結合関数のインターフェース"""
    def __call__(self, core_smiles: str, frag_smiles: str) -> str:
        """
        2つのダミーアトム付きSMILESを結合し、
        ダミーアトムを含まないSMILESを返す
        """
        ...

@runtime_checkable
class HydrogenReplaceFunc(Protocol):
    """重水素→ダミーアトム置換関数のインターフェース"""
    def __call__(self, smiles_no_dummy: str) -> list[str]:
        """
        重水素原子をダミーアトムに置換する全パターンを列挙
        Returns: ダミーアトムを1個含むSMILESのリスト
        """
        ...

@runtime_checkable
class RemoveDeuteriumFunc(Protocol):
    """重水素除去関数のインターフェース"""
    def __call__(self, smiles_with_deuterium: str) -> str:
        """
        重水素原子を水素に変換し、水素除去したSMILESを返す
        """
        ...

@runtime_checkable
class AlertElemFunc(Protocol):
    """Alert判定関数（状態用）のインターフェース"""
    def __call__(self, smiles: str) -> int:
        """
        状態（ダミーアトム付きSMILES）のAlert判定
        Returns: 1=OK, 0=Alert該当
        """
        ...

@runtime_checkable
class AlertMolFunc(Protocol):
    """Alert判定関数（Leaf用）のインターフェース"""
    def __call__(self, smiles: str) -> int:
        """
        完成分子のAlert判定
        Returns: 1=OK, 0=Alert該当
        """
        ...

@runtime_checkable
class MeasureMolPropsFunc(Protocol):
    """分子プロパティ計測関数のインターフェース"""
    def __call__(self, smiles: str) -> MolProps:
        """分子プロパティを計算"""
        ...

@runtime_checkable
class CountSubspaceFunc(Protocol):
    """サブスペースサイズ算出関数のインターフェース"""
    def __call__(self, state_smiles: str) -> int:
        """そのノードの先に存在し得る全化合物数を返す"""
        ...

@runtime_checkable
class RewardFunc(Protocol):
    """報酬関数のインターフェース（バッチ対応）"""
    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        """
        報酬を計算（0-1の範囲）
        単一SMILESまたはリストを受け取り、対応する形式で返す
        """
        ...

@runtime_checkable
class PolicyValueModel(Protocol):
    """統合モデル（Policy head + Value head）のインターフェース"""
    def __call__(self, state_smiles: str) -> tuple[torch.Tensor, float]:
        """
        Args:
            state_smiles: 状態を表すSMILES
        Returns:
            policy_logits: 全Fragmentに対するlogit (Tensor[num_fragments])
            value: 状態価値 (float)
        """
        ...
```

---

## 4. MCTSNode クラス (`core/mcts_node.py`)

### 4.1 クラス定義

```python
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from .types import LeafCalcStatus, MolProps

if TYPE_CHECKING:
    from .mcts_tree import MCTSTree

@dataclass
class MCTSNode:
    """
    MCTSの探索木におけるノード。
    状態（部分構造SMILES）と統計情報を保持する。
    """

    # === 状態情報 ===
    state_smiles: str                              # canonical SMILES（ダミーアトム含む）
    depth: int                                     # ルートからの深さ
    leaf_smiles: Optional[str] = None              # 完成分子SMILES（重水素なし）
    leaf_calc: LeafCalcStatus = LeafCalcStatus.NOT_READY

    # === 統計量 ===
    visit_count: int = 0                           # N
    total_reward: float = 0.0                      # W

    # === 構造情報 ===
    parent: Optional['MCTSNode'] = None            # 親ノード（グラフ参照用）
    incoming_action: Optional[int] = None          # このノードへの遷移に使用したFragment index
    children: dict[int, 'MCTSNode'] = field(default_factory=dict)  # {fragment_index: child_node}
    is_terminal: bool = False
    num_sub: int = 0                               # 化学サブスペースサイズ

    # === その他 ===
    metadata: Optional[dict] = None
    _mol_props: Optional[MolProps] = field(default=None, repr=False)

    # === プロパティ ===
    @property
    def q_value(self) -> float:
        """平均報酬 Q = W / N"""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def node_id(self) -> tuple[str, int]:
        """ノードの一意識別子"""
        return (self.state_smiles, self.depth)

    def has_untried_actions(self, legal_actions: list[int]) -> bool:
        """未試行の行動があるかチェック"""
        return any(a not in self.children for a in legal_actions)

    def get_untried_actions(self, legal_actions: list[int]) -> list[int]:
        """未試行の行動リストを取得"""
        return [a for a in legal_actions if a not in self.children]

    def add_child(self, fragment_index: int, child_node: 'MCTSNode') -> None:
        """子ノードを追加（最初の到達時のみ親を設定）"""
        if fragment_index in self.children:
            # 既存の子ノードがある場合は何もしない（トランスポジション）
            return
        child_node.parent = self
        child_node.incoming_action = fragment_index
        self.children[fragment_index] = child_node

    def get_child(self, fragment_index: int) -> Optional['MCTSNode']:
        """子ノードを取得"""
        return self.children.get(fragment_index)

    def get_visit_distribution(self) -> dict[int, float]:
        """子ノードの訪問回数分布 π(a) = N(s,a) / Σ N(s,a)"""
        total_visits = sum(c.visit_count for c in self.children.values())
        if total_visits == 0:
            return {idx: 0.0 for idx in self.children.keys()}
        return {
            idx: child.visit_count / total_visits
            for idx, child in self.children.items()
        }

    def update_stats(self, reward: float) -> None:
        """統計量を更新（Backpropagation用）"""
        self.visit_count += 1
        self.total_reward += reward

    def set_mol_props(self, props: MolProps) -> None:
        """分子プロパティをキャッシュ"""
        self._mol_props = props

    def get_mol_props(self) -> Optional[MolProps]:
        """キャッシュされた分子プロパティを取得"""
        return self._mol_props

    def to_dict(self) -> dict:
        """シリアライズ用の辞書に変換"""
        return {
            'state_smiles': self.state_smiles,
            'depth': self.depth,
            'leaf_smiles': self.leaf_smiles,
            'leaf_calc': self.leaf_calc.value,
            'visit_count': self.visit_count,
            'total_reward': self.total_reward,
            'incoming_action': self.incoming_action,
            'is_terminal': self.is_terminal,
            'num_sub': self.num_sub,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MCTSNode':
        """辞書からインスタンスを復元"""
        node = cls(
            state_smiles=data['state_smiles'],
            depth=data['depth'],
            leaf_smiles=data.get('leaf_smiles'),
            leaf_calc=LeafCalcStatus(data['leaf_calc']),
            visit_count=data['visit_count'],
            total_reward=data['total_reward'],
            incoming_action=data.get('incoming_action'),
            is_terminal=data['is_terminal'],
            num_sub=data['num_sub'],
            metadata=data.get('metadata'),
        )
        return node

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSNode):
            return False
        return self.node_id == other.node_id
```

---

## 5. MCTSTree クラス (`core/mcts_tree.py`)

### 5.1 クラス定義

```python
from typing import Optional, Callable
from collections import defaultdict
import random
import math

from .mcts_node import MCTSNode
from .types import LeafCalcStatus, MCTSMode

class MCTSTree:
    """
    MCTS探索木を管理するクラス。
    木構造の操作と探索ロジックを担当する。
    """

    def __init__(
        self,
        root_smiles: str,
        environment: 'Environment',
        agent: 'Agent',
        config: 'SearchConfig'
    ):
        """
        Args:
            root_smiles: コア構造（ダミーアトム付きSMILES）
            environment: 環境クラス
            agent: エージェントクラス
            config: 探索設定
        """
        self.environment = environment
        self.agent = agent
        self.config = config

        # ルートノード作成
        self.root = MCTSNode(
            state_smiles=root_smiles,
            depth=0,
            leaf_calc=LeafCalcStatus.NOT_READY
        )
        self.root.num_sub = environment.count_subspace(root_smiles)

        # ノードインデックス: (state_smiles, depth) -> MCTSNode
        self._node_index: dict[tuple[str, int], MCTSNode] = {
            self.root.node_id: self.root
        }

        # 評価待ちキュー: (node, path) のリスト
        self._pending_items: list[tuple[MCTSNode, list[MCTSNode]]] = []

    def get_node(self, state_smiles: str, depth: int) -> Optional[MCTSNode]:
        """ノードインデックスからノードを取得"""
        return self._node_index.get((state_smiles, depth))

    def register_node(self, node: MCTSNode) -> None:
        """ノードをインデックスに登録"""
        self._node_index[node.node_id] = node

    @property
    def node_count(self) -> int:
        """ノード数を取得"""
        return len(self._node_index)

    def search(
        self,
        n_simulations: int,
        batch_eval_interval: int,
        train_interval: Optional[int] = None
    ) -> None:
        """
        MCTS探索のメインループ。

        Args:
            n_simulations: Simulation回数
            batch_eval_interval: バッチ評価間隔
            train_interval: モデル学習間隔（Noneの場合は学習しない）
        """
        for i in range(n_simulations):
            path = self._run_simulation()

            # 行き詰まりの場合（pathの最後がreadyでない）
            last_node = path[-1]
            if last_node.leaf_calc == LeafCalcStatus.READY:
                self._pending_items.append((last_node, path.copy()))
                last_node.leaf_calc = LeafCalcStatus.PENDING
            elif last_node.is_terminal and last_node.depth < self.config.min_depth:
                # 行き詰まり: 即時報酬0でBackpropagate
                self._backpropagate(path, 0.0)

            # バッチ評価トリガー
            if len(self._pending_items) >= batch_eval_interval:
                self._execute_batch_evaluation()

            # モデル学習トリガー
            if train_interval and (i + 1) % train_interval == 0:
                self.agent.train_step(self)

        # 残りのノードを処理
        if self._pending_items:
            self._execute_batch_evaluation()

    def _run_simulation(self) -> list[MCTSNode]:
        """1回のSimulationを実行し、辿った経路を返す"""
        path = [self.root]
        node = self.root

        while True:
            # Terminalまたはmax_depthに達した場合は停止
            if node.is_terminal or node.depth >= self.config.max_depth:
                break

            # READY状態なら評価対象として停止
            if node.leaf_calc == LeafCalcStatus.READY:
                break

            # Legal Actionsを取得
            legal_actions = self.environment.get_legal_action_indices(node.state_smiles)

            if not legal_actions:
                # 有効な行動がない場合はterminalとして扱う
                node.is_terminal = True
                break

            # 行動選択
            action = self._select_action(node, legal_actions)

            # 展開
            new_node = self._expand(node, action)
            path.append(new_node)
            node = new_node

        return path

    def _select_action(self, node: MCTSNode, legal_actions: list[int]) -> int:
        """UCT/PUCTに基づく行動選択"""
        if self.config.mode == MCTSMode.UCT:
            return self._select_action_uct(node, legal_actions)
        else:
            return self._select_action_puct(node, legal_actions)

    def _select_action_uct(self, node: MCTSNode, legal_actions: list[int]) -> int:
        """UCTによる行動選択"""
        # 1. 未試行の行動があれば優先（ランダム）
        untried = node.get_untried_actions(legal_actions)
        if untried:
            return random.choice(untried)

        # 2. 全て試行済みならUCTスコアで選択
        best_score = float('-inf')
        best_action = legal_actions[0]
        N_parent = node.visit_count

        for action in legal_actions:
            child = node.children.get(action)
            if child is None or child.leaf_calc == LeafCalcStatus.PENDING:
                continue

            Q = child.q_value
            N_child = child.visit_count
            score = Q + self.config.c_uct * math.sqrt(
                math.log(N_parent + 1) / (1 + N_child)
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _select_action_puct(self, node: MCTSNode, legal_actions: list[int]) -> int:
        """PUCTによる行動選択（AlphaGoZero方式）"""
        # Policy Networkから確率を取得
        probs = self.agent.compute_action_probs(
            node.state_smiles, legal_actions
        )

        best_score = float('-inf')
        best_action = legal_actions[0]
        N_parent = node.visit_count

        for i, action in enumerate(legal_actions):
            child = node.children.get(action)
            if child is not None and child.leaf_calc == LeafCalcStatus.PENDING:
                continue

            P = probs[i].item()
            Q = child.q_value if child else 0.0
            N_child = child.visit_count if child else 0

            score = Q + self.config.c_puct * P * (
                math.sqrt(N_parent + 1) / (1 + N_child)
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _expand(self, parent: MCTSNode, action: int) -> MCTSNode:
        """ノードを展開"""
        # 既存の子ノードがあれば返す
        existing = parent.get_child(action)
        if existing is not None:
            return existing

        # Fragment SMILESを取得
        fragment_smiles = self.environment.get_fragment_smiles(action)

        # combine_smiles + hydrogen_replace
        combined = self.environment.combine_smiles(
            parent.state_smiles, fragment_smiles
        )
        next_states = self.environment.hydrogen_replace(combined)

        # 次状態を選択
        if self.config.mode == MCTSMode.UCT or len(next_states) == 1:
            next_state = random.choice(next_states)
        else:
            # PUCTモード: Value Networkで選択
            next_state = self._select_next_state_by_value(next_states)

        # 既存ノードの確認（トランスポジション）
        new_depth = parent.depth + 1
        existing_node = self.get_node(next_state, new_depth)

        if existing_node:
            # 既存ノードを再利用（親は更新しない）
            parent.children[action] = existing_node
            return existing_node

        # 新ノード作成
        new_node = self._create_node(next_state, new_depth, parent, action)
        parent.add_child(action, new_node)
        self.register_node(new_node)

        return new_node

    def _create_node(
        self,
        state_smiles: str,
        depth: int,
        parent: MCTSNode,
        action: int
    ) -> MCTSNode:
        """新ノードを作成"""
        node = MCTSNode(
            state_smiles=state_smiles,
            depth=depth
        )

        # Alert判定
        if self.environment.alert_ok_elem(state_smiles) == 0:
            node.is_terminal = True

        # max_depth判定
        if depth >= self.config.max_depth:
            node.is_terminal = True

        # num_sub計算
        node.num_sub = self.environment.count_subspace(state_smiles)

        # leaf_calc初期化
        if depth >= self.config.min_depth:
            mol_props = self.environment.calc_mol_props(state_smiles)
            if self.environment.satisfies_min_constraints(mol_props):
                node.leaf_calc = LeafCalcStatus.READY
                # leaf_smilesを生成
                node.leaf_smiles = self.environment.remove_deuterium(state_smiles)

        return node

    def _select_next_state_by_value(self, candidates: list[str]) -> str:
        """Value Networkを用いて次状態を選択（温度付きsoftmax）"""
        import torch
        import torch.nn.functional as F

        values = []
        for state in candidates:
            v = self.agent.compute_state_value(state)
            values.append(v)

        tau = self.agent.get_current_tau()
        logits = torch.tensor(values) / tau
        probs = F.softmax(logits, dim=0)

        idx = torch.multinomial(probs, 1).item()
        return candidates[idx]

    def _execute_batch_evaluation(self) -> None:
        """バッチ評価を実行"""
        if not self._pending_items:
            return

        nodes = [item[0] for item in self._pending_items]
        paths = [item[1] for item in self._pending_items]

        # Leaf SMILESリストを作成
        smiles_list = [n.leaf_smiles for n in nodes]

        # バッチ評価
        rewards = self.environment.evaluate_batch(smiles_list)

        # バッチBackpropagation
        self._backpropagate_batch(list(zip(paths, rewards)))

        # leaf_calcを更新
        for node, _ in self._pending_items:
            node.leaf_calc = LeafCalcStatus.EVALUATED

        self._pending_items.clear()

    def _backpropagate(self, path: list[MCTSNode], reward: float) -> None:
        """経路を遡って統計量を更新"""
        for node in reversed(path):
            node.update_stats(reward)

    def _backpropagate_batch(
        self,
        path_reward_pairs: list[tuple[list[MCTSNode], float]]
    ) -> None:
        """バッチBackpropagation"""
        updates: dict[tuple, dict] = defaultdict(
            lambda: {"reward_sum": 0.0, "visit_delta": 0}
        )

        for path, reward in path_reward_pairs:
            for node in path:
                key = node.node_id
                updates[key]["reward_sum"] += reward
                updates[key]["visit_delta"] += 1

        for key, delta in updates.items():
            node = self._node_index.get(key)
            if node:
                node.visit_count += delta["visit_delta"]
                node.total_reward += delta["reward_sum"]

    # === 学習データ抽出 ===

    def extract_value_training_data(
        self,
        q_min: Optional[float] = None
    ) -> list['TrainingDataValue']:
        """Value Head学習用データを抽出"""
        from .types import TrainingDataValue

        data = []
        for node in self._node_index.values():
            if node.visit_count == 0:
                continue
            if q_min is not None and node.q_value < q_min:
                continue
            data.append(TrainingDataValue(
                state_smiles=node.state_smiles,
                value_target=node.q_value
            ))
        return data

    def extract_policy_training_data(self) -> list['TrainingDataPolicy']:
        """Policy Head学習用データを抽出"""
        from .types import TrainingDataPolicy

        data = []
        for node in self._node_index.values():
            # 評価済みの子ノードのみ対象
            evaluated_children = {
                idx: child for idx, child in node.children.items()
                if child.leaf_calc == LeafCalcStatus.EVALUATED
            }

            if not evaluated_children:
                continue

            total_visits = sum(c.visit_count for c in evaluated_children.values())
            if total_visits == 0:
                continue

            action_indices = list(evaluated_children.keys())
            visit_dist = [
                evaluated_children[idx].visit_count / total_visits
                for idx in action_indices
            ]

            data.append(TrainingDataPolicy(
                state_smiles=node.state_smiles,
                action_indices=action_indices,
                visit_distribution=visit_dist
            ))

        return data

    # === ノードフィルタリング ===

    def filter_nodes(
        self,
        q_min: Optional[float] = None,
        total_reward_min: Optional[float] = None,
        num_sub_min: Optional[int] = None,
        depth_range: Optional[tuple[int, int]] = None
    ) -> list[MCTSNode]:
        """条件を満たすノードを抽出"""
        results = []
        for node in self._node_index.values():
            if q_min is not None and node.q_value < q_min:
                continue
            if total_reward_min is not None and node.total_reward < total_reward_min:
                continue
            if num_sub_min is not None and node.num_sub < num_sub_min:
                continue
            if depth_range is not None:
                if node.depth < depth_range[0] or node.depth > depth_range[1]:
                    continue
            results.append(node)
        return results

    # === シリアライズ ===

    def to_dict(self) -> dict:
        """シリアライズ用辞書に変換"""
        nodes_data = []
        edges_data = []

        for node in self._node_index.values():
            nodes_data.append(node.to_dict())
            for action, child in node.children.items():
                edges_data.append({
                    'parent_id': node.node_id,
                    'child_id': child.node_id,
                    'action': action
                })

        return {
            'root_id': self.root.node_id,
            'nodes': nodes_data,
            'edges': edges_data
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        environment: 'Environment',
        agent: 'Agent',
        config: 'SearchConfig'
    ) -> 'MCTSTree':
        """辞書からインスタンスを復元"""
        # ノードを復元
        nodes = {
            tuple(n['state_smiles'], n['depth']): MCTSNode.from_dict(n)
            for n in data['nodes']
        }

        # エッジを復元
        for edge in data['edges']:
            parent = nodes[tuple(edge['parent_id'])]
            child = nodes[tuple(edge['child_id'])]
            parent.children[edge['action']] = child
            if child.parent is None:
                child.parent = parent
                child.incoming_action = edge['action']

        # Treeを構築
        root = nodes[tuple(data['root_id'])]
        tree = cls.__new__(cls)
        tree.root = root
        tree._node_index = nodes
        tree.environment = environment
        tree.agent = agent
        tree.config = config
        tree._pending_items = []

        return tree
```

---

## 6. Environment クラス (`env/environment.py`)

### 6.1 クラス定義

```python
import pandas as pd
import numpy as np
from typing import Optional, Callable
from ..core.types import (
    MolProps, Constraints,
    CombineSmilesFunc, HydrogenReplaceFunc, RemoveDeuteriumFunc,
    AlertElemFunc, AlertMolFunc, MeasureMolPropsFunc,
    CountSubspaceFunc, RewardFunc
)

class Environment:
    """
    化学的な制約・評価関数・Fragment情報を管理するクラス。
    """

    def __init__(
        self,
        fragment_df: pd.DataFrame,
        combine_fn: CombineSmilesFunc,
        hydrogen_replace_fn: HydrogenReplaceFunc,
        remove_deuterium_fn: RemoveDeuteriumFunc,
        reward_fns: list[RewardFunc],
        alert_elem_fn: AlertElemFunc,
        alert_mol_fn: AlertMolFunc,
        calc_props_fn: MeasureMolPropsFunc,
        count_subspace_fn: CountSubspaceFunc,
        constraints: Constraints
    ):
        """
        Args:
            fragment_df: Fragmentテーブル（smiles, HAC, cnt_hetero, cnt_chiral, MW列必須）
            combine_fn: SMILES結合関数
            hydrogen_replace_fn: 重水素置換関数
            remove_deuterium_fn: 重水素除去関数
            reward_fns: 報酬関数リスト
            alert_elem_fn: 状態用Alert判定関数
            alert_mol_fn: Leaf用Alert判定関数
            calc_props_fn: 分子プロパティ計測関数
            count_subspace_fn: サブスペースサイズ算出関数
            constraints: 探索制約条件
        """
        self._fragment_df = fragment_df.copy()
        self._combine_fn = combine_fn
        self._hydrogen_replace_fn = hydrogen_replace_fn
        self._remove_deuterium_fn = remove_deuterium_fn
        self._reward_fns = reward_fns
        self._alert_elem_fn = alert_elem_fn
        self._alert_mol_fn = alert_mol_fn
        self._calc_props_fn = calc_props_fn
        self._count_subspace_fn = count_subspace_fn
        self._constraints = constraints

        # Fragmentプロパティをキャッシュ
        self._fragment_props = self._fragment_df[
            ['HAC', 'cnt_hetero', 'cnt_chiral', 'MW']
        ].to_dict('records')

    @property
    def num_fragments(self) -> int:
        """Fragment数を取得"""
        return len(self._fragment_df)

    def get_fragment_smiles(self, index: int) -> str:
        """インデックスからFragment SMILESを取得"""
        return self._fragment_df.iloc[index]['smiles']

    def get_legal_action_indices(self, state_smiles: str) -> list[int]:
        """
        状態から有効なFragment indexリストを取得。
        制約条件でフィルタリングする。
        """
        state_props = self._calc_props_fn(state_smiles)
        legal_indices = []

        for idx, frag_props in enumerate(self._fragment_props):
            # 結合後のプロパティを予測
            combined_HAC = state_props['HAC'] + frag_props['HAC']
            combined_hetero = state_props['cnt_hetero'] + frag_props['cnt_hetero']
            combined_chiral = state_props['cnt_chiral'] + frag_props['cnt_chiral']
            combined_MW = state_props['MW'] + frag_props['MW']

            # 上限チェック
            if combined_HAC > self._constraints['HAC_max']:
                continue
            if combined_hetero > self._constraints['cnt_hetero_max']:
                continue
            if combined_chiral > self._constraints['cnt_chiral_max']:
                continue
            if combined_MW > self._constraints['MW_max']:
                continue

            legal_indices.append(idx)

        return legal_indices

    def combine_smiles(self, core_smiles: str, frag_smiles: str) -> str:
        """2つのSMILESを結合"""
        return self._combine_fn(core_smiles, frag_smiles)

    def hydrogen_replace(self, smiles: str) -> list[str]:
        """重水素をダミーアトムに置換した全パターンを生成"""
        return self._hydrogen_replace_fn(smiles)

    def remove_deuterium(self, smiles: str) -> str:
        """重水素を除去してLeaf SMILESを生成"""
        return self._remove_deuterium_fn(smiles)

    def alert_ok_elem(self, smiles: str) -> int:
        """状態のAlert判定 (1=OK, 0=Alert)"""
        return self._alert_elem_fn(smiles)

    def alert_ok_mol(self, smiles: str) -> int:
        """LeafのAlert判定 (1=OK, 0=Alert)"""
        return self._alert_mol_fn(smiles)

    def calc_mol_props(self, smiles: str) -> MolProps:
        """分子プロパティを計算"""
        return self._calc_props_fn(smiles)

    def count_subspace(self, state_smiles: str) -> int:
        """サブスペースサイズを算出"""
        return self._count_subspace_fn(state_smiles)

    def satisfies_min_constraints(self, props: MolProps) -> bool:
        """下限制約を満たすかチェック"""
        return (
            props['HAC'] >= self._constraints['HAC_min'] and
            props['cnt_hetero'] >= self._constraints['cnt_hetero_min'] and
            props['cnt_chiral'] >= self._constraints['cnt_chiral_min'] and
            props['MW'] >= self._constraints['MW_min']
        )

    def evaluate_single(self, smiles: str) -> float:
        """単一のLeaf SMILESを評価"""
        # Alert判定
        if self._alert_mol_fn(smiles) == 0:
            return 0.0

        # 各報酬関数を評価して相乗平均を計算
        rewards = []
        for reward_fn in self._reward_fns:
            r = reward_fn(smiles)
            rewards.append(r)

        if not rewards:
            return 0.0

        # 相乗平均
        product = np.prod(rewards)
        return float(product ** (1.0 / len(rewards)))

    def evaluate_batch(self, smiles_list: list[str]) -> list[float]:
        """複数のLeaf SMILESをバッチ評価"""
        n = len(smiles_list)

        # Alert判定（バッチ）
        alert_results = [self._alert_mol_fn(s) for s in smiles_list]

        # 各報酬関数をバッチ実行
        all_rewards = []
        for reward_fn in self._reward_fns:
            try:
                # バッチ対応の場合
                batch_rewards = reward_fn(smiles_list)
                all_rewards.append(batch_rewards)
            except TypeError:
                # 単一処理にフォールバック
                single_rewards = [reward_fn(s) for s in smiles_list]
                all_rewards.append(single_rewards)

        # 相乗平均を計算
        results = []
        for i in range(n):
            if alert_results[i] == 0:
                results.append(0.0)
                continue

            rewards = [r[i] for r in all_rewards]
            product = np.prod(rewards)
            geom_mean = float(product ** (1.0 / len(rewards)))
            results.append(geom_mean)

        return results
```

---

## 7. Agent クラス (`agent/agent.py`)

### 7.1 クラス定義

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.types import MCTSMode, TauSchedulerType, PolicyValueModel
from ..core.mcts_tree import MCTSTree

class Agent(nn.Module):
    """
    MLモデルの保持・学習を担当するクラス。
    PolicyValueNetwork（一体型）を管理する。
    """

    def __init__(
        self,
        policy_value_network: PolicyValueModel,
        config: 'AgentConfig'
    ):
        """
        Args:
            policy_value_network: 一体型モデル（Policy + Value Head）
            config: エージェント設定
        """
        super().__init__()
        self.policy_value_network = policy_value_network
        self.config = config

        # 温度スケジューラ初期化
        self._current_step = 0
        self._tau = config.tau_initial

    def compute_action_probs(
        self,
        state_smiles: str,
        legal_action_indices: list[int]
    ) -> torch.Tensor:
        """
        Legal Actionsに対する選択確率を計算。

        Args:
            state_smiles: 状態SMILES
            legal_action_indices: 有効なFragmentのインデックスリスト

        Returns:
            確率テンソル (サイズ: len(legal_action_indices))
        """
        with torch.no_grad():
            policy_logits, _ = self.policy_value_network(state_smiles)

            # Legal Actionsのlogitを抽出
            legal_logits = policy_logits[legal_action_indices]

            # Softmaxで確率化
            probs = F.softmax(legal_logits, dim=0)

        return probs

    def compute_state_value(self, state_smiles: str) -> float:
        """状態価値を計算"""
        with torch.no_grad():
            _, value = self.policy_value_network(state_smiles)
        return float(value)

    def get_current_tau(self) -> float:
        """現在の温度パラメータを取得"""
        return self._tau

    def update_tau(self, step: int) -> None:
        """温度パラメータを更新"""
        self._current_step = step

        if self.config.tau_scheduler_type == TauSchedulerType.LINEAR:
            # 線形減衰
            progress = min(step / self.config.max_steps, 1.0)
            self._tau = self.config.tau_initial + \
                (self.config.tau_final - self.config.tau_initial) * progress

        elif self.config.tau_scheduler_type == TauSchedulerType.EXPONENTIAL:
            # 指数減衰
            import math
            k = math.log(self.config.tau_initial / self.config.tau_final) / self.config.max_steps
            self._tau = self.config.tau_initial * math.exp(-k * step)

    def train_step(self, tree: MCTSTree) -> dict:
        """
        Treeから学習データを抽出してモデルを学習。

        Args:
            tree: MCTSTree

        Returns:
            Loss情報を含む辞書
        """
        # 学習データ抽出
        value_data = tree.extract_value_training_data(
            q_min=self.config.q_threshold_for_training
        )
        policy_data = tree.extract_policy_training_data()

        if not value_data or not policy_data:
            return {'loss': 0.0, 'loss_value': 0.0, 'loss_policy': 0.0}

        # ミニバッチ学習
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        n_batches = 0

        # Value学習
        for i in range(0, len(value_data), self.config.batch_size):
            batch = value_data[i:i + self.config.batch_size]
            value_loss = self._compute_value_loss(batch)
            total_value_loss += value_loss.item()
            n_batches += 1

        # Policy学習
        for i in range(0, len(policy_data), self.config.batch_size):
            batch = policy_data[i:i + self.config.batch_size]
            policy_loss = self._compute_policy_loss(batch)
            total_policy_loss += policy_loss.item()

        # 結合Loss（実際の学習はoptimizer経由で外部から行う想定）
        total_loss = total_value_loss + total_policy_loss

        return {
            'loss': total_loss / max(n_batches, 1),
            'loss_value': total_value_loss / max(n_batches, 1),
            'loss_policy': total_policy_loss / max(n_batches, 1)
        }

    def _compute_value_loss(self, batch: list) -> torch.Tensor:
        """Value Lossを計算: L_value = (v - z)²"""
        losses = []
        for item in batch:
            _, v = self.policy_value_network(item['state_smiles'])
            z = item['value_target']
            loss = (v - z) ** 2
            losses.append(loss)
        return torch.mean(torch.stack(losses))

    def _compute_policy_loss(self, batch: list) -> torch.Tensor:
        """Policy Lossを計算: L_policy = -Σ_a π(a) log p(a)"""
        losses = []
        for item in batch:
            policy_logits, _ = self.policy_value_network(item['state_smiles'])
            action_indices = item['action_indices']
            target_dist = torch.tensor(item['visit_distribution'])

            # Legal actionsのlogitを抽出してsoftmax
            legal_logits = policy_logits[action_indices]
            log_probs = F.log_softmax(legal_logits, dim=0)

            # クロスエントロピー: -Σ π(a) log p(a)
            loss = -torch.sum(target_dist * log_probs)
            losses.append(loss)

        return torch.mean(torch.stack(losses))

    def save(self, path: str) -> None:
        """モデルを保存"""
        torch.save({
            'policy_value_network': self.policy_value_network.state_dict(),
            'config': self.config,
            'current_step': self._current_step,
            'tau': self._tau
        }, path)

    def load(self, path: str) -> None:
        """モデルを読み込み"""
        checkpoint = torch.load(path)
        self.policy_value_network.load_state_dict(
            checkpoint['policy_value_network']
        )
        self._current_step = checkpoint.get('current_step', 0)
        self._tau = checkpoint.get('tau', self.config.tau_initial)
```

---

## 8. 設定クラス (`config.py`)

```python
from dataclasses import dataclass
from typing import Optional
from .core.types import MCTSMode, TauSchedulerType, Constraints

@dataclass
class SearchConfig:
    """MCTS探索設定"""
    mode: MCTSMode = MCTSMode.UCT
    max_depth: int = 10
    min_depth: int = 3
    c_uct: float = 1.41
    c_puct: float = 1.0
    batch_eval_interval: int = 128
    constraints: Optional[Constraints] = None

@dataclass
class AgentConfig:
    """Agent設定"""
    tau_initial: float = 1.0
    tau_final: float = 0.1
    tau_scheduler_type: TauSchedulerType = TauSchedulerType.LINEAR
    max_steps: int = 10000
    train_interval: int = 256
    batch_size: int = 64
    q_threshold_for_training: Optional[float] = None
```

---

## 9. 並列処理・木マージ (`parallel/tree_merge.py`)

```python
from typing import Optional
from collections import defaultdict
from ..core.mcts_tree import MCTSTree
from ..core.mcts_node import MCTSNode
from ..core.types import LeafCalcStatus

def merge_trees(trees: list[MCTSTree]) -> MCTSTree:
    """
    複数のMCTS木をマージして1つの木を作成。

    Args:
        trees: マージ対象の木リスト

    Returns:
        マージ後の木
    """
    if not trees:
        raise ValueError("At least one tree is required")

    if len(trees) == 1:
        return trees[0]

    # 最初の木をベースとする
    base_tree = trees[0]

    for other_tree in trees[1:]:
        merge_into(base_tree, other_tree)

    return base_tree

def merge_into(base_tree: MCTSTree, other_tree: MCTSTree) -> None:
    """
    other_treeの内容をbase_treeにマージする（破壊的操作）。

    Args:
        base_tree: マージ先の木
        other_tree: マージ元の木
    """
    for node_id, other_node in other_tree._node_index.items():
        base_node = base_tree._node_index.get(node_id)

        if base_node is None:
            # 新しいノードを追加
            new_node = MCTSNode(
                state_smiles=other_node.state_smiles,
                depth=other_node.depth,
                leaf_smiles=other_node.leaf_smiles,
                leaf_calc=other_node.leaf_calc,
                visit_count=other_node.visit_count,
                total_reward=other_node.total_reward,
                is_terminal=other_node.is_terminal,
                num_sub=other_node.num_sub,
                metadata=other_node.metadata
            )
            base_tree._node_index[node_id] = new_node
        else:
            # 既存ノードの統計量をマージ
            base_node.visit_count += other_node.visit_count
            base_node.total_reward += other_node.total_reward

            # leaf_calcは評価済みを優先
            if other_node.leaf_calc == LeafCalcStatus.EVALUATED:
                base_node.leaf_calc = LeafCalcStatus.EVALUATED

    # 子ノード関係を再構築
    for node_id, other_node in other_tree._node_index.items():
        base_node = base_tree._node_index[node_id]

        for action, other_child in other_node.children.items():
            if action not in base_node.children:
                child_in_base = base_tree._node_index.get(other_child.node_id)
                if child_in_base:
                    base_node.children[action] = child_in_base
                    if child_in_base.parent is None:
                        child_in_base.parent = base_node
                        child_in_base.incoming_action = action
```

---

## 10. シリアライズ (`io/serialization.py`)

```python
import pickle
import gzip
from pathlib import Path
from typing import Union
from ..core.mcts_tree import MCTSTree

def save_tree(tree: MCTSTree, path: Union[str, Path]) -> None:
    """
    MCTSTreeを圧縮バイナリ形式で保存。

    Args:
        tree: 保存対象のTree
        path: 保存先パス
    """
    data = tree.to_dict()
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def load_tree(
    path: Union[str, Path],
    environment: 'Environment',
    agent: 'Agent',
    config: 'SearchConfig'
) -> MCTSTree:
    """
    保存されたMCTSTreeを読み込み。

    Args:
        path: ファイルパス
        environment: Environment
        agent: Agent
        config: SearchConfig

    Returns:
        復元されたMCTSTree
    """
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return MCTSTree.from_dict(data, environment, agent, config)
```

---

## 11. 推論専用モード (`inference.py`)

```python
from typing import Optional
from .core.mcts_tree import MCTSTree
from .core.mcts_node import MCTSNode

class InferenceEngine:
    """推論専用モードのユーティリティクラス"""

    def __init__(self, tree: MCTSTree):
        """
        Args:
            tree: 探索済みのMCTSTree
        """
        self.tree = tree

    def get_high_reward_nodes(
        self,
        q_threshold: float,
        num_sub_min: Optional[int] = None,
        total_reward_min: Optional[float] = None,
        depth_range: Optional[tuple[int, int]] = None
    ) -> list[MCTSNode]:
        """
        高報酬ノードを抽出。

        Args:
            q_threshold: Q値の閾値（必須）
            num_sub_min: サブスペースサイズの最小値
            total_reward_min: 累計報酬の最小値
            depth_range: 深さの範囲 (min, max)

        Returns:
            条件を満たすノードのリスト
        """
        return self.tree.filter_nodes(
            q_min=q_threshold,
            num_sub_min=num_sub_min,
            total_reward_min=total_reward_min,
            depth_range=depth_range
        )

    def get_node_info(self, node: MCTSNode) -> dict:
        """ノードの詳細情報を取得"""
        return {
            'state_smiles': node.state_smiles,
            'leaf_smiles': node.leaf_smiles,
            'depth': node.depth,
            'q_value': node.q_value,
            'visit_count': node.visit_count,
            'total_reward': node.total_reward,
            'num_sub': node.num_sub,
            'is_terminal': node.is_terminal
        }

    def export_results(
        self,
        nodes: list[MCTSNode],
        format: str = 'csv'
    ) -> str:
        """
        結果をエクスポート。

        Args:
            nodes: 出力対象ノード
            format: 出力形式 ('csv' or 'json')

        Returns:
            フォーマットされた文字列
        """
        import json
        import csv
        from io import StringIO

        data = [self.get_node_info(n) for n in nodes]

        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            if not data:
                return ""
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        else:
            raise ValueError(f"Unknown format: {format}")
```

---

## 12. ロギング (`logging_utils.py`)

```python
import logging
from typing import Optional

def setup_logger(
    name: str = 'ecmporl',
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """ロガーをセットアップ"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ（オプション）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

class SearchLogger:
    """MCTS探索用ロガー"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger()
        self._simulation_count = 0

    def log_simulation(self, tree_size: int, avg_reward: float) -> None:
        """Simulation進行状況をログ"""
        self._simulation_count += 1
        if self._simulation_count % 100 == 0:
            self.logger.info(
                f"Simulation {self._simulation_count}: "
                f"Tree size={tree_size}, Avg reward={avg_reward:.4f}"
            )

    def log_batch_eval(self, batch_size: int) -> None:
        """バッチ評価をログ"""
        self.logger.debug(f"Batch evaluation: {batch_size} nodes")

    def log_alert(self, smiles: str, alert_type: str) -> None:
        """Alert判定をログ"""
        self.logger.warning(f"Alert ({alert_type}): {smiles}")

    def log_dead_end(self, smiles: str, depth: int) -> None:
        """行き詰まりをログ"""
        self.logger.debug(f"Dead end at depth {depth}: {smiles}")
```

---

## 13. 使用例

```python
import pandas as pd
from ecmporl_02 import (
    MCTSTree, Environment, Agent,
    SearchConfig, AgentConfig,
    MCTSMode, Constraints
)

# Fragmentテーブルを読み込み
fragment_df = pd.read_csv('fragments.csv')

# 制約条件
constraints = Constraints(
    HAC_min=10, HAC_max=50,
    cnt_hetero_min=1, cnt_hetero_max=10,
    cnt_chiral_min=0, cnt_chiral_max=3,
    MW_min=200.0, MW_max=600.0
)

# Environmentを作成
env = Environment(
    fragment_df=fragment_df,
    combine_fn=my_combine_fn,
    hydrogen_replace_fn=my_hydrogen_replace_fn,
    remove_deuterium_fn=my_remove_deuterium_fn,
    reward_fns=[reward_fn_1, reward_fn_2],
    alert_elem_fn=my_alert_elem_fn,
    alert_mol_fn=my_alert_mol_fn,
    calc_props_fn=my_calc_props_fn,
    count_subspace_fn=my_count_subspace_fn,
    constraints=constraints
)

# 設定
search_config = SearchConfig(
    mode=MCTSMode.PUCT,
    max_depth=8,
    min_depth=3,
    c_puct=1.0,
    batch_eval_interval=256,
    constraints=constraints
)

agent_config = AgentConfig(
    tau_initial=1.0,
    tau_final=0.1,
    train_interval=512,
    batch_size=128
)

# Agentを作成
agent = Agent(
    policy_value_network=my_policy_value_network,
    config=agent_config
)

# MCTSTreeを作成して探索
tree = MCTSTree(
    root_smiles='*c1ccccc1',
    environment=env,
    agent=agent,
    config=search_config
)

tree.search(
    n_simulations=10000,
    batch_eval_interval=256,
    train_interval=512
)

# 結果を取得
high_reward_nodes = tree.filter_nodes(q_min=0.7)
for node in high_reward_nodes:
    print(f"{node.leaf_smiles}: Q={node.q_value:.4f}")

# 保存
from ecmporl_02.io import save_tree
save_tree(tree, 'result_tree.pkl.gz')
agent.save('agent_weights.pt')
```

---

以上が ECMPORL_02 パッケージの設計仕様書 v4.0。
