# ECMPORL 設計仕様書 v2

## 1. 文書情報

- 名称: ECMPORL 設計仕様書
- バージョン: v2.0
- 作成日: 2024年
- 対応要件定義書: ECMPORL_requirements_v2.md
- 対象読者:
  - 本パッケージの実装担当エンジニア
  - コードレビュー担当者

---

## 2. パッケージ構成

### 2.1 ディレクトリ構造

```
ecmporl_02/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── mcts_node.py        # MCTSNode クラス
│   ├── mcts_tree.py        # MCTSTree クラス
│   └── types.py            # 型定義（TypedDict, Enum, Protocol）
├── env/
│   ├── __init__.py
│   ├── environment.py      # Environment クラス
│   └── constraints.py      # 制約管理クラス
├── agent/
│   ├── __init__.py
│   ├── agent.py            # Agent クラス
│   ├── selection.py        # UCT/PUCT Selection ロジック
│   └── scheduler.py        # 温度スケジューラ
├── io/
│   ├── __init__.py
│   └── serialization.py    # シリアライズ/デシリアライズ
├── config.py               # 設定クラス（AgentConfig等）
├── inference.py            # 推論専用ユーティリティ
└── logging_utils.py        # ロギングユーティリティ
```

### 2.2 依存関係図

```
                    ┌─────────────┐
                    │   Agent     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Environment │ │  MCTSTree   │ │ユーザモデル │
    └──────┬──────┘ └──────┬──────┘ └─────────────┘
           │               │
           │               ▼
           │        ┌─────────────┐
           │        │  MCTSNode   │
           │        └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Constraints │
    └─────────────┘
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
    EVALUATED = "evaluated"    # 評価済み

class SchedulerType(Enum):
    """温度スケジューラの種類"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
```

### 3.2 TypedDict 定義

```python
from typing import TypedDict, Optional, Any
import numpy as np
import torch

class MolProps(TypedDict):
    """分子プロパティ"""
    HAC: int
    cnt_hetero: int
    cnt_chiral: int
    MW: float

class FragmentInfo(TypedDict):
    """Fragment情報"""
    index: int              # Fragment テーブル内のインデックス
    smiles: str             # ダミーアトム付きSMILES
    HAC: int
    cnt_hetero: int
    cnt_chiral: int
    MW: float

class Constraints(TypedDict, total=False):
    """制約条件"""
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

class ChildInfo(TypedDict):
    """子ノード情報（学習データ抽出用）"""
    fragment_index: int     # 選択されたFragmentのインデックス
    visit_count: int        # その子ノードの訪問回数
    child_node_id: str      # 子ノードのID

class TrainingData(TypedDict):
    """学習データ"""
    state_smiles: str
    policy_target: np.ndarray  # shape: (N,) N=Fragment数
    value_target: float

class SerializedNode(TypedDict):
    """シリアライズされたノード"""
    state_smiles: str
    depth: int
    leaf_smiles: Optional[str]
    leaf_calc: str
    visit_count: int
    total_reward: float
    incoming_action: Optional[int]  # Fragment index
    is_terminal: bool
    num_sub: int
    children_ids: list[str]
    metadata: Optional[dict]
```

### 3.3 Protocol 定義（ユーザ提供コンポーネント）

```python
from typing import Protocol, runtime_checkable

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
        重水素原子をダミーアトムに置換した全パターンを列挙
        """
        ...

@runtime_checkable
class RemoveDeuteriumFunc(Protocol):
    """重水素除去関数のインターフェース"""
    def __call__(self, smiles_with_deuterium: str) -> str:
        """
        重水素を通常の水素に変換し、水素除去したSMILESを返す
        """
        ...

@runtime_checkable
class RewardFunc(Protocol):
    """報酬関数のインターフェース"""
    def __call__(self, smiles: str | list[str]) -> float | list[float]:
        """
        SMILES（単一または複数）を受け取り、[0,1]の報酬を返す
        """
        ...

@runtime_checkable
class AlertElemFunc(Protocol):
    """Alert判定関数（状態用）のインターフェース"""
    def __call__(self, smiles: str) -> int:
        """
        ダミーアトム付きSMILESのAlert判定
        Returns: 1=OK, 0=Alert
        """
        ...

@runtime_checkable
class AlertMolFunc(Protocol):
    """Alert判定関数（Leaf用）のインターフェース"""
    def __call__(self, smiles: str) -> int:
        """
        完成分子のAlert判定
        Returns: 1=OK, 0=Alert
        """
        ...

@runtime_checkable
class MeasureMolPropsFunc(Protocol):
    """分子プロパティ計測関数のインターフェース"""
    def __call__(self, smiles: str) -> MolProps:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"無効な SMILES です: {smiles}")
        # HAC (Heavy Atom Count)
        hac = mol.GetNumHeavyAtoms()
        # MW
        mw = Descriptors.MolWt(mol)
        # Chiral center
        chiral_centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        cnt_chiral = len(chiral_centers)
        # cnt_hetero
        cnt_hetero = sum(
            1 for atom in mol.GetAtoms() if atom.GetSymbol() in {"N", "O", "S", "P"}
        )

        return MolProps(
            HAC=hac,
            cnt_hetero=cnt_hetero,
            cnt_chiral=cnt_chiral,
            MW=float(mw),
        )

@runtime_checkable
class CountSubspaceFunc(Protocol):
    """化学サブスペースサイズ算出関数のインターフェース"""
    def __call__(self, state_smiles: str) -> int:
        """
        そのノードの先に存在し得る全化合物数を返す
        """
        ...

@runtime_checkable
class PolicyValueModel(Protocol):
    """統合モデル（Policy head + Value head）のインターフェース"""
    def __call__(self, state_smiles: str) -> tuple[torch.Tensor, float]:
        """
        Args:
            state_smiles: ダミーアトム付きSMILES
        Returns:
            policy_logits: shape (N,) 全Fragmentに対するlogits
            value: 状態価値スカラー
        """
        ...

    def batch_forward(
        self, state_smiles_list: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        バッチ推論
        Args:
            state_smiles_list: SMILESのリスト
        Returns:
            policy_logits: shape (batch, N)
            values: shape (batch,)
        """
        ...
```

---

## 4. MCTSNode クラス設計 (`core/mcts_node.py`)

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
    state_smiles: str                          # canonical SMILES（ダミーアトム付き）
    depth: int                                  # ルートからの深さ（root=0）
    leaf_smiles: Optional[str] = None          # Leaf SMILES（重水素なし完成分子）
    leaf_calc: LeafCalcStatus = LeafCalcStatus.NOT_READY

    # === 統計量 ===
    visit_count: int = 0                       # N: 訪問回数
    total_reward: float = 0.0                  # W: 累計報酬

    # === 構造情報 ===
    parent: Optional['MCTSNode'] = field(default=None, repr=False)
    incoming_action: Optional[int] = None      # このノードへの遷移に使用したFragment index
    children: dict[int, 'MCTSNode'] = field(default_factory=dict)  # key: fragment_index
    is_terminal: bool = False                  # 展開禁止フラグ
    num_sub: int = 0                           # 化学サブスペースサイズ

    # === その他 ===
    metadata: Optional[dict] = None            # 補助情報
    _mol_props: Optional[MolProps] = field(default=None, repr=False)  # キャッシュ

    # === プロパティ ===
    @property
    def q_value(self) -> float:
        """平均報酬 Q = W / N"""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def node_id(self) -> str:
        """ノードの一意識別子（state_smiles + depth）"""
        return f"{self.state_smiles}@{self.depth}"

    @property
    def is_root(self) -> bool:
        """ルートノードかどうか"""
        return self.parent is None

    @property
    def is_leaf_node(self) -> bool:
        """葉ノード（子を持たない）かどうか"""
        return len(self.children) == 0

    @property
    def is_expandable(self) -> bool:
        """展開可能かどうか"""
        return not self.is_terminal

    # === メソッド ===
    def add_child(self, fragment_index: int, child_node: 'MCTSNode') -> None:
        """
        子ノードを追加する。

        Args:
            fragment_index: 選択されたFragmentのインデックス
            child_node: 子ノード
        """
        if fragment_index in self.children:
            raise ValueError(f"Fragment {fragment_index} already has a child")
        child_node.parent = self
        child_node.incoming_action = fragment_index
        self.children[fragment_index] = child_node

    def get_child(self, fragment_index: int) -> Optional['MCTSNode']:
        """
        指定されたFragmentに対応する子ノードを取得する。

        Args:
            fragment_index: Fragmentのインデックス

        Returns:
            子ノード、存在しない場合はNone
        """
        return self.children.get(fragment_index)

    def has_child(self, fragment_index: int) -> bool:
        """指定されたFragmentに対応する子ノードが存在するか"""
        return fragment_index in self.children

    def get_children_visit_distribution(self) -> dict[int, float]:
        """
        子ノードの訪問回数分布を取得する。
        Policy学習のターゲットとして使用。

        Returns:
            {fragment_index: 訪問比率} の辞書
        """
        if not self.children:
            return {}

        total_visits = sum(child.visit_count for child in self.children.values())
        if total_visits == 0:
            return {idx: 0.0 for idx in self.children.keys()}

        return {
            idx: child.visit_count / total_visits
            for idx, child in self.children.items()
        }

    def update_stats(self, reward: float) -> None:
        """
        統計量を更新する（Backpropagation用）。

        Args:
            reward: 逆伝播する報酬
        """
        self.visit_count += 1
        self.total_reward += reward

    def get_path_to_root(self) -> list['MCTSNode']:
        """
        このノードからルートまでのパスを取得する。

        Returns:
            [self, parent, ..., root] の順のリスト
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path

    def set_mol_props(self, props: MolProps) -> None:
        """分子プロパティをキャッシュする"""
        self._mol_props = props

    def get_mol_props(self) -> Optional[MolProps]:
        """キャッシュされた分子プロパティを取得"""
        return self._mol_props

    def to_dict(self) -> dict:
        """
        シリアライズ用の辞書に変換する。
        親・子への参照は含めない（別途管理）。
        """
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
        """
        辞書からノードを復元する。
        親・子への参照は別途設定が必要。
        """
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

## 5. MCTSTree クラス設計 (`core/mcts_tree.py`)

### 5.1 クラス定義

```python
from typing import Optional, Iterator, Callable
from dataclasses import dataclass, field
import numpy as np

from .mcts_node import MCTSNode
from .types import TrainingData, Constraints

@dataclass
class MCTSTree:
    """
    MCTS探索木全体を管理するクラス。
    ノードのインデックス管理、検索、学習データ抽出を担当する。
    """

    root: MCTSNode
    _node_index: dict[str, MCTSNode] = field(default_factory=dict, repr=False)
    _num_fragments: int = 0  # Fragment数（Policy target生成用）

    def __post_init__(self):
        """初期化後処理：ルートノードをインデックスに登録"""
        self._register_node(self.root)

    # === ノード管理 ===
    def _register_node(self, node: MCTSNode) -> None:
        """ノードをインデックスに登録"""
        self._node_index[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """
        ノードIDからノードを取得する。

        Args:
            node_id: "state_smiles@depth" 形式のID

        Returns:
            ノード、存在しない場合はNone
        """
        return self._node_index.get(node_id)

    def get_node_by_state(self, state_smiles: str, depth: int) -> Optional[MCTSNode]:
        """
        状態SMILESとdepthからノードを取得する。

        Args:
            state_smiles: canonical SMILES
            depth: 深さ

        Returns:
            ノード、存在しない場合はNone
        """
        node_id = f"{state_smiles}@{depth}"
        return self.get_node(node_id)

    def has_node(self, state_smiles: str, depth: int) -> bool:
        """指定された状態のノードが存在するか"""
        return self.get_node_by_state(state_smiles, depth) is not None

    def add_node(
        self,
        parent: MCTSNode,
        fragment_index: int,
        child_state_smiles: str,
        depth: int,
        **kwargs
    ) -> MCTSNode:
        """
        新しいノードを追加する。
        同一状態のノードが既に存在する場合は既存ノードを返す（トランスポジション）。

        Args:
            parent: 親ノード
            fragment_index: 選択されたFragmentのインデックス
            child_state_smiles: 子ノードの状態SMILES
            depth: 子ノードの深さ
            **kwargs: MCTSNodeの追加引数

        Returns:
            追加または既存のノード
        """
        # トランスポジションチェック
        existing = self.get_node_by_state(child_state_smiles, depth)
        if existing is not None:
            # 既存ノードを親の子として登録（別経路からの到達）
            if not parent.has_child(fragment_index):
                parent.add_child(fragment_index, existing)
            return existing

        # 新規ノード作成
        new_node = MCTSNode(
            state_smiles=child_state_smiles,
            depth=depth,
            **kwargs
        )
        parent.add_child(fragment_index, new_node)
        self._register_node(new_node)

        return new_node

    # === プロパティ ===
    @property
    def size(self) -> int:
        """ノード総数"""
        return len(self._node_index)

    @property
    def num_fragments(self) -> int:
        """Fragment数"""
        return self._num_fragments

    @num_fragments.setter
    def num_fragments(self, value: int) -> None:
        """Fragment数を設定"""
        self._num_fragments = value

    # === イテレーション ===
    def __iter__(self) -> Iterator[MCTSNode]:
        """全ノードをイテレート"""
        return iter(self._node_index.values())

    def iter_nodes(
        self,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None
    ) -> Iterator[MCTSNode]:
        """
        条件付きでノードをイテレートする。

        Args:
            min_depth: 最小深さ（この深さ以上）
            max_depth: 最大深さ（この深さ以下）
        """
        for node in self._node_index.values():
            if min_depth is not None and node.depth < min_depth:
                continue
            if max_depth is not None and node.depth > max_depth:
                continue
            yield node

    # === 検索・フィルタリング ===
    def filter_nodes(
        self,
        q_min: Optional[float] = None,
        total_reward_min: Optional[float] = None,
        num_sub_min: Optional[int] = None,
        depth_range: Optional[tuple[int, int]] = None,
        visit_count_min: Optional[int] = None,
        custom_filter: Optional[Callable[[MCTSNode], bool]] = None
    ) -> list[MCTSNode]:
        """
        条件を満たすノードを抽出する。

        Args:
            q_min: 最小Q値
            total_reward_min: 最小累計報酬
            num_sub_min: 最小サブスペースサイズ
            depth_range: (min_depth, max_depth) のタプル
            visit_count_min: 最小訪問回数
            custom_filter: カスタムフィルタ関数

        Returns:
            条件を満たすノードのリスト
        """
        results = []

        for node in self._node_index.values():
            # Q値フィルタ
            if q_min is not None and node.q_value < q_min:
                continue

            # 累計報酬フィルタ
            if total_reward_min is not None and node.total_reward < total_reward_min:
                continue

            # サブスペースサイズフィルタ
            if num_sub_min is not None and node.num_sub < num_sub_min:
                continue

            # 深さフィルタ
            if depth_range is not None:
                if node.depth < depth_range[0] or node.depth > depth_range[1]:
                    continue

            # 訪問回数フィルタ
            if visit_count_min is not None and node.visit_count < visit_count_min:
                continue

            # カスタムフィルタ
            if custom_filter is not None and not custom_filter(node):
                continue

            results.append(node)

        return results

    def get_top_nodes(
        self,
        n: int,
        key: str = 'q_value',
        reverse: bool = True
    ) -> list[MCTSNode]:
        """
        指定されたキーでソートした上位n件のノードを取得する。

        Args:
            n: 取得件数
            key: ソートキー ('q_value', 'visit_count', 'total_reward', 'num_sub')
            reverse: 降順かどうか

        Returns:
            上位n件のノード
        """
        all_nodes = list(self._node_index.values())

        if key == 'q_value':
            sorted_nodes = sorted(all_nodes, key=lambda x: x.q_value, reverse=reverse)
        elif key == 'visit_count':
            sorted_nodes = sorted(all_nodes, key=lambda x: x.visit_count, reverse=reverse)
        elif key == 'total_reward':
            sorted_nodes = sorted(all_nodes, key=lambda x: x.total_reward, reverse=reverse)
        elif key == 'num_sub':
            sorted_nodes = sorted(all_nodes, key=lambda x: x.num_sub, reverse=reverse)
        else:
            raise ValueError(f"Unknown key: {key}")

        return sorted_nodes[:n]

    # === 学習データ抽出 ===
    def extract_training_data(
        self,
        q_min: Optional[float] = None,
        visit_count_min: int = 1,
        include_policy: bool = True,
        sample_ratio: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> list[TrainingData]:
        """
        統合モデル学習用のデータを抽出する。

        Args:
            q_min: 最小Q値（この値以上のノードを優先）
            visit_count_min: 最小訪問回数
            include_policy: Policyターゲットを含めるか
            sample_ratio: サンプリング比率（Noneで全件）
            random_state: 乱数シード

        Returns:
            学習データのリスト
        """
        if self._num_fragments == 0:
            raise ValueError("num_fragments must be set before extracting training data")

        rng = np.random.default_rng(random_state)
        training_data = []

        for node in self._node_index.values():
            # 訪問回数フィルタ
            if node.visit_count < visit_count_min:
                continue

            # Q値フィルタ（優先サンプリング用）
            if q_min is not None and node.q_value < q_min:
                # q_min未満でもサンプリングはするが、確率を下げる
                if sample_ratio is not None and rng.random() > 0.1:
                    continue

            # サンプリング
            if sample_ratio is not None and rng.random() > sample_ratio:
                continue

            # Policy target の構築
            policy_target = np.zeros(self._num_fragments, dtype=np.float32)
            if include_policy and node.children:
                visit_dist = node.get_children_visit_distribution()
                for frag_idx, ratio in visit_dist.items():
                    if 0 <= frag_idx < self._num_fragments:
                        policy_target[frag_idx] = ratio

            training_data.append(TrainingData(
                state_smiles=node.state_smiles,
                policy_target=policy_target,
                value_target=node.q_value
            ))

        return training_data

    # === 統計情報 ===
    def get_statistics(self) -> dict:
        """
        ツリー全体の統計情報を取得する。

        Returns:
            統計情報の辞書
        """
        if self.size == 0:
            return {}

        q_values = [n.q_value for n in self._node_index.values() if n.visit_count > 0]
        visit_counts = [n.visit_count for n in self._node_index.values()]
        depths = [n.depth for n in self._node_index.values()]

        return {
            'total_nodes': self.size,
            'max_depth': max(depths),
            'avg_depth': np.mean(depths),
            'total_visits': self.root.visit_count,
            'avg_q_value': np.mean(q_values) if q_values else 0.0,
            'max_q_value': max(q_values) if q_values else 0.0,
            'min_q_value': min(q_values) if q_values else 0.0,
            'avg_visit_count': np.mean(visit_counts),
            'terminal_nodes': sum(1 for n in self._node_index.values() if n.is_terminal),
            'evaluated_nodes': sum(
                1 for n in self._node_index.values()
                if n.leaf_calc == LeafCalcStatus.EVALUATED
            ),
        }

    # === シリアライズ ===
    def to_dict(self) -> dict:
        """
        シリアライズ用の辞書に変換する。
        """
        nodes_data = {}
        edges_data = []

        for node in self._node_index.values():
            nodes_data[node.node_id] = node.to_dict()

            # エッジ情報（親→子）
            for frag_idx, child in node.children.items():
                edges_data.append({
                    'parent_id': node.node_id,
                    'child_id': child.node_id,
                    'fragment_index': frag_idx
                })

        return {
            'root_id': self.root.node_id,
            'num_fragments': self._num_fragments,
            'nodes': nodes_data,
            'edges': edges_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MCTSTree':
        """
        辞書からツリーを復元する。
        """
        # ノードを復元
        nodes = {}
        for node_id, node_data in data['nodes'].items():
            nodes[node_id] = MCTSNode.from_dict(node_data)

        # エッジを復元（親子関係の再構築）
        for edge in data['edges']:
            parent = nodes[edge['parent_id']]
            child = nodes[edge['child_id']]
            frag_idx = edge['fragment_index']
            parent.children[frag_idx] = child
            child.parent = parent

        # ルートノードを取得
        root = nodes[data['root_id']]

        # ツリーを構築
        tree = cls(root=root)
        tree._node_index = nodes
        tree._num_fragments = data.get('num_fragments', 0)

        return tree
```

---

## 6. Environment クラス設計 (`env/environment.py`)

### 6.1 クラス定義

```python
from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd
import numpy as np

from ..core.types import (
    MolProps, Constraints, FragmentInfo,
    CombineSmilesFunc, HydrogenReplaceFunc, RemoveDeuteriumFunc,
    RewardFunc, AlertElemFunc, AlertMolFunc,
    MeasureMolPropsFunc, CountSubspaceFunc
)

@dataclass
class Environment:
    """
    化学的な制約・評価関数・Fragment情報を管理するクラス。
    MCTSの状態遷移と報酬計算を担当する。
    """

    # === Fragment管理 ===
    fragment_df: pd.DataFrame                # Fragment テーブル

    # === ユーザ提供関数 ===
    combine_smiles_fn: CombineSmilesFunc
    hydrogen_replace_fn: HydrogenReplaceFunc
    remove_deuterium_fn: RemoveDeuteriumFunc
    reward_fns: list[RewardFunc]
    alert_elem_fn: AlertElemFunc
    alert_mol_fn: AlertMolFunc
    measure_mol_props_fn: MeasureMolPropsFunc
    count_subspace_fn: CountSubspaceFunc

    # === 制約条件 ===
    constraints: Constraints = field(default_factory=dict)

    # === キャッシュ ===
    _fragment_info_cache: list[FragmentInfo] = field(default_factory=list, repr=False)
    _valid_fragment_cache: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """初期化後処理：Fragment情報のキャッシュ構築"""
        self._build_fragment_cache()
        self._validate_constraints()

    def _build_fragment_cache(self) -> None:
        """Fragmentテーブルからキャッシュを構築"""
        self._fragment_info_cache = []

        required_cols = ['smiles', 'HAC', 'cnt_hetero', 'cnt_chiral', 'MW']
        for col in required_cols:
            if col not in self.fragment_df.columns:
                raise ValueError(f"Fragment table must have column: {col}")

        for idx, row in self.fragment_df.iterrows():
            self._fragment_info_cache.append(FragmentInfo(
                index=idx,
                smiles=row['smiles'],
                HAC=row['HAC'],
                cnt_hetero=row['cnt_hetero'],
                cnt_chiral=row['cnt_chiral'],
                MW=row['MW']
            ))

    def _validate_constraints(self) -> None:
        """制約条件の妥当性を検証"""
        for key in self.constraints:
            if key not in [
                'HAC_min', 'HAC_max',
                'cnt_hetero_min', 'cnt_hetero_max',
                'cnt_chiral_min', 'cnt_chiral_max',
                'MW_min', 'MW_max'
            ]:
                raise ValueError(f"Unknown constraint key: {key}")

    # === プロパティ ===
    @property
    def num_fragments(self) -> int:
        """Fragment数"""
        return len(self._fragment_info_cache)

    def get_fragment_info(self, index: int) -> FragmentInfo:
        """指定インデックスのFragment情報を取得"""
        return self._fragment_info_cache[index]

    def get_all_fragment_smiles(self) -> list[str]:
        """全FragmentのSMILESリストを取得"""
        return [f['smiles'] for f in self._fragment_info_cache]

    # === 分子プロパティ ===
    def calc_mol_props(self, smiles: str) -> MolProps:
        """
        分子プロパティを計算する。

        Args:
            smiles: 計算対象のSMILES（ダミーアトム付き可）

        Returns:
            分子プロパティ
        """
        return self.measure_mol_props_fn(smiles)

    # === 制約チェック ===
    def check_upper_constraints(
        self,
        current_props: MolProps,
        fragment_props: FragmentInfo
    ) -> bool:
        """
        Fragment追加後も上限制約を満たすかチェックする。

        Args:
            current_props: 現在の状態のプロパティ
            fragment_props: 追加するFragmentのプロパティ

        Returns:
            制約を満たす場合True
        """
        # HAC上限
        if 'HAC_max' in self.constraints:
            if current_props['HAC'] + fragment_props['HAC'] > self.constraints['HAC_max']:
                return False

        # cnt_hetero上限
        if 'cnt_hetero_max' in self.constraints:
            if current_props['cnt_hetero'] + fragment_props['cnt_hetero'] > self.constraints['cnt_hetero_max']:
                return False

        # cnt_chiral上限
        if 'cnt_chiral_max' in self.constraints:
            if current_props['cnt_chiral'] + fragment_props['cnt_chiral'] > self.constraints['cnt_chiral_max']:
                return False

        # MW上限
        if 'MW_max' in self.constraints:
            if current_props['MW'] + fragment_props['MW'] > self.constraints['MW_max']:
                return False

        return True

    def check_lower_constraints(self, props: MolProps) -> bool:
        """
        下限制約を満たすかチェックする（Leaf評価可否判定用）。

        Args:
            props: チェック対象のプロパティ

        Returns:
            制約を満たす場合True
        """
        if 'HAC_min' in self.constraints:
            if props['HAC'] < self.constraints['HAC_min']:
                return False

        if 'cnt_hetero_min' in self.constraints:
            if props['cnt_hetero'] < self.constraints['cnt_hetero_min']:
                return False

        if 'cnt_chiral_min' in self.constraints:
            if props['cnt_chiral'] < self.constraints['cnt_chiral_min']:
                return False

        if 'MW_min' in self.constraints:
            if props['MW'] < self.constraints['MW_min']:
                return False

        return True

    # === 有効行動の列挙 ===
    def get_valid_actions(
        self,
        state_smiles: str,
        use_cache: bool = True
    ) -> list[int]:
        """
        現在の状態から選択可能なFragmentのインデックスリストを返す。

        Args:
            state_smiles: 現在の状態SMILES
            use_cache: キャッシュを使用するか

        Returns:
            有効なFragmentインデックスのリスト
        """
        # キャッシュチェック
        if use_cache and state_smiles in self._valid_fragment_cache:
            return self._valid_fragment_cache[state_smiles]

        # 現在の状態のプロパティを計算
        current_props = self.calc_mol_props(state_smiles)

        valid_indices = []
        for frag_info in self._fragment_info_cache:
            # 上限制約チェック
            if self.check_upper_constraints(current_props, frag_info):
                valid_indices.append(frag_info['index'])

        # キャッシュに保存
        if use_cache:
            self._valid_fragment_cache[state_smiles] = valid_indices

        return valid_indices

    def create_action_mask(self, state_smiles: str) -> np.ndarray:
        """
        Policyモデル用の行動マスクを作成する。

        Args:
            state_smiles: 現在の状態SMILES

        Returns:
            shape (num_fragments,) のbool配列。有効な行動はTrue。
        """
        mask = np.zeros(self.num_fragments, dtype=bool)
        valid_actions = self.get_valid_actions(state_smiles)
        mask[valid_actions] = True
        return mask

    # === 状態遷移 ===
    def step(
        self,
        state_smiles: str,
        fragment_index: int
    ) -> list[str]:
        """
        状態遷移を実行し、次の状態候補リストを返す。

        Args:
            state_smiles: 現在の状態SMILES
            fragment_index: 選択するFragmentのインデックス

        Returns:
            次の状態候補SMILESのリスト（ダミーアトム付き）
        """
        fragment_smiles = self._fragment_info_cache[fragment_index]['smiles']

        # 結合（ダミーアトムなしSMILESが返る）
        combined = self.combine_smiles_fn(state_smiles, fragment_smiles)

        # 重水素→ダミーアトム置換で次状態候補を列挙
        next_states = self.hydrogen_replace_fn(combined)

        return next_states

    def generate_leaf_smiles(self, state_smiles: str) -> str:
        """
        状態SMILESからLeaf SMILES（完成分子）を生成する。

        Args:
            state_smiles: 状態SMILES（ダミーアトム付き、重水素あり）

        Returns:
            Leaf SMILES（ダミーアトムなし、重水素なし）
        """
        return self.remove_deuterium_fn(state_smiles)

    # === Alert判定 ===
    def is_alert_elem(self, smiles: str) -> bool:
        """
        状態（ダミーアトム付きSMILES）のAlert判定。

        Args:
            smiles: 判定対象のSMILES

        Returns:
            AlertならTrue
        """
        return self.alert_elem_fn(smiles) == 0

    def is_alert_mol(self, smiles: str) -> bool:
        """
        Leaf（完成分子SMILES）のAlert判定。

        Args:
            smiles: 判定対象のSMILES

        Returns:
            AlertならTrue
        """
        return self.alert_mol_fn(smiles) == 0

    # === サブスペースサイズ ===
    def calc_num_sub(self, state_smiles: str) -> int:
        """
        化学サブスペースサイズを算出する。

        Args:
            state_smiles: 状態SMILES

        Returns:
            サブスペースサイズ
        """
        return self.count_subspace_fn(state_smiles)

    # === 報酬計算 ===
    def calc_reward(self, leaf_smiles: str) -> float:
        """
        Leaf SMILESの報酬を計算する（相乗平均）。

        Args:
            leaf_smiles: Leaf SMILES（完成分子）

        Returns:
            報酬値 [0, 1]
        """
        # Alert判定
        if self.is_alert_mol(leaf_smiles):
            return 0.0

        # 各報酬関数を評価
        rewards = []
        for reward_fn in self.reward_fns:
            r = reward_fn(leaf_smiles)
            if isinstance(r, list):
                r = r[0]
            rewards.append(max(0.0, min(1.0, r)))  # [0, 1]にクリップ

        if not rewards:
            return 0.0

        # 相乗平均
        product = 1.0
        for r in rewards:
            product *= r

        return product ** (1.0 / len(rewards))

    def calc_reward_batch(self, leaf_smiles_list: list[str]) -> list[float]:
        """
        複数のLeaf SMILESの報酬をバッチ計算する。

        Args:
            leaf_smiles_list: Leaf SMILESのリスト

        Returns:
            報酬値のリスト
        """
        return [self.calc_reward(s) for s in leaf_smiles_list]

    # === キャッシュ管理 ===
    def clear_cache(self) -> None:
        """有効行動キャッシュをクリアする"""
        self._valid_fragment_cache.clear()
```

---

## 7. Agent クラス設計 (`agent/agent.py`)

### 7.1 設定クラス (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
from .core.types import MCTSMode, SchedulerType

@dataclass
class AgentConfig:
    """Agent の設定パラメータ"""

    # === 探索パラメータ ===
    max_depth: int = 10                        # 最大探索深さ
    min_depth: int = 3                         # Leaf評価可能な最小深さ
    max_simulation: int = 10000                # シミュレーション回数

    # === MCTS モード ===
    mcts_mode: MCTSMode = MCTSMode.PUCT
    c_uct: float = 1.414                       # UCT探索定数
    c_puct: float = 1.0                        # PUCT探索定数

    # === 学習関連 ===
    train_interval: int = 100                  # 学習間隔（シミュレーション数）
    batch_size: int = 64                       # ミニバッチサイズ
    learning_rate: float = 0.001               # 学習率
    weight_decay: float = 1e-4                 # L2正則化係数
    q_threshold_for_training: Optional[float] = None  # 高報酬ノード優先サンプリング閾値

    # === 温度スケジューラ ===
    tau_initial: float = 1.0                   # 初期温度
    tau_final: float = 0.1                     # 最終温度
    tau_scheduler_type: SchedulerType = SchedulerType.LINEAR
    tau_decay_steps: Optional[int] = None      # 減衰ステップ数（Noneでmax_simulation）

    # === その他 ===
    device: str = 'cuda'                       # PyTorchデバイス
    seed: Optional[int] = None                 # 乱数シード
    enable_logging: bool = True                # ロギング有効化
    log_interval: int = 100                    # ログ出力間隔
```

### 7.2 温度スケジューラ (`agent/scheduler.py`)

```python
from abc import ABC, abstractmethod
import math
from ..core.types import SchedulerType

class TemperatureScheduler(ABC):
    """温度スケジューラの基底クラス"""

    @abstractmethod
    def get_temperature(self, step: int) -> float:
        """現在のステップに対応する温度を返す"""
        pass

class LinearScheduler(TemperatureScheduler):
    """線形減衰スケジューラ"""

    def __init__(self, tau_initial: float, tau_final: float, total_steps: int):
        self.tau_initial = tau_initial
        self.tau_final = tau_final
        self.total_steps = total_steps

    def get_temperature(self, step: int) -> float:
        if step >= self.total_steps:
            return self.tau_final

        progress = step / self.total_steps
        return self.tau_initial + (self.tau_final - self.tau_initial) * progress

class ExponentialScheduler(TemperatureScheduler):
    """指数減衰スケジューラ"""

    def __init__(self, tau_initial: float, tau_final: float, total_steps: int):
        self.tau_initial = tau_initial
        self.tau_final = tau_final
        self.total_steps = total_steps

        # 減衰係数を計算: tau_final = tau_initial * exp(-k * total_steps)
        self.k = -math.log(tau_final / tau_initial) / total_steps

    def get_temperature(self, step: int) -> float:
        if step >= self.total_steps:
            return self.tau_final

        return self.tau_initial * math.exp(-self.k * step)

def create_scheduler(
    scheduler_type: SchedulerType,
    tau_initial: float,
    tau_final: float,
    total_steps: int
) -> TemperatureScheduler:
    """スケジューラを作成するファクトリ関数"""
    if scheduler_type == SchedulerType.LINEAR:
        return LinearScheduler(tau_initial, tau_final, total_steps)
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        return ExponentialScheduler(tau_initial, tau_final, total_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
```

### 7.3 Selection ロジック (`agent/selection.py`)

```python
import math
import numpy as np
import torch
from typing import Optional
from ..core.mcts_node import MCTSNode
from ..core.types import MCTSMode

def calc_uct_score(
    node: MCTSNode,
    action: int,
    c_uct: float,
    parent_visits: int
) -> float:
    """
    UCTスコアを計算する。

    score(s,a) = Q(s,a) + c_uct * sqrt(log(N_total + 1) / (1 + N_sa))

    Args:
        node: 親ノード
        action: 行動（Fragment index）
        c_uct: 探索定数
        parent_visits: 親ノードの訪問回数

    Returns:
        UCTスコア
    """
    child = node.get_child(action)

    if child is None:
        # 未訪問の行動は無限大のスコア（優先的に選択）
        return float('inf')

    q_value = child.q_value
    n_sa = child.visit_count

    exploration = c_uct * math.sqrt(math.log(parent_visits + 1) / (1 + n_sa))

    return q_value + exploration

def calc_puct_score(
    node: MCTSNode,
    action: int,
    prior: float,
    c_puct: float,
    parent_visits: int
) -> float:
    """
    PUCTスコアを計算する。

    score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_total + 1) / (1 + N_sa)

    Args:
        node: 親ノード
        action: 行動（Fragment index）
        prior: 事前確率 P(s,a)
        c_puct: 探索定数
        parent_visits: 親ノードの訪問回数

    Returns:
        PUCTスコア
    """
    child = node.get_child(action)

    if child is None:
        q_value = 0.0
        n_sa = 0
    else:
        q_value = child.q_value
        n_sa = child.visit_count

    exploration = c_puct * prior * math.sqrt(parent_visits + 1) / (1 + n_sa)

    return q_value + exploration

def select_action_uct(
    node: MCTSNode,
    valid_actions: list[int],
    c_uct: float
) -> int:
    """
    UCTに基づいて行動を選択する。

    Args:
        node: 現在のノード
        valid_actions: 有効な行動のリスト
        c_uct: 探索定数

    Returns:
        選択された行動（Fragment index）
    """
    if not valid_actions:
        raise ValueError("No valid actions")

    parent_visits = node.visit_count

    best_action = valid_actions[0]
    best_score = float('-inf')

    for action in valid_actions:
        score = calc_uct_score(node, action, c_uct, parent_visits)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action

def select_action_puct(
    node: MCTSNode,
    valid_actions: list[int],
    policy_probs: np.ndarray,
    c_puct: float
) -> int:
    """
    PUCTに基づいて行動を選択する。

    Args:
        node: 現在のノード
        valid_actions: 有効な行動のリスト
        policy_probs: 全行動に対する事前確率（softmax後）
        c_puct: 探索定数

    Returns:
        選択された行動（Fragment index）
    """
    if not valid_actions:
        raise ValueError("No valid actions")

    parent_visits = node.visit_count

    best_action = valid_actions[0]
    best_score = float('-inf')

    for action in valid_actions:
        prior = policy_probs[action]
        score = calc_puct_score(node, action, prior, c_puct, parent_visits)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action

def compute_policy_probs(
    policy_logits: torch.Tensor,
    valid_actions: list[int],
    temperature: float = 1.0
) -> np.ndarray:
    """
    Policyのlogitsから確率分布を計算する（マスキング付き）。

    Args:
        policy_logits: shape (N,) のlogits
        valid_actions: 有効な行動のリスト
        temperature: 温度パラメータ

    Returns:
        shape (N,) の確率分布
    """
    logits = policy_logits.detach().cpu().numpy()

    # マスキング（無効な行動は-inf）
    mask = np.full(len(logits), float('-inf'))
    mask[valid_actions] = 0.0
    masked_logits = logits + mask

    # 温度付きsoftmax
    scaled_logits = masked_logits / temperature

    # 数値安定性のため最大値を引く
    scaled_logits -= np.max(scaled_logits)

    exp_logits = np.exp(scaled_logits)
    probs = exp_logits / np.sum(exp_logits)

    return probs

def select_next_state_by_value(
    next_states: list[str],
    values: np.ndarray,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> tuple[int, str]:
    """
    Value headの出力に基づいて次の状態を選択する。

    Args:
        next_states: 次の状態候補リスト
        values: 各状態のValue（shape: (len(next_states),)）
        temperature: 温度パラメータ
        rng: 乱数生成器

    Returns:
        (選択されたインデックス, 選択された状態SMILES)
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(next_states) == 1:
        return 0, next_states[0]

    # 温度付きsoftmax
    scaled_values = values / temperature
    scaled_values -= np.max(scaled_values)  # 数値安定性
    exp_values = np.exp(scaled_values)
    probs = exp_values / np.sum(exp_values)

    # サンプリング
    selected_idx = rng.choice(len(next_states), p=probs)

    return selected_idx, next_states[selected_idx]
```

### 7.4 Agent クラス本体 (`agent/agent.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..core.mcts_node import MCTSNode
from ..core.mcts_tree import MCTSTree
from ..core.types import LeafCalcStatus, MCTSMode, PolicyValueModel
from ..env.environment import Environment
from ..config import AgentConfig
from .scheduler import create_scheduler, TemperatureScheduler
from .selection import (
    select_action_uct,
    select_action_puct,
    compute_policy_probs,
    select_next_state_by_value
)

logger = logging.getLogger(__name__)

@dataclass
class Agent:
    """
    MCTS探索を制御するエージェント。
    統合モデルの学習も担当する。
    """

    env: Environment
    model: PolicyValueModel
    config: AgentConfig

    # === 内部状態 ===
    tree: Optional[MCTSTree] = field(default=None, repr=False)
    optimizer: Optional[optim.Optimizer] = field(default=None, repr=False)
    scheduler: Optional[TemperatureScheduler] = field(default=None, repr=False)

    _current_step: int = field(default=0, repr=False)
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        """初期化後処理"""
        # 乱数生成器
        self._rng = np.random.default_rng(self.config.seed)

        # 温度スケジューラ
        total_steps = self.config.tau_decay_steps or self.config.max_simulation
        self.scheduler = create_scheduler(
            self.config.tau_scheduler_type,
            self.config.tau_initial,
            self.config.tau_final,
            total_steps
        )

        # Optimizer（モデルがnn.Moduleの場合）
        if isinstance(self.model, nn.Module):
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

    # === 探索実行 ===
    def run_mcts(self, core_smiles: str) -> MCTSTree:
        """
        MCTSを実行する。

        Args:
            core_smiles: コア構造のSMILES（ダミーアトム付き）

        Returns:
            探索後のMCTSTree
        """
        # ルートノード作成
        root = self._create_root_node(core_smiles)
        self.tree = MCTSTree(root=root)
        self.tree.num_fragments = self.env.num_fragments

        self._current_step = 0

        # シミュレーションループ
        for sim in range(self.config.max_simulation):
            self._current_step = sim

            # 1回のシミュレーション
            self._run_simulation()

            # 学習（一定間隔）
            if (sim + 1) % self.config.train_interval == 0:
                self._train_model()

            # ログ出力
            if self.config.enable_logging and (sim + 1) % self.config.log_interval == 0:
                self._log_progress(sim + 1)

        return self.tree

    def _create_root_node(self, core_smiles: str) -> MCTSNode:
        """ルートノードを作成する"""
        mol_props = self.env.calc_mol_props(core_smiles)
        num_sub = self.env.calc_num_sub(core_smiles)

        # leaf_calcの初期化
        leaf_calc = LeafCalcStatus.NOT_READY
        if self.config.min_depth == 0 and self.env.check_lower_constraints(mol_props):
            leaf_calc = LeafCalcStatus.READY

        node = MCTSNode(
            state_smiles=core_smiles,
            depth=0,
            leaf_calc=leaf_calc,
            num_sub=num_sub
        )
        node.set_mol_props(mol_props)

        return node

    def _run_simulation(self) -> None:
        """
        1回のMCTSシミュレーションを実行する。
        Selection → Expansion → Rollout → Backpropagation
        """
        # === Selection ===
        node = self._select(self.tree.root)

        # === Rollout（深さ優先展開 + Leaf評価） ===
        reward, path = self._rollout(node)

        # === Backpropagation ===
        self._backpropagate(path, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection: 展開対象のノードを選択する。

        Args:
            node: 探索開始ノード

        Returns:
            展開対象のノード
        """
        while not node.is_leaf_node and not node.is_terminal:
            valid_actions = self.env.get_valid_actions(node.state_smiles)

            if not valid_actions:
                break

            # 未展開の行動があれば優先
            unexpanded = [a for a in valid_actions if not node.has_child(a)]
            if unexpanded:
                # 未展開の行動を選んで展開へ
                break

            # 全ての行動が展開済みの場合、UCT/PUCTで選択
            action = self._select_action(node, valid_actions)
            child = node.get_child(action)
            if child is None:
                break
            node = child

        return node

    def _select_action(self, node: MCTSNode, valid_actions: list[int]) -> int:
        """
        UCT/PUCTに基づいて行動を選択する。

        Args:
            node: 現在のノード
            valid_actions: 有効な行動のリスト

        Returns:
            選択された行動（Fragment index）
        """
        if self.config.mcts_mode == MCTSMode.UCT:
            return select_action_uct(node, valid_actions, self.config.c_uct)

        else:  # PUCT
            # Policy headからpriorを取得
            with torch.no_grad():
                policy_logits, _ = self.model(node.state_smiles)

            tau = self.scheduler.get_temperature(self._current_step)
            policy_probs = compute_policy_probs(policy_logits, valid_actions, tau)

            return select_action_puct(
                node, valid_actions, policy_probs, self.config.c_puct
            )

    def _rollout(self, node: MCTSNode) -> tuple[float, list[MCTSNode]]:
        """
        Rollout: 深さ優先展開とLeaf評価を行う。

        Args:
            node: 展開開始ノード

        Returns:
            (報酬, 経路ノードリスト)
        """
        path = [node]
        reward = 0.0
        evaluated = False

        current = node

        while True:
            # === Leaf評価チェック ===
            if current.leaf_calc == LeafCalcStatus.READY and not evaluated:
                # Leaf評価を実行
                leaf_smiles = self.env.generate_leaf_smiles(current.state_smiles)
                current.leaf_smiles = leaf_smiles
                reward = self.env.calc_reward(leaf_smiles)
                current.leaf_calc = LeafCalcStatus.EVALUATED
                evaluated = True

            # === 展開継続判定 ===
            if current.is_terminal:
                break

            if current.depth >= self.config.max_depth:
                current.is_terminal = True
                break

            valid_actions = self.env.get_valid_actions(current.state_smiles)
            if not valid_actions:
                # 有効な行動がない
                if current.depth < self.config.min_depth and not evaluated:
                    # 行き詰まり（min_depth未満で展開不能）
                    reward = 0.0
                break

            # === Expansion ===
            # 行動選択
            action = self._select_action_for_expansion(current, valid_actions)

            # 状態遷移
            next_states = self.env.step(current.state_smiles, action)

            if not next_states:
                break

            # 次状態選択（Value headによるsoftmaxサンプリング）
            selected_state = self._select_next_state(next_states)

            # 新ノード作成または既存ノード取得
            child = self._create_or_get_node(current, action, selected_state)

            path.append(child)
            current = child

        return reward, path

    def _select_action_for_expansion(
        self,
        node: MCTSNode,
        valid_actions: list[int]
    ) -> int:
        """
        展開時の行動選択。
        """
        # 未展開の行動を優先
        unexpanded = [a for a in valid_actions if not node.has_child(a)]

        if unexpanded:
            if self.config.mcts_mode == MCTSMode.UCT:
                # ランダム選択
                return self._rng.choice(unexpanded)
            else:
                # PUCTのpriorに基づいて選択
                with torch.no_grad():
                    policy_logits, _ = self.model(node.state_smiles)

                tau = self.scheduler.get_temperature(self._current_step)
                policy_probs = compute_policy_probs(policy_logits, unexpanded, tau)

                # priorに基づいてサンプリング
                probs = policy_probs[unexpanded]
                probs = probs / probs.sum()  # 再正規化
                return self._rng.choice(unexpanded, p=probs)

        # 全て展開済みの場合
        return self._select_action(node, valid_actions)

    def _select_next_state(self, next_states: list[str]) -> str:
        """
        Value headの出力に基づいて次の状態を選択する。
        """
        if len(next_states) == 1:
            return next_states[0]

        # 各状態のValueを計算
        with torch.no_grad():
            values = []
            for state in next_states:
                _, value = self.model(state)
                values.append(value)

        values = np.array(values)
        tau = self.scheduler.get_temperature(self._current_step)

        _, selected_state = select_next_state_by_value(
            next_states, values, tau, self._rng
        )

        return selected_state

    def _create_or_get_node(
        self,
        parent: MCTSNode,
        action: int,
        state_smiles: str
    ) -> MCTSNode:
        """
        新しいノードを作成するか、既存のノードを取得する。
        """
        depth = parent.depth + 1

        # 既存ノードチェック（トランスポジション）
        existing = self.tree.get_node_by_state(state_smiles, depth)
        if existing is not None:
            if not parent.has_child(action):
                parent.add_child(action, existing)
            return existing

        # 新規ノード作成
        mol_props = self.env.calc_mol_props(state_smiles)
        num_sub = self.env.calc_num_sub(state_smiles)

        # Alert判定
        is_terminal = False
        if self.env.is_alert_elem(state_smiles):
            is_terminal = True
        if depth >= self.config.max_depth:
            is_terminal = True

        # leaf_calc初期化
        leaf_calc = LeafCalcStatus.NOT_READY
        if depth >= self.config.min_depth and self.env.check_lower_constraints(mol_props):
            leaf_calc = LeafCalcStatus.READY

        new_node = self.tree.add_node(
            parent=parent,
            fragment_index=action,
            child_state_smiles=state_smiles,
            depth=depth,
            leaf_calc=leaf_calc,
            is_terminal=is_terminal,
            num_sub=num_sub
        )
        new_node.set_mol_props(mol_props)

        return new_node

    def _backpropagate(self, path: list[MCTSNode], reward: float) -> None:
        """
        Backpropagation: 報酬を経路上のノードに逆伝播する。

        Args:
            path: 経路ノードリスト（root側が先頭ではない）
            reward: 報酬
        """
        for node in path:
            node.update_stats(reward)

    # === 学習 ===
    def _train_model(self) -> None:
        """統合モデルを学習する"""
        if self.optimizer is None:
            return

        # 学習データ抽出
        training_data = self.tree.extract_training_data(
            q_min=self.config.q_threshold_for_training,
            visit_count_min=1,
            sample_ratio=None
        )

        if len(training_data) < self.config.batch_size:
            return

        # ミニバッチサンプリング
        indices = self._rng.choice(
            len(training_data),
            size=min(self.config.batch_size, len(training_data)),
            replace=False
        )

        batch = [training_data[i] for i in indices]

        # バッチデータ準備
        state_smiles_list = [d['state_smiles'] for d in batch]
        policy_targets = torch.tensor(
            np.stack([d['policy_target'] for d in batch]),
            dtype=torch.float32
        )
        value_targets = torch.tensor(
            [d['value_target'] for d in batch],
            dtype=torch.float32
        )

        # Forward
        policy_logits, values = self.model.batch_forward(state_smiles_list)

        # Loss計算（AlphaGoZero方式）
        # Policy loss: Cross entropy
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(policy_targets * policy_log_probs, dim=-1).mean()

        # Value loss: MSE
        value_loss = torch.mean((values - value_targets) ** 2)

        # Total loss
        loss = policy_loss + value_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.config.enable_logging:
            logger.debug(
                f"Training: policy_loss={policy_loss.item():.4f}, "
                f"value_loss={value_loss.item():.4f}"
            )

    # === ログ ===
    def _log_progress(self, sim: int) -> None:
        """進捗をログ出力する"""
        stats = self.tree.get_statistics()
        tau = self.scheduler.get_temperature(self._current_step)

        logger.info(
            f"Sim {sim}/{self.config.max_simulation}: "
            f"nodes={stats['total_nodes']}, "
            f"max_depth={stats['max_depth']}, "
            f"avg_q={stats['avg_q_value']:.4f}, "
            f"max_q={stats['max_q_value']:.4f}, "
            f"tau={tau:.4f}"
        )

    # === モデル保存・読み込み ===
    def save_model(self, path: str) -> None:
        """統合モデルを保存する"""
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """統合モデルを読み込む"""
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path))
```

---

## 8. シリアライズ (`io/serialization.py`)

```python
import pickle
import gzip
from pathlib import Path
from typing import Union
import json

from ..core.mcts_tree import MCTSTree
from ..config import AgentConfig

def save_tree(tree: MCTSTree, path: Union[str, Path], compress: bool = True) -> None:
    """
    MCTSTreeを保存する。

    Args:
        tree: 保存するツリー
        path: 保存先パス
        compress: gzip圧縮するか
    """
    path = Path(path)
    data = tree.to_dict()

    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_tree(path: Union[str, Path], compressed: bool = True) -> MCTSTree:
    """
    MCTSTreeを読み込む。

    Args:
        path: 読み込み元パス
        compressed: gzip圧縮されているか

    Returns:
        復元されたMCTSTree
    """
    path = Path(path)

    if compressed:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    return MCTSTree.from_dict(data)

def save_config(config: AgentConfig, path: Union[str, Path]) -> None:
    """
    AgentConfigをJSON形式で保存する。

    Args:
        config: 保存する設定
        path: 保存先パス
    """
    path = Path(path)

    # Enumを文字列に変換
    data = {
        'max_depth': config.max_depth,
        'min_depth': config.min_depth,
        'max_simulation': config.max_simulation,
        'mcts_mode': config.mcts_mode.name,
        'c_uct': config.c_uct,
        'c_puct': config.c_puct,
        'train_interval': config.train_interval,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'q_threshold_for_training': config.q_threshold_for_training,
        'tau_initial': config.tau_initial,
        'tau_final': config.tau_final,
        'tau_scheduler_type': config.tau_scheduler_type.name,
        'tau_decay_steps': config.tau_decay_steps,
        'device': config.device,
        'seed': config.seed,
        'enable_logging': config.enable_logging,
        'log_interval': config.log_interval,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_config(path: Union[str, Path]) -> AgentConfig:
    """
    AgentConfigをJSON形式から読み込む。

    Args:
        path: 読み込み元パス

    Returns:
        復元されたAgentConfig
    """
    from ..core.types import MCTSMode, SchedulerType

    path = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return AgentConfig(
        max_depth=data['max_depth'],
        min_depth=data['min_depth'],
        max_simulation=data['max_simulation'],
        mcts_mode=MCTSMode[data['mcts_mode']],
        c_uct=data['c_uct'],
        c_puct=data['c_puct'],
        train_interval=data['train_interval'],
        batch_size=data['batch_size'],
        learning_rate=data['learning_rate'],
        weight_decay=data['weight_decay'],
        q_threshold_for_training=data.get('q_threshold_for_training'),
        tau_initial=data['tau_initial'],
        tau_final=data['tau_final'],
        tau_scheduler_type=SchedulerType[data['tau_scheduler_type']],
        tau_decay_steps=data.get('tau_decay_steps'),
        device=data['device'],
        seed=data.get('seed'),
        enable_logging=data['enable_logging'],
        log_interval=data['log_interval'],
    )
```

---

## 9. 推論ユーティリティ (`inference.py`)

```python
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import pandas as pd

from .core.mcts_tree import MCTSTree
from .core.mcts_node import MCTSNode
from .io.serialization import load_tree

@dataclass
class InferenceResult:
    """推論結果"""
    state_smiles: str
    leaf_smiles: Optional[str]
    q_value: float
    visit_count: int
    total_reward: float
    num_sub: int
    depth: int

def extract_high_reward_nodes(
    tree: MCTSTree,
    q_threshold: float,
    num_sub_min: Optional[int] = None,
    total_reward_min: Optional[float] = None,
    depth_range: Optional[tuple[int, int]] = None,
    top_n: Optional[int] = None
) -> list[InferenceResult]:
    """
    高報酬ノードを抽出する。

    Args:
        tree: MCTSTree
        q_threshold: Q値の閾値
        num_sub_min: 最小サブスペースサイズ
        total_reward_min: 最小累計報酬
        depth_range: 深さの範囲 (min, max)
        top_n: 上位N件のみ取得

    Returns:
        推論結果のリスト（Q値降順）
    """
    nodes = tree.filter_nodes(
        q_min=q_threshold,
        num_sub_min=num_sub_min,
        total_reward_min=total_reward_min,
        depth_range=depth_range
    )

    # Q値でソート
    nodes.sort(key=lambda x: x.q_value, reverse=True)

    # 上位N件に制限
    if top_n is not None:
        nodes = nodes[:top_n]

    # 結果に変換
    results = []
    for node in nodes:
        results.append(InferenceResult(
            state_smiles=node.state_smiles,
            leaf_smiles=node.leaf_smiles,
            q_value=node.q_value,
            visit_count=node.visit_count,
            total_reward=node.total_reward,
            num_sub=node.num_sub,
            depth=node.depth
        ))

    return results

def results_to_dataframe(results: list[InferenceResult]) -> pd.DataFrame:
    """
    推論結果をDataFrameに変換する。

    Args:
        results: 推論結果のリスト

    Returns:
        DataFrame
    """
    data = []
    for r in results:
        data.append({
            'state_smiles': r.state_smiles,
            'leaf_smiles': r.leaf_smiles,
            'q_value': r.q_value,
            'visit_count': r.visit_count,
            'total_reward': r.total_reward,
            'num_sub': r.num_sub,
            'depth': r.depth
        })

    return pd.DataFrame(data)

def load_and_extract(
    tree_path: Union[str, Path],
    q_threshold: float,
    **kwargs
) -> list[InferenceResult]:
    """
    保存されたTreeを読み込んで高報酬ノードを抽出する。

    Args:
        tree_path: Treeファイルのパス
        q_threshold: Q値の閾値
        **kwargs: extract_high_reward_nodesの追加引数

    Returns:
        推論結果のリスト
    """
    tree = load_tree(tree_path)
    return extract_high_reward_nodes(tree, q_threshold, **kwargs)
```

---

## 10. ロギングユーティリティ (`logging_utils.py`)

```python
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = 'ecmporl',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    ロガーをセットアップする。

    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルパス（Noneでコンソールのみ）
        format_string: フォーマット文字列

    Returns:
        設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

class ProgressTracker:
    """シミュレーション進捗トラッカー"""

    def __init__(
        self,
        total_simulations: int,
        log_interval: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        self.total = total_simulations
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger('ecmporl')
        self.current = 0

    def update(self, tree_stats: dict, tau: float) -> None:
        """進捗を更新する"""
        self.current += 1

        if self.current % self.log_interval == 0:
            progress = self.current / self.total * 100
            self.logger.info(
                f"Progress: {self.current}/{self.total} ({progress:.1f}%) | "
                f"Nodes: {tree_stats['total_nodes']} | "
                f"Max depth: {tree_stats['max_depth']} | "
                f"Avg Q: {tree_stats['avg_q_value']:.4f} | "
                f"Max Q: {tree_stats['max_q_value']:.4f} | "
                f"Tau: {tau:.4f}"
            )
```

---

## 11. アルゴリズム詳細（疑似コード）

### 11.1 MCTSシミュレーション全体フロー

```
function run_simulation(tree, env, model, config):
    # === Selection ===
    node = tree.root
    while node has children and not node.is_terminal:
        valid_actions = env.get_valid_actions(node.state_smiles)
        unexpanded = [a for a in valid_actions if not node.has_child(a)]

        if unexpanded is not empty:
            break  # 展開フェーズへ

        action = select_action(node, valid_actions, model, config)
        node = node.get_child(action)

    # === Rollout（深さ優先展開） ===
    path = [node]
    reward = 0.0
    evaluated = False

    while True:
        # Leaf評価チェック
        if node.leaf_calc == READY and not evaluated:
            leaf_smiles = env.generate_leaf_smiles(node.state_smiles)
            node.leaf_smiles = leaf_smiles
            reward = env.calc_reward(leaf_smiles)
            node.leaf_calc = EVALUATED
            evaluated = True

        # 展開継続判定
        if node.is_terminal or node.depth >= config.max_depth:
            break

        valid_actions = env.get_valid_actions(node.state_smiles)
        if valid_actions is empty:
            if node.depth < config.min_depth and not evaluated:
                reward = 0.0  # 行き詰まり
            break

        # Expansion
        action = select_action_for_expansion(node, valid_actions, model, config)
        next_states = env.step(node.state_smiles, action)

        if next_states is empty:
            break

        selected_state = select_next_state(next_states, model, config.tau)
        child = create_or_get_node(tree, node, action, selected_state, config)

        path.append(child)
        node = child

    # === Backpropagation ===
    for node in path:
        node.visit_count += 1
        node.total_reward += reward
```

### 11.2 PUCT行動選択

```
function select_action_puct(node, valid_actions, policy_probs, c_puct):
    parent_visits = node.visit_count
    best_action = None
    best_score = -infinity

    for action in valid_actions:
        child = node.get_child(action)

        if child is None:
            Q = 0.0
            N_sa = 0
        else:
            Q = child.q_value
            N_sa = child.visit_count

        P = policy_probs[action]

        # PUCT score
        score = Q + c_puct * P * sqrt(parent_visits + 1) / (1 + N_sa)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
```

### 11.3 次状態選択（Value head使用）

```
function select_next_state(next_states, model, tau):
    if len(next_states) == 1:
        return next_states[0]

    values = []
    for state in next_states:
        _, value = model(state)
        values.append(value)

    # 温度付きsoftmax
    scaled_values = values / tau
    scaled_values -= max(scaled_values)  # 数値安定性
    exp_values = exp(scaled_values)
    probs = exp_values / sum(exp_values)

    # サンプリング
    selected_idx = sample_from_distribution(probs)
    return next_states[selected_idx]
```

---

## 12. エラーハンドリング

### 12.1 例外クラス定義

```python
class ECMPORLError(Exception):
    """ECMPORLの基底例外クラス"""
    pass

class InvalidSMILESError(ECMPORLError):
    """無効なSMILESエラー"""
    pass

class ConstraintViolationError(ECMPORLError):
    """制約違反エラー"""
    pass

class ModelError(ECMPORLError):
    """モデル関連エラー"""
    pass

class SerializationError(ECMPORLError):
    """シリアライズ/デシリアライズエラー"""
    pass
```

### 12.2 エラーハンドリング方針

1. **ユーザ提供関数のエラー**:
   - try-exceptでキャッチし、ログに記録
   - Alert判定やReward計算でエラー時はデフォルト値（Alert=True, Reward=0）を使用

2. **制約違反**:
   - get_valid_actionsで事前にフィルタリング
   - 万が一違反が発生した場合はConstraintViolationErrorを送出

3. **モデル推論エラー**:
   - PyTorchのエラーをキャッチしてModelErrorに変換
   - バッチ推論失敗時は個別推論にフォールバック

---

## 13. 使用例

```python
import pandas as pd
import torch
import torch.nn as nn

from ecmporl_02 import Environment, Agent, AgentConfig, MCTSMode
from ecmporl_02.io import save_tree, load_tree
from ecmporl_02.inference import extract_high_reward_nodes, results_to_dataframe

# === ユーザ定義コンポーネント ===
class MyPolicyValueModel(nn.Module):
    """統合モデルの実装例"""
    def __init__(self, num_fragments: int, hidden_dim: int = 256):
        super().__init__()
        self.num_fragments = num_fragments

        # 共通トランク（ここではダミー実装）
        self.trunk = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_fragments)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state_smiles: str) -> tuple[torch.Tensor, float]:
        # 実際にはSMILESをグラフエンコーディングする
        x = torch.randn(100)  # ダミー入力

        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).item()

        return policy_logits, value

    def batch_forward(self, state_smiles_list: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(state_smiles_list)
        x = torch.randn(batch_size, 100)  # ダミー入力

        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)

        return policy_logits, values

# ユーザ定義関数（ダミー実装）
def combine_smiles(core: str, frag: str) -> str:
    return core.replace('*', '') + frag.replace('*', '')

def hydrogen_replace(smiles: str) -> list[str]:
    return [f"*{smiles}"]

def remove_deuterium(smiles: str) -> str:
    return smiles

def reward_fn(smiles: str) -> float:
    return 0.5

def alert_elem(smiles: str) -> int:
    return 1

def alert_mol(smiles: str) -> int:
    return 1

def measure_mol_props(smiles: str) -> dict:
    return {'HAC': 10, 'cnt_hetero': 2, 'cnt_chiral': 0, 'MW': 200.0}

def count_subspace(smiles: str) -> int:
    return 1000

# === セットアップ ===
# Fragment テーブル
fragment_df = pd.DataFrame({
    'smiles': ['*C', '*CC', '*CCC', '*N', '*O'],
    'HAC': [1, 2, 3, 1, 1],
    'cnt_hetero': [0, 0, 0, 1, 1],
    'cnt_chiral': [0, 0, 0, 0, 0],
    'MW': [15.0, 29.0, 43.0, 16.0, 17.0]
})

# 制約
constraints = {
    'HAC_min': 5,
    'HAC_max': 30,
    'MW_min': 100.0,
    'MW_max': 500.0
}

# Environment
env = Environment(
    fragment_df=fragment_df,
    combine_smiles_fn=combine_smiles,
    hydrogen_replace_fn=hydrogen_replace,
    remove_deuterium_fn=remove_deuterium,
    reward_fns=[reward_fn],
    alert_elem_fn=alert_elem,
    alert_mol_fn=alert_mol,
    measure_mol_props_fn=measure_mol_props,
    count_subspace_fn=count_subspace,
    constraints=constraints
)

# モデル
model = MyPolicyValueModel(num_fragments=len(fragment_df))

# 設定
config = AgentConfig(
    max_depth=5,
    min_depth=2,
    max_simulation=1000,
    mcts_mode=MCTSMode.PUCT,
    c_puct=1.0,
    train_interval=100,
    tau_initial=1.0,
    tau_final=0.1
)

# Agent
agent = Agent(env=env, model=model, config=config)

# === 探索実行 ===
tree = agent.run_mcts(core_smiles='*c1ccccc1')

# === 結果保存 ===
save_tree(tree, 'mcts_tree.pkl.gz')
agent.save_model('model.pth')

# === 推論 ===
results = extract_high_reward_nodes(tree, q_threshold=0.3)
df = results_to_dataframe(results)
print(df)
```

---

## 14. 補足：要件定義書との対応表

| 要件定義書セクション | 設計仕様書セクション | 備考 |
|---------------------|---------------------|------|
| 6.1 MCTS全体フロー | 11.1 アルゴリズム詳細 | 疑似コードで詳細化 |
| 6.2 MCTSNode | 4. MCTSNode クラス設計 | dataclass使用 |
| 6.3 MCTSTree | 5. MCTSTree クラス設計 | トランスポジション対応 |
| 6.4 Environment | 6. Environment クラス設計 | Protocol定義含む |
| 6.5 Agent | 7. Agent クラス設計 | 統合モデル対応 |
| 6.6 UCT/PUCT | 7.3 Selection ロジック | 関数として分離 |
| 6.7 Rollout | 7.4 Agent._rollout() | 深さ優先展開 |
| 6.8 トランスポジション | 5.1 MCTSTree.add_node() | node_idで管理 |
| 6.9 I/O | 8. シリアライズ | pickle+gzip使用 |
| 6.10 推論モード | 9. 推論ユーティリティ | 高報酬ノード抽出 |

---

以上が ECMPORL_02 パッケージの設計仕様書 v2。
