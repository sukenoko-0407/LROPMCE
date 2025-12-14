# ECMPORL 設計仕様書 v1

## 1. パッケージ構成

### 1.1 ディレクトリ構成（案）

```text
ecmporl/
  __init__.py

  config.py                # 各種設定用 dataclass 群

  core/
    node.py                # MCTSNode 定義
    tree.py                # MCTSTree 定義
    selection.py           # UCT / PUCT 選択ロジック
    backup.py              # Backpropagation ロジック（必要なら分離）

  env/
    environment.py         # Environment クラス

  agent/
    agent.py               # Agent クラス（MCTS 制御）
    memory.py              # リプレイバッファ & Inception メモリ
    schedulers.py          # τ スケジューラ実装

  io/
    serialization.py       # Tree とモデルのシリアライズ/デシリアライズ

  inference/
    query.py               # 高 Q ノード抽出などのユーティリティ
```

`__init__.py` からは主に以下を公開:

- `Environment`
- `Agent`
- `MCTSTree`, `MCTSNode`
- `MCTSConfig`, `TrainingConfig`, `ConstraintConfig`, `InferenceConfig`

---

## 2. 設定クラス（config.py）

### 2.1 MCTSConfig

```python
@dataclass
class MCTSConfig:
    max_depth: int
    min_depth: int
    max_simulation: int

    mcts_mode: str = "puct"   # "uct" or "puct"
    c_uct: float = 1.4
    c_puct: float = 1.0
```

### 2.2 ConstraintConfig

```python
@dataclass
class ConstraintConfig:
    hac_min: int | None = None
    hac_max: int | None = None
    cnt_hetero_min: int | None = None
    cnt_hetero_max: int | None = None
    cnt_chiral_min: int | None = None
    cnt_chiral_max: int | None = None
    mw_min: float | None = None
    mw_max: float | None = None
```

### 2.3 TrainingConfig

```python
@dataclass
class TrainingConfig:
    train_interval: int       # 何シミュレーションごとに学習
    batch_size: int
    memory_capacity: int
    inception_capacity: int
    inception_threshold: float

    # inverse-temp for softmax(P) 等が必要なら
    puct_policy_temperature: float = 1.0

    # 損失関連（シンプルに MSE を想定）
    v_loss_weight: float = 1.0
    qsa_loss_weight: float = 1.0
```

### 2.4 TauSchedulerConfig（τ スケジューラ）

```python
@dataclass
class TauSchedulerConfig:
    tau_initial: float
    tau_final: float
    num_steps: int              # 減衰を完了させるステップ数
    scheduler_type: str = "linear"  # "linear" or "exponential"
```

### 2.5 InferenceConfig

```python
@dataclass
class InferenceConfig:
    q_value_threshold: float
    num_sub_min: int | None = None
    total_reward_min: float | None = None
    depth_min: int | None = None
    depth_max: int | None = None
```

---

## 3. コアクラス設計

## 3.1 MCTSNode（core/node.py）

### 3.1.1 属性

```python
@dataclass
class MCTSNode:
    node_id: int           # Tree 内で一意の整数 ID
    state_smiles: str      # ダミーアトム付き canonical SMILES
    depth: int

    visit_count: int = 0   # N
    total_reward: float = 0.0  # W
    q_value: float = 0.0       # W / N（N>0 のとき）

    is_terminal: bool = False
    num_sub: int | None = None

    # children: action_id -> child_node_id
    children: dict[int, int] = field(default_factory=dict)

    # 便利用（バックアップでは path を持つので parent は必須ではない）
    parent_id: int | None = None

    # 追加メタ情報
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### 状態一意性

- Tree 内では `(state_smiles, depth)` をキーとし、一意な `MCTSNode` を保持。
- `MCTSTree` 側で `nodes_by_key: dict[tuple[str, int], int]` を管理する。

### 3.1.2 メソッド

```python
def update_stats(self, reward: float) -> None:
    self.visit_count += 1
    self.total_reward += reward
    self.q_value = self.total_reward / self.visit_count
```

---

## 3.2 MCTSTree（core/tree.py）

### 3.2.1 属性

```python
class MCTSTree:
    def __init__(self, root_state_smiles: str):
        self.nodes: dict[int, MCTSNode] = {}
        self.nodes_by_key: dict[tuple[str, int], int] = {}

        self.root_id: int = self._create_root(root_state_smiles)
        self._next_node_id: int = self.root_id + 1
```

### 3.2.2 主要メソッド

```python
def _create_root(self, state_smiles: str) -> int:
    node = MCTSNode(
        node_id=0,
        state_smiles=state_smiles,
        depth=0,
    )
    self.nodes[node.node_id] = node
    self.nodes_by_key[(state_smiles, 0)] = node.node_id
    return node.node_id

def get_node(self, node_id: int) -> MCTSNode:
    return self.nodes[node_id]

def get_or_create_child(
    self,
    parent_id: int,
    child_state_smiles: str,
    child_depth: int,
) -> int:
    """
    (state_smiles, depth) で既存ノードを探し、
    存在すればそれを返し、なければ新規作成して返す。
    parent.children にも登録する。
    """
```

#### ノード抽出・フィルタリング

```python
def filter_nodes(
    self,
    q_min: float | None = None,
    total_reward_min: float | None = None,
    num_sub_min: int | None = None,
    depth_min: int | None = None,
    depth_max: int | None = None,
) -> list[MCTSNode]:
    """
    条件に合致するノードをリストで返す。
    """
```

#### シリアライズ対応（詳細は io/serialization.py）

- `to_serializable()`:
  - ノード情報（ID, state_smiles, depth, visit_count, total_reward, q_value, is_terminal, num_sub, children）を辞書に変換。
- `from_serializable(data: dict) -> MCTSTree`:
  - シリアライズデータから Tree を再構築。

---

## 4. Environment 設計（env/environment.py）

### 4.1 コンストラクタ

```python
class Environment:
    def __init__(
        self,
        fragment_df: pd.DataFrame,
        combine_fn: Callable[[str, str], str],
        hydrogen_replace_fn: Callable[[str], list[str]],
        reward_fns: list[Callable[[str | list[str]], float | list[float]]],
        alert_fn: Callable[[str], bool],
        calc_props_fn: Callable[[str], dict[str, float | int]],
        count_subspace_fn: Callable[[str], int],
        fragment_featurizer: Callable[[str], Any],
        constraints: ConstraintConfig,
    ):
        self.fragment_df = fragment_df.reset_index(drop=True)
        self.combine_fn = combine_fn
        self.hydrogen_replace_fn = hydrogen_replace_fn
        self.reward_fns = reward_fns
        self.alert_fn = alert_fn
        self.calc_props_fn = calc_props_fn
        self.count_subspace_fn = count_subspace_fn
        self.constraints = constraints

        # Fragment グラフ表現を前計算
        self.fragment_features: list[Any] = [
            fragment_featurizer(sm) for sm in self.fragment_df["smiles"]
        ]
```

### 4.2 プロパティ制約チェック

```python
def check_constraints(self, smiles: str) -> bool:
    """
    分子全体の HAC / cnt_hetero / cnt_chiral / MW を計算し、
    すべての下限・上限を満たしていれば True を返す。
    満たさない場合は False。
    """
```

### 4.3 Alert 判定

```python
def is_alert(self, smiles: str) -> bool:
    return self.alert_fn(smiles)
```

### 4.4 num_sub 計算

```python
def calc_num_sub(self, state_smiles: str) -> int:
    return self.count_subspace_fn(state_smiles)
```

### 4.5 報酬評価

```python
def evaluate_leaf(self, smiles: str) -> float:
    """
    1 つの SMILES に対して全報酬関数を適用し、相乗平均を返す。
    """
```

### 4.6 有効 Fragment の列挙

```python
def get_valid_actions(self, state_smiles: str) -> list[int]:
    """
    現在状態に対して、分子全体プロパティ制約を満たす
    Fragment の index リストを返す。

    - combine_fn(state, fragment) で得られる中間 SMILES について
      check_constraints を行い、その結果によってフィルタリングする。
    """
```

### 4.7 状態遷移処理

```python
@dataclass
class TransitionResult:
    next_state_smiles: str
    leaf_smiles: str | None
    is_terminal: bool
```

```python
def apply_action(
    self,
    state_smiles: str,
    fragment_idx: int,
) -> list[TransitionResult]:
    """
    1 つの行動（Fragment）に対して、
    combine_smiles -> hydrogen_replace で得られる
    複数の次状態候補を生成する。

    戻り値は複数の TransitionResult。
    """
```

---

## 5. メモリ・スケジューラ設計

### 5.1 Transition 型（agent/memory.py）

```python
@dataclass
class Transition:
    state_smiles: str
    action_idx: int
    reward: float
    next_state_smiles: str
    done: bool       # terminal or 行き詰まりなど
```

### 5.2 ReplayBuffer

```python
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        ...
```

### 5.3 InceptionBuffer

```python
class InceptionBuffer(ReplayBuffer):
    pass
```

### 5.4 τ スケジューラ（agent/schedulers.py）

```python
class TauScheduler:
    def __init__(self, config: TauSchedulerConfig):
        self.cfg = config
        self.step = 0

    def get_tau(self) -> float:
        if self.cfg.scheduler_type == "linear":
            ratio = min(self.step / self.cfg.num_steps, 1.0)
            return self.cfg.tau_initial + ratio * (self.cfg.tau_final - self.cfg.tau_initial)
        elif self.cfg.scheduler_type == "exponential":
            k = 5.0 / self.cfg.num_steps
            return self.cfg.tau_final +                 (self.cfg.tau_initial - self.cfg.tau_final) * math.exp(-k * self.step)
        else:
            return self.cfg.tau_initial

    def step_forward(self) -> None:
        self.step += 1
```

---

## 6. Agent 設計（agent/agent.py）

### 6.1 コンストラクタ

```python
class Agent:
    def __init__(
        self,
        env: Environment,
        v_model: torch.nn.Module,
        qsa_model: torch.nn.Module,
        mcts_config: MCTSConfig,
        constraint_config: ConstraintConfig,
        training_config: TrainingConfig,
        tau_config: TauSchedulerConfig,
        device: torch.device | str = "cpu",
    ):
        self.env = env
        self.v_model = v_model.to(device)
        self.qsa_model = qsa_model.to(device)

        self.mcts_config = mcts_config
        self.training_config = training_config
        self.constraints = constraint_config
        self.device = device

        self.tree: MCTSTree | None = None

        self.replay_buffer = ReplayBuffer(training_config.memory_capacity)
        self.inception_buffer = InceptionBuffer(training_config.inception_capacity)

        self.tau_scheduler = TauScheduler(tau_config)

        # Optimizer / loss
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=1e-4)
        self.qsa_optimizer = torch.optim.Adam(self.qsa_model.parameters(), lr=1e-4)
        self.v_loss_fn = torch.nn.MSELoss()
        self.qsa_loss_fn = torch.nn.MSELoss()

        # 内部カウンタ
        self.simulation_count: int = 0
```

### 6.2 公開メソッド

#### 6.2.1 run_mcts

```python
def run_mcts(self, core_smiles: str) -> MCTSTree:
    self.tree = MCTSTree(root_state_smiles=core_smiles)

    for sim in range(self.mcts_config.max_simulation):
        self._run_single_simulation()
        self.simulation_count += 1

        if (
            self.training_config.train_interval > 0
            and self.simulation_count % self.training_config.train_interval == 0
        ):
            self._train_models()

        self.tau_scheduler.step_forward()

    return self.tree
```

#### 6.2.2 モデル保存・読み込み

```python
def save_models(self, v_path: str, qsa_path: str) -> None:
    torch.save(self.v_model.state_dict(), v_path)
    torch.save(self.qsa_model.state_dict(), qsa_path)

def load_models(self, v_path: str, qsa_path: str) -> None:
    self.v_model.load_state_dict(torch.load(v_path, map_location=self.device))
    self.qsa_model.load_state_dict(torch.load(qsa_path, map_location=self.device))
```

#### 6.2.3 推論ユーティリティ

```python
def get_high_q_nodes(
    self,
    inference_cfg: InferenceConfig,
) -> list[MCTSNode]:
    return self.tree.filter_nodes(
        q_min=inference_cfg.q_value_threshold,
        total_reward_min=inference_cfg.total_reward_min,
        num_sub_min=inference_cfg.num_sub_min,
        depth_min=inference_cfg.depth_min,
        depth_max=inference_cfg.depth_max,
    )
```

### 6.3 内部メソッド：1 シミュレーション

```python
def _run_single_simulation(self) -> None:
    assert self.tree is not None

    path_node_ids: list[int] = []
    path_actions: list[int] = []
    leaf_smiles: str | None = None
    reward: float = 0.0
    done: bool = False

    node_id = self.tree.root_id
    while True:
        node = self.tree.get_node(node_id)
        path_node_ids.append(node_id)

        if node.is_terminal or node.depth >= self.mcts_config.max_depth:
            done = True
            reward = 0.0
            break

        valid_actions = self.env.get_valid_actions(node.state_smiles)

        if len(valid_actions) == 0:
            if node.depth < self.mcts_config.min_depth:
                done = True
                reward = 0.0
                break
            else:
                leaf_smiles = self._state_to_leaf_smiles(node.state_smiles)
                reward = float(self.env.evaluate_leaf(leaf_smiles))
                done = True
                break

        if self._has_unvisited_child(node, valid_actions):
            action_idx = self._select_unvisited_action(node, valid_actions)
        else:
            action_idx = self._select_action(node, valid_actions)

        path_actions.append(action_idx)

        candidates = self.env.apply_action(node.state_smiles, action_idx)
        if not candidates:
            done = True
            reward = 0.0
            break

        tau = self.tau_scheduler.get_tau()
        next_state_smiles = self._select_next_state_by_v(candidates, tau)

        next_depth = node.depth + 1
        child_id = self.tree.get_or_create_child(
            parent_id=node_id,
            child_state_smiles=next_state_smiles,
            child_depth=next_depth,
        )
        child_node = self.tree.get_node(child_id)

        if child_node.num_sub is None:
            child_node.num_sub = self.env.calc_num_sub(child_node.state_smiles)
        if not child_node.is_terminal:
            if self.env.is_alert(child_node.state_smiles):
                child_node.is_terminal = True

        node_id = child_id

        if child_node.depth >= self.mcts_config.min_depth:
            leaf_smiles = self._state_to_leaf_smiles(child_node.state_smiles)
            reward = float(self.env.evaluate_leaf(leaf_smiles))
            done = True
            break

    self._backup(path_node_ids, reward)

    if len(path_node_ids) >= 2 and len(path_actions) >= 1:
        last_state_id = path_node_ids[-2]
        last_next_state_id = path_node_ids[-1]
        last_node = self.tree.get_node(last_state_id)
        next_node = self.tree.get_node(last_next_state_id)
        transition = Transition(
            state_smiles=last_node.state_smiles,
            action_idx=path_actions[-1],
            reward=reward,
            next_state_smiles=next_node.state_smiles,
            done=done,
        )
        self._store_transition(transition)
```

### 6.4 Action Selection（UCT / PUCT）

```python
def _select_action(self, node: MCTSNode, valid_actions: list[int]) -> int:
    if self.mcts_config.mcts_mode == "uct":
        return self._select_action_uct(node, valid_actions)
    elif self.mcts_config.mcts_mode == "puct":
        return self._select_action_puct(node, valid_actions)
    else:
        raise ValueError(...)
```

#### 6.4.1 未訪問行動優先

```python
def _has_unvisited_child(self, node: MCTSNode, valid_actions: list[int]) -> bool:
    for a in valid_actions:
        if a not in node.children:
            return True
    return False

def _select_unvisited_action(self, node: MCTSNode, valid_actions: list[int]) -> int:
    unvisited = [a for a in valid_actions if a not in node.children]
    return random.choice(unvisited)
```

#### 6.4.2 UCT

```python
def _select_action_uct(self, node: MCTSNode, valid_actions: list[int]) -> int:
    parent_n = max(node.visit_count, 1)
    best_score = -float("inf")
    best_action = valid_actions[0]

    for a in valid_actions:
        if a not in node.children:
            continue
        child_id = node.children[a]
        child = self.tree.get_node(child_id)

        q = child.q_value
        n_sa = max(child.visit_count, 1)
        u = self.mcts_config.c_uct * math.sqrt(math.log(parent_n + 1) / n_sa)
        score = q + u

        if score > best_score:
            best_score = score
            best_action = a

    return best_action
```

#### 6.4.3 PUCT

```python
def _select_action_puct(self, node: MCTSNode, valid_actions: list[int]) -> int:
    state_smiles = node.state_smiles
    frag_smiles_list = [self.env.fragment_df.loc[a, "smiles"] for a in valid_actions]

    with torch.no_grad():
        qsa_values = self._forward_qsa(state_smiles, frag_smiles_list)

    values = np.array(qsa_values, dtype=float)
    temp = max(self.training_config.puct_policy_temperature, 1e-6)
    logits = values / temp
    exp_logits = np.exp(logits - logits.max())
    p = exp_logits / exp_logits.sum()

    parent_n = max(node.visit_count, 1)

    best_score = -float("inf")
    best_action = valid_actions[0]

    for idx, a in enumerate(valid_actions):
        prior = p[idx]
        if a in node.children:
            child = self.tree.get_node(node.children[a])
            q = child.q_value
            n_sa = max(child.visit_count, 1)
        else:
            q = 0.0
            n_sa = 0

        u = self.mcts_config.c_puct * prior * math.sqrt(parent_n + 1) / (1 + n_sa)
        score = q + u

        if score > best_score:
            best_score = score
            best_action = a

    return best_action
```

### 6.5 V(s) による次状態選択

```python
def _select_next_state_by_v(
    self,
    candidates: list[TransitionResult],
    tau: float,
) -> str:
    state_list = [c.next_state_smiles for c in candidates]

    with torch.no_grad():
        v_values = self._forward_v(state_list)

    v_values = np.array(v_values, dtype=float)
    logits = v_values / max(tau, 1e-6)
    exp_logits = np.exp(logits - logits.max())
    p = exp_logits / exp_logits.sum()

    idx = np.random.choice(len(candidates), p=p)
    return candidates[idx].next_state_smiles
```

---

## 7. Backup と学習

### 7.1 Backup

```python
def _backup(self, path_node_ids: list[int], reward: float) -> None:
    for node_id in path_node_ids:
        node = self.tree.get_node(node_id)
        node.update_stats(reward)
```

### 7.2 メモリ格納

```python
def _store_transition(self, transition: Transition) -> None:
    self.replay_buffer.add(transition)
    if transition.reward >= self.training_config.inception_threshold:
        self.inception_buffer.add(transition)
```

### 7.3 モデル学習

```python
def _train_models(self) -> None:
    if len(self.replay_buffer.buffer) == 0:
        return

    batch_all = self.replay_buffer.sample(self.training_config.batch_size)
    batch_inception_size = min(
        self.training_config.batch_size // 2,
        len(self.inception_buffer.buffer)
    )
    if batch_inception_size > 0:
        batch_inception = self.inception_buffer.sample(batch_inception_size)
        batch = batch_all + batch_inception
    else:
        batch = batch_all

    states = [t.state_smiles for t in batch]
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)

    v_pred = self._forward_v(states, as_tensor=True)
    v_loss = self.v_loss_fn(v_pred, rewards)

    frag_smiles = [self.env.fragment_df.loc[t.action_idx, "smiles"] for t in batch]
    qsa_pred = self._forward_qsa_batch(states, frag_smiles)
    qsa_loss = self.qsa_loss_fn(qsa_pred, rewards)

    loss = (
        self.training_config.v_loss_weight * v_loss
        + self.training_config.qsa_loss_weight * qsa_loss
    )

    self.v_optimizer.zero_grad()
    self.qsa_optimizer.zero_grad()
    loss.backward()
    self.v_optimizer.step()
    self.qsa_optimizer.step()
```

---

## 8. シリアライズ設計（io/serialization.py）

### 8.1 Tree の保存形式

```python
def save_tree(tree: MCTSTree, path: str) -> None:
    data = {
        "root_id": tree.root_id,
        "nodes": [
            {
                "node_id": n.node_id,
                "state_smiles": n.state_smiles,
                "depth": n.depth,
                "visit_count": n.visit_count,
                "total_reward": n.total_reward,
                "q_value": n.q_value,
                "is_terminal": n.is_terminal,
                "num_sub": n.num_sub,
                "children": n.children,
            }
            for n in tree.nodes.values()
        ],
    }
    packed = msgpack.packb(data)
    compressed = zstandard.ZstdCompressor().compress(packed)
    with open(path, "wb") as f:
        f.write(compressed)
```

- `load_tree(path) -> MCTSTree` で逆変換。

### 8.2 モデルの保存

- PyTorch の `state_dict` をそのまま `torch.save` / `torch.load`。

---

## 9. Inference ユーティリティ（inference/query.py）

```python
def extract_high_q_nodes(
    tree: MCTSTree,
    config: InferenceConfig,
) -> list[MCTSNode]:
    return tree.filter_nodes(
        q_min=config.q_value_threshold,
        total_reward_min=config.total_reward_min,
        num_sub_min=config.num_sub_min,
        depth_min=config.depth_min,
        depth_max=config.depth_max,
    )
```

---

## 10. 想定される使用例（コードイメージ）

```python
from ecmporl import (
    Environment, Agent,
    MCTSConfig, ConstraintConfig, TrainingConfig, TauSchedulerConfig,
)

env = Environment(
    fragment_df=fragment_df,
    combine_fn=combine_smiles,
    hydrogen_replace_fn=hydrogen_replace,
    reward_fns=[reward1, reward2],
    alert_fn=is_alert,
    calc_props_fn=calc_props,
    count_subspace_fn=count_subspace,
    fragment_featurizer=fragment_featurizer,
    constraints=ConstraintConfig(...),
)

agent = Agent(
    env=env,
    v_model=v_model,
    qsa_model=qsa_model,
    mcts_config=MCTSConfig(...),
    constraint_config=ConstraintConfig(...),
    training_config=TrainingConfig(...),
    tau_config=TauSchedulerConfig(...),
    device="cuda",
)

tree = agent.run_mcts(core_smiles="*c1ccccc1")

save_tree(tree, "tree.zst")
agent.save_models("v.pt", "qsa.pt")

cfg = InferenceConfig(q_value_threshold=0.6, num_sub_min=10)
high_nodes = agent.get_high_q_nodes(cfg)
```

---

以上が ECMPORL の設計仕様書 v1。
