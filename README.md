---

## 演进路线

本项目会持续推进，大致分为以下几个阶段：

### Phase 1：单 Agent 基础（已完成 ✅）
- [x] 原生手写 ReAct Agent（`plain/raw_agent.py`）
- [x] CrewAI 单 Agent 实现（`crew/crew_agent.py`）
- [x] LangGraph 单 Agent 实现（`langx/graph_agent.py`）
- [x] 统一 LLM 封装层（`llm/qwen_llm.py`）

### Phase 2：Multi-Agent 协作（进行中 🚧）
- [ ] CrewAI Multi-Agent：多个角色分工协作，编排复杂工作流
- [ ] LangGraph Multi-Agent：多节点并行、子图调用、条件路由
- [ ] Agent 间通信机制：消息传递、状态共享、结果聚合

### Phase 3：工程化能力建设（规划中 📋）
- [ ] **上下文管理**：长对话窗口压缩、Token 预算控制、历史消息摘要
- [ ] **用户记忆**：长期记忆存储（向量数据库）、用户画像构建、个性化回复
- [ ] **权限与安全**：工具调用权限控制、敏感操作确认、输出内容审核
- [ ] **可观测性**：Agent 执行链路追踪、思考过程可视化、性能指标采集
- [ ] **配置化**：Prompt 模板外部化管理、Agent 行为热更新、A/B 测试支持
- [ ] **部署与扩展**：异步任务队列、流式输出、API 服务化封装

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 语言 | Python >= 3.12 |
| 包管理 | uv |
| LLM | 阿里云通义千问（DashScope） |
| Agent 框架 | CrewAI、LangGraph |
| 底层依赖 | LangChain Core、Requests |

---

## 学习建议

如果你想通过本项目深入理解 Agent，推荐按以下顺序阅读源码：

1. **`plain/raw_agent.py`** → 先看透"没有框架时 Agent 长什么样"
2. **`langx/graph_agent.py`** → 再看 LangGraph 如何把同样的逻辑抽象成图
3. **`crew/crew_agent.py`** → 最后感受 CrewAI 的高阶业务抽象
4. **`llm/qwen_llm.py`** → 随时查阅底层 LLM 调用细节

**核心对比**：`plain` 中的 `while` 循环 = `langx` 中的 `StateGraph` = `crew` 中隐式的 `Crew.kickoff()`。

---

## 许可证

MIT License
# talented-agent
