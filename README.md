# Course RAG FastAPI

一个面向课程资料的 RAG 知识库服务。支持上传 Markdown/TXT 文件、向量检索、带会话历史的问答，以及轻量网页界面。

它是独立服务；另一个 `RAG Agent FastAPI` 项目通过 HTTP 调用本服务，把知识库检索作为 Agent 工具。

## 核心能力

- 上传 UTF-8 编码的 `.txt` 和 `.md` 课程资料
- 文本切分后使用 `text-embedding-v4` 生成向量
- Chroma 持久化向量数据库，实现相似度检索
- MySQL 保存知识文件元数据
- 文件原文、MySQL 记录、Chroma 切片三处同步管理
- MD5 内容去重，避免重复上传相同资料
- 基于 `qwen3-max` 的 RAG 问答
- 基于 `session_id` 的多轮对话历史
- FastAPI 接口文档与轻量网页问答界面

## 技术栈

Python、FastAPI、LangChain、通义千问、Chroma、MySQL、SQLAlchemy、Docker。

## 架构

用户上传资料 → 文件校验与 MD5 去重 → 文本切分 → 向量化 → Chroma。

用户提问 → Chroma 检索相关资料 → 组装提示词与会话历史 → 大语言模型生成回答。

## 本地配置

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 中填写百炼密钥和 MySQL 信息：

```env
DASHSCOPE_API_KEY=你的百炼API密钥
DB_PASSWORD=你的MySQL密码
```

不要上传 `.env`。

## 启动

确保 MySQL 已启动后，在项目根目录运行：

```bash
python init_db.py
uvicorn api:app --reload --port 8000
```

打开：

- 网页界面：http://127.0.0.1:8000
- 接口文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

## 主要接口

- `POST /api/knowledge/upload`：上传 `.txt` 或 `.md` 资料
- `GET /api/knowledge/files`：分页查看资料
- `DELETE /api/knowledge/files/{file_id}`：删除资料及关联向量
- `POST /api/rag/chat`：基于知识库进行问答

问答请求示例：

```json
{
  "question": "INFS7410 的 Weekly quizzes 如何计分？",
  "session_id": "demo-session"
}
```

## Docker

项目提供 `Dockerfile`，可作为 Agent 项目 Docker Compose 中的 RAG 服务镜像构建。

## 后续迭代

- 支持 PDF、Word 等更多文档格式
- 增加重排序与混合检索，提高召回准确率
- 加入 RAG 评测集与检索指标
- 作为独立 RAG 服务接入更多 Agent 工具