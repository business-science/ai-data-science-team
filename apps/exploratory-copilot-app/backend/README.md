EDA Copilot API

开发运行

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

生产运行（建议）

- 使用 `uvicorn` 或 `gunicorn + uvicorn workers`
- 前置 Nginx/网关做 TLS、鉴权、限流
- 配置 CORS 的允许来源白名单

