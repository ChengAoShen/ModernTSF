# 部署说明（Cloudflare Workers + 静态资源）

本站是 **Next.js 静态导出**（`next.config.mjs` 里 `output:"export"` → `out/`），部署在
**Cloudflare Workers** 上，域名 `diaugeia.ai`：

- `out/` 的静态文件由 **Workers Static Assets** 直接伺服（不占 Worker 3 MiB 代码上限）。
- 一个很小的 `worker.js` 只在 `POST /api/submit` 时运行，通过 **Email Routing 的
  `send_email` 原生绑定**把表单发到 `chengao_shen@ieee.org`——无第三方、无 API key。

> 旧的 GitHub Pages 链路已退役，后续只维护 Cloudflare。

---

## 架构要点

| 文件 | 作用 |
|------|------|
| `next.config.mjs` | `output:"export"`（静态导出到 `out/`） |
| `worker.js` | `POST /api/submit` → 校验/蜜罐/限长 → `env.SEB.send()`；其余 → `env.ASSETS.fetch()` |
| `wrangler.jsonc` | `main=worker.js`、`assets=out/`、`send_email` 绑定 `SEB`（锁定收件人） |

发信细节：`mimetext` 组装邮件，`Reply-To` 设为提交者邮箱，收到后可直接回复。`send_email`
绑定的 `destination_address` 锁死为 `chengao_shen@ieee.org`，只能发到这一个已验证邮箱。

---

## 1. GitHub 推送后自动部署（Cloudflare Workers Builds）

控制台 → **Workers & Pages** → 选中 `diaugeia` Worker → **Settings → Builds** →
**Connect**，选本仓库（`Diaugeia/HomePageSourceCode`），授权一次 GitHub。构建配置：

| 项 | 值 |
|----|----|
| Build command | `bun run build` |
| Deploy command | `bunx wrangler deploy` |
| Branch | `main` |
| Root directory | 仓库根 |

之后每次 push 到 `main`，Cloudflare 自动 `bun install` → 构建 → 部署。仓库里**不需要**任何
CI 文件，也不用管 API token。

> 也可本地一条命令部署：`bun run deploy`（= `next build && wrangler deploy`）。

---

## 2. send_email 绑定（一次性）

`send_email` 要求 **Email Routing 已启用 + 目标地址已验证**——本域都满足
（`chengao_shen@ieee.org` 是账号邮箱，已自动验证），`wrangler deploy` 会按
`wrangler.jsonc` 自动创建绑定。若报「需授权对外发信」，去 **Email Routing → Settings**
开启 Email Sending 即可。

---

## 3. 把域名从「跳转」改成「直连」（部署成功后）

当前 `diaugeia.ai` 是一条 301 跳转规则 `ai-to-com`（→ `diaugeia.com`），DNS 是占位 IP。
部署成功后：

1. `diaugeia` Worker → **Settings → Domains & Routes** → 添加 `diaugeia.ai` 和
   `www.diaugeia.ai`（Cloudflare 自动改 DNS 指向 Worker）。
2. **删除跳转规则**：`diaugeia.ai` → **Rules → Overview → Redirect Rules** → 删除 `ai-to-com`。
3. 打开 `https://diaugeia.ai` 确认直接显示本站。
4. 域名切换、验证 OK 后，旧公开仓库 `Diaugeia/HomePage` 可删。

---

## 4. 邮箱转发（已完成）

Cloudflare **Email Routing** 已启用，`cshen@diaugeia.ai` 与 `contact@diaugeia.ai`
转发到 `chengao_shen@ieee.org`。（这是「收信转发」，与表单「发信」的 `send_email`
绑定是两套独立机制。）

---

## 5. 本地开发 / 测试

```bash
bun run dev      # 普通 Next 开发；/api/submit 不可用（没有 worker）
bun run preview  # next build + wrangler dev：完整 worker，本地模拟 send_email
```

`wrangler dev` 下 `send_email` 是**模拟**（打印不真发），可验证表单链路：

```bash
curl -X POST http://localhost:8787/api/submit -H 'content-type: application/json' \
  -d '{"kind":"contact","name":"Test","email":"a@b.com","message":"hi"}'   # → {"ok":true}
```
