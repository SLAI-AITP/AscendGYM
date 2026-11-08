# AscendGYM 项目主页

项目主页与评测榜单，使用 Vue 3、Vite 和随包测量数据生成静态网站。评测框架的安装、模型配置和使用方法见[根 README](../README.md)。

## 本地预览

环境：Node.js 22、npm、Python 3.12（构建前的 shape 数据校验使用 Python 标准库）。

```bash
cd website
npm ci
npm run dev -- --host 127.0.0.1
```

开发地址为 `http://127.0.0.1:5173`。当前默认数据来自 `src/data/supplement_bundle.json` 与 `src/data/shape_results.json`；静态展示不需要模型密钥、AscendForge 服务或 NPU。

## 构建

```bash
npm run build
npm run preview -- --host 127.0.0.1
```

构建产物为 `dist/`，预览端口为 4174。构建前会从随包的 `public/templates/shape-results.csv` 校验并生成 shape 数据。默认相对资源路径及 hash 路由支持静态托管。

针对本项目 GitHub Pages 子路径的本地验证命令：

```bash
npm run build -- --base /AscendGYM/
npm run preview -- --host 127.0.0.1 --base /AscendGYM/
```

随后访问 `http://127.0.0.1:4174/AscendGYM/`。正式部署由工作流读取 Pages 的实际 base path，不在组件中写死站点根路径。

## 部署到 GitHub Pages

仓库已经提供 [pages.yml](../.github/workflows/pages.yml)。当前仅允许手动触发，普通代码推送不会自动发布网站。

1. 将主页与工作流提交并合入 `main`。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。设置页面需要相应的仓库管理权限。
3. 在 **Actions → Deploy project website → Run workflow** 选择 `main`，运行构建和部署。
4. 在成功的运行记录中打开部署地址，核对首页、Methodology、Imports、logo、技术报告和仓库按钮。

未设置自定义域名时，预计地址为 `https://slai-aitp.github.io/AscendGYM/`，以部署输出为准。当前文档不表示网站已经上线。

工作流只上传 `website/dist/`，使用 GitHub 自动提供的授权完成 Pages 部署，不需要把个人 token 写入仓库。参见 [GitHub Pages 官方配置说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 和 [Vite 部署说明](https://vite.dev/guide/static-deploy.html#github-pages)。

## 文件和数据

| 路径 | 内容 |
|---|---|
| `src/components/`、`src/pages/` | 页面和图表组件 |
| `src/data/` | 当前展示所用的随包数据与历史辅助数据 |
| `public/assets/logo/` | 项目 logo，根 README 直接引用同一份资源 |
| `public/assets/` | 图标与来源说明 |
| `public/AscendGYM_TechReport.pdf` | 技术报告 |
| `public/templates/` | shape 数据和导入模板 |
| `scripts/` | 构建、数据转换和检查工具 |
| `evaluate/` | 原主页项目携带的离线评估辅助脚本 |

`npm ci` 和生产构建使用随包数据即可。`data:sync`、`test:supplement` 需要额外的原始测量归档 `../eval_plot/png_600dpi_v2/`；这些归档不属于本站的构建依赖，也没有随本次主页源码复制。原有 `npm test` 包含该归档核对，因此不能把缺少归档时的执行结果表述为完整验收通过。

可以单独运行不依赖外部归档的现有检查：

```bash
npm run test:anchors
npm run test:mapping
npm run test:precision
npm run test:pricing
npm run test:pagination
npm run test:shapes
npm run test:submission
```

上述检查的范围与独立原始数据复核不同。更新测量值或重算指标时，还需核对相应原始记录及评测定义。

## 发布范围

源码、锁文件、随包数据、公开资源及来源声明纳入 Git；`node_modules/`、`dist/`、截图、临时日志、macOS 元数据和本地部署缓存不纳入。页面的 Hugging Face 入口保留为占位，等待真实的数据集发布地址。

许可范围见[主仓许可说明](../README.md#license)，第三方图标及其他材料保留各自来源与声明。
