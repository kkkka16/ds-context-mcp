# ds-context-mcp

ローカルのデータセット（CSV / Parquet / Feather）のスキーマ・統計情報・リレーションを LLM に伝える MCP サーバー。

GitHub Copilot や Claude Code のようなMCPクライアントに接続することで、「AIが存在しないカラム名をautocompleteしてくる」問題を解決します。

> 🚧 **Work in progress** — Zenn コンテスト「GitHub Copilot 活用選手権」応募作品として開発中

## 特徴

- 📂 プロジェクト内の CSV / Parquet / Feather を自動検出
- 🔍 スキーマ、dtype、欠損率、カーディナリティを LLM に提供
- 🔗 複数ファイル間の JOIN 候補を推定
- 🦆 DuckDB による ad-hoc SQL クエリ
- ⚡ Polars / pandas 両対応
- 🛡 パストラバーサル対策、サンドボックス済みパス検証

## 必要環境

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## インストール

```bash
git clone https://github.com/YOUR_NAME/ds-context-mcp.git
cd ds-context-mcp
uv sync
```

## クイックスタート

### Claude Code から使う

`~/.claude.json` に以下を追加:

```json
{
  "mcpServers": {
    "ds-context": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/ds-context-mcp", "run", "python", "-m", "ds_context_mcp.server"]
    }
  }
}
```

### VS Code の GitHub Copilot から使う

`.vscode/mcp.json` に以下を追加:

```json
{
  "mcpServers": {
    "ds-context": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/ds-context-mcp", "run", "python", "-m", "ds_context_mcp.server"]
    }
  }
}
```

## 提供ツール

| ツール | 説明 |
|---|---|
| `list_datasets` | 指定ディレクトリ配下のデータファイル一覧 |
| `describe_dataset` | スキーマ・dtype・欠損率・メモリサイズ |
| `sample_rows` | 先頭 N 行のプレビュー |
| `column_profile` | 特定カラムの分布・ユニーク数・外れ値 |
| `detect_relations` | 複数ファイル間の JOIN 候補 |
| `query_sql` | DuckDB による SQL クエリ |

## 開発

```bash
# 依存インストール
uv sync

# 開発サーバー起動（MCP Inspector UI）
uv run mcp dev src/ds_context_mcp/server.py

# テスト
uv run pytest

# Lint / Format / Type check
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

## ライセンス

MIT
