# CLAUDE.md

このファイルは Claude Code がこのリポジトリで作業する際の指示書です。

## プロジェクト概要

`ds-context-mcp` は、ローカルのデータセット（CSV / Parquet / Feather）のスキーマ・統計情報・リレーションを LLM に伝えるための MCP サーバーです。GitHub Copilot や Claude Code 等の MCP クライアントに接続することで、「Copilotが存在しないカラム名を提案してくる」問題を解消することを目的とします。

Zenn コンテスト「GitHub Copilot 活用選手権」（〜2026/4/30）への応募作品として開発しています。

## 開発方針

### 言語・ツール
- Python 3.12 固定（`.python-version`）
- パッケージ管理は **uv のみ**。`pip install` や `poetry` は禁止
- 依存追加は `uv add <pkg>`、開発依存は `uv add --dev <pkg>`
- 実行は `uv run <cmd>`

### コーディング規約
- **型ヒント必須**（`mypy --strict` が通ることを目指す）
- docstring は Google スタイル
- ファイル冒頭に `from __future__ import annotations` を入れる
- import 順: 標準 → サードパーティ → ローカル（ruff に従う）
- 1ファイル 300行を超えたら分割を検討

### MCP ツール実装ルール
- `src/ds_context_mcp/tools/` 配下に機能別ファイルで置く
- すべてのツール関数は:
  - 引数・戻り値に Pydantic モデルまたは TypedDict を使う
  - docstring の1行目は「LLMに見える説明」として書く（明確・簡潔）
  - エラー時は例外でなく構造化されたエラーレスポンスを返す
  - ファイルパスは必ず絶対パス化して検証する（path traversal 対策）
- 新しいツールを追加したら `tests/` に必ず対応するテストを書く

### テスト
- `uv run pytest` が通ることを PR の最低基準とする
- フィクスチャは `tests/fixtures/` に小さい（数KB）CSVを生成するスクリプトで用意
- 実データ（Kaggle等）はリポジトリにコミットしない

### Lint / 型チェック
- コミット前に必ず以下を実行:
  - `uv run ruff check .`
  - `uv run ruff format .`
  - `uv run mypy src/`

## やってはいけないこと

- `git commit` / `git push` を Claude Code 側で自動実行しない（ユーザーが行う）
- `main` ブランチへの直接 push をしない
- 実データセット・APIキー・個人情報をコミットしない
- 依存ライブラリを勝手に増やさない（提案はOK、追加は確認後）
- `requirements.txt` を作らない（uv管理のため）

## プロジェクト構造

````
src/ds_context_mcp/
  server.py          # FastMCPエントリポイント、tool登録
  tools/
    datasets.py      # list_datasets, describe_dataset, sample_rows
    columns.py       # column_profile
    relations.py     # detect_relations（JOIN候補推定）
  utils/
    readers.py       # CSV/parquet/feather の読み込み抽象
    formatters.py    # LLM向け出力整形
````

## 主要な MCP ツール（予定）

| ツール名 | 機能 | 優先度 |
|---|---|---|
| list_datasets | プロジェクト内のデータファイル一覧 | P0 |
| describe_dataset | スキーマ・dtype・欠損率・メモリ | P0 |
| sample_rows | 先頭N行を返す | P0 |
| column_profile | 特定カラムの詳細プロファイル | P0 |
| detect_relations | 複数ファイル間の共通カラム（JOIN候補） | P1 |
| query_sql | DuckDBでCSV/parquetにSQL実行 | P1 |
| project_memo | README.md / DATA.md の内容を返す | P2 |

## 作業の進め方

1. ユーザーから「この機能を実装して」と依頼されたら、まず該当ツールの**テストを先に書くか確認**する
2. 実装は小さい PR 単位（1ツール1コミット目安）
3. 実装後は必ず `mcp dev` コマンドで起動確認できるかを伝える

## 参考情報

- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- FastMCP は MCP SDK に統合済み（`from mcp.server.fastmcp import FastMCP`）
- Copilot SDK 連携は Phase 2（Copilotサブスク確定後）
