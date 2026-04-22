"""FastMCP entrypoint for ds-context-mcp.

このファイルは MCP サーバーのエントリポイント。ツールの実装は
`src/ds_context_mcp/tools/` 以下で行い、このファイルでは登録のみ行う。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ds_context_mcp.tools.datasets import register_dataset_tools
from ds_context_mcp.tools.relations import register_relation_tools
from ds_context_mcp.tools.sql import register_sql_tools

mcp: FastMCP = FastMCP("ds-context-mcp")
register_dataset_tools(mcp)
register_relation_tools(mcp)
register_sql_tools(mcp)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
