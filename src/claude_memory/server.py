from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from claude_memory.schema import BreakthroughParams, EntityCreateParams, RelationshipCreateParams
from claude_memory.tools import MemoryService

# Initialize MCP Server
mcp = FastMCP("claude-memory")

# Initialize Service
# We defer initialization to startup or first request
# but for simplicity we instantiate here.
service = MemoryService()


@mcp.tool()  # type: ignore
async def create_entity(
    name: str,
    node_type: str,
    project_id: str,
    properties: Dict[str, Any] = {},
    certainty: str = "confirmed",
    evidence: List[str] = [],
) -> Dict[str, Any]:
    """Create a new entity in the memory graph."""
    params = EntityCreateParams(
        name=name,
        node_type=node_type,
        project_id=project_id,
        properties=properties,
        certainty=certainty,
        evidence=evidence,
    )
    return await service.create_entity(params)  # type: ignore


@mcp.tool()  # type: ignore
async def create_relationship(
    from_entity: str,
    to_entity: str,
    relationship_type: str,
    properties: Dict[str, Any] = {},
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Create a relationship between two entities."""
    params = RelationshipCreateParams(
        from_entity=from_entity,
        to_entity=to_entity,
        relationship_type=relationship_type,
        properties=properties,
        confidence=confidence,
    )
    return await service.create_relationship(params)  # type: ignore


@mcp.tool()  # type: ignore
async def record_breakthrough(
    name: str,
    moment: str,
    session_id: str,
    analogy_used: Optional[str] = None,
    concepts_unlocked: List[str] = [],
) -> Dict[str, Any]:
    """Record a learning breakthrough."""
    params = BreakthroughParams(
        name=name,
        moment=moment,
        session_id=session_id,
        analogy_used=analogy_used,
        concepts_unlocked=concepts_unlocked,
    )
    return await service.record_breakthrough(params)  # type: ignore


@mcp.tool()  # type: ignore
async def search_memory(
    query: str, project_id: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for entities using hybrid search."""
    results = await service.search(query, project_id, limit)
    return [res.model_dump() for res in results]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
