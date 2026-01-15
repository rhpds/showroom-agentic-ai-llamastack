#!/usr/bin/env python3
"""
Showroom AI Assistant Backend v2
FastAPI service with LlamaStack multi-agent system, RAG, and MCP integration
"""
import asyncio
import os
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Literal
from pathlib import Path
import re
from contextlib import asynccontextmanager

import httpx
import yaml
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage, SystemMessage, CompletionMessage
from PyPDF2 import PdfReader

# Import LlamaStack Responses API helper
from llamastack_responses import stream_response, format_response_event_for_sse

# Import RAG initialization
from rag_init import initialize_vector_store

try:
    from fastmcp import Client as MCPClient
except ImportError:
    try:
        from fastmcp.client import Client as MCPClient
    except ImportError:
        try:
            from mcp.client import Client as MCPClient
        except ImportError:
            MCPClient = None

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_value = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=log_level_value)
logger = logging.getLogger(__name__)


# Load configuration
def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override values taking precedence"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    """Load configuration from assistant-config.yaml with optional local overrides"""
    # Try multiple config paths (local development, Docker, absolute)
    possible_paths = [
        os.getenv("ASSISTANT_CONFIG_PATH"),
        "../config/assistant-config.yaml",  # Local development
        "/app/config/assistant-config.yaml",  # Docker
        "./config/assistant-config.yaml",  # Running from project root
    ]

    config_data = {}
    config_file_path = None

    for config_path in possible_paths:
        if not config_path:
            continue

        config_file_path = Path(config_path)
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded base configuration from {config_file_path}")
                logger.info(f"Config keys: {list(config_data.keys())}")
                break
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file_path}: {e}")

    if not config_data:
        logger.warning("No configuration file found, using defaults")
        return {}

    # Check for local override file (assistant-config-local.yaml)
    # This is used for local development to override specific settings
    if config_file_path:
        local_override_path = config_file_path.parent / "assistant-config-local.yaml"
        if local_override_path.exists():
            try:
                with open(local_override_path, 'r') as f:
                    local_config = yaml.safe_load(f) or {}
                logger.info(f"Found local override config at {local_override_path}")
                logger.info(f"Local override keys: {list(local_config.keys())}")
                # Deep merge local config into base config
                config_data = deep_merge(config_data, local_config)
                logger.info("Merged local override configuration")
            except Exception as e:
                logger.warning(f"Failed to load local override from {local_override_path}: {e}")

    return config_data


class Config:
    def __init__(self):
        self.config_data = load_config()

        # LlamaStack Configuration
        self.LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", "http://localhost:8321")

        # LLM Configuration
        # Support both old (single engine) and new (multiple engines) format
        engines_config = self._get_config_value("llm.engines", None)
        if engines_config:
            # New format: list of engines
            self.LLM_ENGINES = engines_config if isinstance(engines_config, list) else [engines_config]
        else:
            # Fallback to old format for backward compatibility
            single_engine = self._get_config_value("llm.engine", "openai")
            self.LLM_ENGINES = [single_engine]

        self.LLM_MODEL = self._get_config_value("llm.model", "openai/gpt-4o")
        self.EMBEDDING_MODEL = self._get_config_value("llm.embedding_model", "text-embedding-3-small")
        self.LLM_TIMEOUT = self._get_config_value("llm.timeout", 120.0)

        # Per-engine configuration
        self.OPENAI_ENDPOINT = self._get_config_value("llm.openai.endpoint", None)

        self.VLLM_ENDPOINT = self._get_config_value("llm.vllm.endpoint", None)
        self.VLLM_MAX_TOKENS = self._get_config_value("llm.vllm.max_tokens", None)
        self.VLLM_TLS_VERIFY = self._get_config_value("llm.vllm.tls_verify", None)

        self.OLLAMA_ENDPOINT = self._get_config_value("llm.ollama.endpoint", None)

        # Log LLM engine configuration
        logger.info(f"LLM Engines: {', '.join(self.LLM_ENGINES)}")

        if "openai" in self.LLM_ENGINES and self.OPENAI_ENDPOINT:
            logger.info(f"OpenAI Endpoint: {self.OPENAI_ENDPOINT}")

        if "vllm" in self.LLM_ENGINES:
            if self.VLLM_ENDPOINT:
                logger.info(f"vLLM Endpoint: {self.VLLM_ENDPOINT}")
            if self.VLLM_MAX_TOKENS:
                logger.info(f"vLLM Max Tokens: {self.VLLM_MAX_TOKENS}")
            if self.VLLM_TLS_VERIFY is not None:
                logger.info(f"vLLM TLS Verify: {self.VLLM_TLS_VERIFY}")

        if "ollama" in self.LLM_ENGINES and self.OLLAMA_ENDPOINT:
            logger.info(f"Ollama Endpoint: {self.OLLAMA_ENDPOINT}")

        # Content directories
        # Default to local paths for development (relative to backend directory)
        # In production/Docker, these should be set via environment variables
        # Use RAG-optimized content with resolved AsciiDoc attributes
        self.CONTENT_DIR = os.getenv("CONTENT_DIR", "../rag-content")
        self.PDF_DIR = os.getenv("PDF_DIR", "../content/modules/ROOT/assets/techdocs")

        # Content processing
        self.MIN_CHUNK_SIZE = 100

    def _get_config_value(self, key_path: str, default_value):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config_data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default_value


config = Config()


# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="Current user message")
    agent_type: Optional[Literal["lab_content", "openshift_debugging", "auto"]] = Field(default="auto", description="Which agent to use")
    include_mcp: bool = Field(default=True, description="Whether to include MCP tools")
    page_context: Optional[str] = Field(default=None, description="Current page context")
    previous_response_id: Optional[str] = Field(default=None, description="Previous response ID for conversation continuity")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Search query for RAG system")
    max_tokens: Optional[int] = Field(default=500, description="Maximum tokens in response")


class MCPManager:
    """MCP tools manager (keeping existing MCP integration)"""

    def __init__(self, mcp_config: Dict):
        self.mcp_config = mcp_config
        self.client = None
        self._initialized = False
        self._client_available = MCPClient is not None

    async def initialize(self):
        """Initialize MCP client"""
        if self._initialized or not self._client_available:
            return

        if "mcpServers" not in self.mcp_config:
            logger.warning("No MCP servers configured")
            return

        try:
            self.client = MCPClient(self.mcp_config)
            self._initialized = True
            logger.info("MCP client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")

    async def get_tools(self) -> List[Dict]:
        """Get available MCP tools"""
        if not self._initialized:
            await self.initialize()

        if not self.client:
            return []

        try:
            async with self.client as client:
                tools = await client.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else tool.inputSchema
                    }
                    for tool in tools
                ]
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call MCP tool"""
        if not self._initialized:
            await self.initialize()

        if not self.client:
            return {"error": "MCP not initialized"}

        try:
            async with self.client as client:
                result = await client.call_tool(tool_name, arguments)

                # Extract text from result
                if hasattr(result, 'content') and result.content:
                    if isinstance(result.content, list):
                        texts = []
                        for item in result.content:
                            if hasattr(item, 'text'):
                                texts.append(item.text)
                        return {"result": "\n".join(texts)}
                    elif hasattr(result.content, 'text'):
                        return {"result": result.content.text}

                return {"result": str(result)}
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup MCP client"""
        self.client = None
        self._initialized = False


class MultiAgentSystem:
    """Multi-agent system using LlamaStack"""

    def __init__(self, llama_stack_url: str, mcp_manager: MCPManager, vector_store_id: Optional[str] = None, config_data: dict = None):
        self.llama_stack_url = llama_stack_url
        self.mcp_manager = mcp_manager
        self.vector_store_id = vector_store_id
        self.client = None
        self._config = config_data or {}

        # Load LLM model from config
        llm_config = self._config.get('llm', {})
        self.llm_model = llm_config.get('model', 'openai/gpt-4o')

        # Load MCP server configurations for Responses API
        mcp_config = self._config.get('mcp', {})
        self.mcp_servers = mcp_config.get('servers', {})

        # Allow environment variable override for MCP server URLs
        # This enables different URLs for local Podman (container names) vs OpenShift (localhost)
        mcp_server_url_override = os.getenv('MCP_SERVER_URL')
        if mcp_server_url_override and 'kubernetes' in self.mcp_servers:
            logger.info(f"Overriding MCP server URL with env var: {mcp_server_url_override}")
            self.mcp_servers['kubernetes']['url'] = mcp_server_url_override

        # Load agent configurations from config
        config_agents = self._config.get('agents', {})
        if config_agents:
            self.agents = {
                agent_id: {
                    "name": agent_config.get("name", f"Agent {agent_id}"),
                    "description": agent_config.get("description", ""),
                    "system_prompt": agent_config.get("system_prompt", ""),
                    "toolgroups": agent_config.get("toolgroups", []),
                    "keywords": agent_config.get("keywords", [])
                }
                for agent_id, agent_config in config_agents.items()
            }
            logger.info(f"Loaded {len(self.agents)} agents from configuration")
        else:
            # No agents in config - this is an error
            logger.error("No agents found in configuration file!")
            self.agents = {}

    async def initialize(self):
        """Initialize LlamaStack client"""
        self.client = LlamaStackClient(
            base_url=self.llama_stack_url,
            timeout=config.LLM_TIMEOUT
        )
        logger.info(f"LlamaStack client initialized at {self.llama_stack_url} with {config.LLM_TIMEOUT}s timeout")
        logger.info("Using Responses API - tools will be configured per-request")

    def _select_agent(self, message: str, agent_type: Optional[str] = None) -> str:
        """Select appropriate agent based on message content"""
        if agent_type and agent_type != "auto":
            return agent_type

        # Simple keyword-based routing using each agent's keywords
        message_lower = message.lower()

        # Score each agent based on keyword matches
        agent_scores = {}
        for agent_id, agent_config in self.agents.items():
            keywords = agent_config.get("keywords", [])
            score = sum(1 for kw in keywords if kw in message_lower)
            agent_scores[agent_id] = score

        # Return agent with highest score, or first agent if tie
        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            return best_agent[0]

        # Fallback to first agent if no scores
        return next(iter(self.agents.keys())) if self.agents else "lab_content"


    async def chat(
        self,
        message: str,
        agent_type: Optional[str] = None,
        include_mcp: bool = True,
        page_context: Optional[str] = None,
        previous_response_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response with selected agent"""

        # Select agent
        selected_agent = self._select_agent(message, agent_type)
        agent_config = self.agents[selected_agent]

        agent_name = agent_config["name"]
        yield f"data: {json.dumps({'status': f'Using {agent_name}...'})}\n\n"
        await asyncio.sleep(0.1)

        # Build system prompt
        system_prompt = agent_config["system_prompt"]
        if page_context:
            system_prompt += f"\n\nCURRENT PAGE: {page_context}"

        # Note: RAG context is now handled automatically by the RAG tool

        # Generate response
        yield f"data: {json.dumps({'status': 'Generating response...'})}\n\n"

        # Use LlamaStack Responses API
        async for chunk in self._stream_with_responses_api(
            message,
            system_prompt,
            selected_agent,
            include_mcp,
            previous_response_id
        ):
            yield chunk

    async def _stream_with_responses_api(
        self,
        message: str,
        system_prompt: str,
        agent_type: str,
        include_mcp: bool,
        previous_response_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response using LlamaStack Responses API"""

        # Log conversation continuity
        if previous_response_id:
            logger.info(f"Continuing conversation with response ID: {previous_response_id}")
        else:
            logger.info("New conversation detected")

        try:
            # Get agent configuration
            agent_config = self.agents.get(agent_type, {})
            configured_toolgroups = agent_config.get("toolgroups", [])

            # Build tools list for Responses API
            tools = []

            for toolgroup in configured_toolgroups:
                if toolgroup == "rag":
                    # Add file_search tool for RAG with vector store configuration
                    if self.vector_store_id:
                        tools.append({
                            "type": "file_search",
                            "vector_store_ids": [self.vector_store_id]
                        })
                        logger.info(f"Including file_search tool with vector store {self.vector_store_id}")

                elif toolgroup.startswith("mcp::"):
                    # Add MCP tool if MCP is enabled
                    if include_mcp:
                        # Extract server name from toolgroup (e.g., "mcp::kubernetes" -> "kubernetes")
                        server_name = toolgroup.replace("mcp::", "")
                        server_config = self.mcp_servers.get(server_name)

                        if server_config and 'url' in server_config:
                            mcp_tool = {
                                "type": "mcp",
                                "server_url": server_config['url'],
                                "server_label": f"{server_name.title()} tools"
                            }
                            # Add headers if configured
                            if 'headers' in server_config:
                                mcp_tool['headers'] = server_config['headers']

                            tools.append(mcp_tool)
                            logger.info(f"Including MCP tool: {server_name} at {server_config['url']}")

                elif toolgroup == "builtin::websearch":
                    # Add web_search tool
                    tools.append({
                        "type": "web_search"
                    })
                    logger.info(f"Including web_search tool")

            logger.info(f"Agent '{agent_type}' configured with {len(tools)} tools")

            # Stream the response using Responses API
            async for event in stream_response(
                self.client,
                model=self.llm_model,
                user_message=message,
                instructions=system_prompt,
                tools=tools if tools else None,
                previous_response_id=previous_response_id,
                vector_store_id=self.vector_store_id
            ):
                # Format event for SSE (response_id is now included in the formatted output)
                formatted = format_response_event_for_sse(event, client=self.client, vector_store_id=self.vector_store_id)
                if formatted:
                    yield f"data: {formatted}\n\n"

        except Exception as e:
            logger.error(f"LlamaStack Responses API error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _generate_attribution(self, sources: List[Dict]) -> str:
        """Generate source attribution"""
        if not sources:
            return ""

        workshop_sources = [s for s in sources if s['content_type'] != 'pdf-documentation']
        pdf_sources = [s for s in sources if s['content_type'] == 'pdf-documentation']

        parts = []

        if workshop_sources:
            links = []
            seen = set()
            for source in workshop_sources[:5]:
                title = source['title']
                file_path = source.get('file_path', '')

                # Convert to HTML URL
                if file_path.endswith('.adoc'):
                    filename = Path(file_path).stem
                    html_url = f"{filename}.html"

                    if title not in seen:
                        seen.add(title)
                        links.append(f'link:{html_url}[*{title}*]')

            if links:
                parts.append("RELEVANT WORKSHOP LINKS:\n" + "\n".join(links))

        if pdf_sources:
            pdf_names = list(set(s['title'] for s in pdf_sources[:3]))
            if pdf_names:
                parts.append("TECHDOC REFERENCES:\n" + "\n".join(pdf_names))

        if parts:
            return "\n\n---\n\n" + "\n\n".join(parts)

        return ""


# Global instances
mcp_manager = None
agent_system = None


# Application lifespan
@asynccontextmanager
async def lifespan(app):
    """Application startup and shutdown"""
    global mcp_manager, agent_system

    logger.info("Application starting up...")

    # Initialize LlamaStack client for vector store initialization
    llama_client = LlamaStackClient(
        base_url=config.LLAMA_STACK_URL,
        timeout=config.LLM_TIMEOUT
    )

    # Check for existing vector store (pre-built vectors from init container)
    # Do NOT generate vectors at runtime - they should be pre-built in the container image
    # Retry logic to wait for LlamaStack to be ready
    vector_store_id = None
    max_retries = 25
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Checking for existing vector store (attempt {attempt}/{max_retries})...")
            stores = llama_client.vector_stores.list()

            # Look for workshop-docs vector store
            existing_store = None
            for store in stores.data if hasattr(stores, 'data') else stores:
                store_id = store.id if hasattr(store, 'id') else store.get('id')
                store_name = store.name if hasattr(store, 'name') else store.get('name')
                if store_name == "workshop-docs":
                    existing_store = store
                    vector_store_id = store_id
                    break

            if existing_store:
                # Check file count
                file_counts = existing_store.file_counts if hasattr(existing_store, 'file_counts') else existing_store.get('file_counts', {})
                completed = file_counts.completed if hasattr(file_counts, 'completed') else file_counts.get('completed', 0)
                total = file_counts.total if hasattr(file_counts, 'total') else file_counts.get('total', 0)

                logger.info(f"✓ Found existing vector store: {vector_store_id}")
                logger.info(f"  Files: {completed} completed / {total} total")

                if completed == 0:
                    logger.warning("⚠ Vector store exists but has 0 completed files!")
                    logger.warning("  RAG functionality will not work properly.")
            else:
                logger.warning("⚠ No vector store found!")
                logger.warning("  RAG functionality will be disabled.")
                logger.warning("  Vector embeddings should be pre-built via GitHub Actions and loaded by init container.")

            # Success - break out of retry loop
            break

        except Exception as e:
            logger.warning(f"⚠ Attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                logger.info(f"  Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                # Final attempt failed - give up and exit
                logger.error(f"✗ Failed to connect to LlamaStack after {max_retries} attempts")
                logger.error(f"  Last error: {e}")
                logger.error("  Container will exit. Please check that LlamaStack is running and accessible.")
                raise RuntimeError(f"Failed to connect to LlamaStack after {max_retries} attempts: {e}") from e

    # Load MCP config
    mcp_config = config.config_data.get('mcp', {})
    if 'servers' in mcp_config:
        # Transform to mcpServers format
        cleaned_servers = {}
        for server_name, server_config in mcp_config['servers'].items():
            # Check if this is a remote server (has 'url') or stdio server (has 'command')
            if 'url' in server_config:
                # Remote MCP server
                cleaned_servers[server_name] = {
                    "url": server_config["url"]
                }
            else:
                # Stdio MCP server
                cleaned_servers[server_name] = {
                    "command": server_config.get("command"),
                    "args": server_config.get("args", []),
                    "env": server_config.get("env", {})
                }

        mcp_config = {"mcpServers": cleaned_servers}

    # Initialize MCP manager
    mcp_manager = MCPManager(mcp_config)
    await mcp_manager.initialize()

    # Initialize agent system with vector store ID and config
    agent_system = MultiAgentSystem(
        config.LLAMA_STACK_URL,
        mcp_manager,
        vector_store_id,
        config.config_data  # Pass the full config data
    )
    await agent_system.initialize()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Application shutting down...")
    if mcp_manager:
        mcp_manager.cleanup()


# FastAPI app
app = FastAPI(
    title="Showroom AI Assistant Backend v2",
    description="Multi-agent AI Assistant with LlamaStack, RAG, and MCP integration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.post("/api/chat/stream")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses with multi-agent support"""

    async def generate():
        yield "data: {\"status\": \"starting\"}\n\n"
        await asyncio.sleep(0)

        async for chunk in agent_system.chat(
            chat_request.message,
            chat_request.agent_type,
            chat_request.include_mcp,
            chat_request.page_context,
            chat_request.previous_response_id
        ):
            yield chunk
            await asyncio.sleep(0)  # Force flush

        yield "data: {\"status\": \"complete\"}\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/rag/query")
async def query_rag(rag_request: RAGQueryRequest):
    """Direct vector store search endpoint for testing RAG"""

    # Check if vector store is available
    if not agent_system or not agent_system.vector_store_id:
        raise HTTPException(
            status_code=503,
            detail="RAG not available - vector store not initialized"
        )

    try:
        logger.info(f"Vector store search query: {rag_request.query}")
        logger.info(f"Vector store ID: {agent_system.vector_store_id}")

        # Call the vector store search API directly
        search_url = f"{config.LLAMA_STACK_URL}/v1/vector_stores/{agent_system.vector_store_id}/search"

        search_payload = {
            "query": rag_request.query,
            "max_num_results": 5
        }

        logger.info(f"Calling vector store search API: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(search_url, json=search_payload)
            response.raise_for_status()
            search_results = response.json()

        logger.info(f"Search API response: {search_results}")

        # Extract results from the response
        results = []
        data = search_results.get('data', [])

        for item in data:
            file_id = item.get('file_id', 'unknown')
            filename = item.get('filename', 'unknown')
            score = item.get('score', 0.0)

            # Extract text from content array
            text = ""
            content = item.get('content', [])
            for content_item in content:
                if content_item.get('type') == 'text':
                    text += content_item.get('text', '')

            results.append({
                'file_id': file_id,
                'filename': filename,
                'score': score,
                'text': text[:500]  # First 500 chars
            })

        logger.info(f"Retrieved {len(results)} search results")

        return {
            "query": rag_request.query,
            "vector_store_id": agent_system.vector_store_id,
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Vector store search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Vector store search failed: {str(e)}"
        )


@app.get("/api/agents")
async def get_agents():
    """Get available agents"""
    return {
        "agents": [
            {
                "id": agent_id,
                "name": config["name"],
                "description": config["description"]
            }
            for agent_id, config in agent_system.agents.items()
        ]
    }


@app.get("/api/config")
async def get_config():
    """Get frontend configuration including example questions"""
    example_questions = config._get_config_value("workshop.example_questions", [
        "What is this workshop about?",
        "Help me understand more about Red Hat"
    ])

    return {
        "example_questions": example_questions
    }


@app.get("/api/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools"""
    tools = await mcp_manager.get_tools()
    return {"tools": tools}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    errors = []

    # Check if agent system is initialized
    if not agent_system or not agent_system.client:
        errors.append("Agent system not initialized")

    # Check if LlamaStack is reachable
    llama_stack_healthy = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.LLAMA_STACK_URL}/v1/models")
            llama_stack_healthy = response.status_code == 200
    except Exception as e:
        errors.append(f"LlamaStack unreachable: {str(e)}")

    # Check if MCP manager initialized
    if mcp_manager and not mcp_manager._initialized:
        errors.append("MCP manager not initialized")

    # Check if MCP servers are configured
    mcp_servers = []
    if agent_system and agent_system.mcp_servers:
        mcp_servers = list(agent_system.mcp_servers.keys())

    # Check if vector store initialized
    if not agent_system or not agent_system.vector_store_id:
        errors.append("Vector store not initialized")

    # Determine overall health
    is_healthy = len(errors) == 0

    response_data = {
        "status": "healthy" if is_healthy else "unhealthy",
        "version": "2.0.0",
        "api_type": "responses",  # Using Responses API now
        "llama_stack": {
            "enabled": True,
            "url": config.LLAMA_STACK_URL,
            "healthy": llama_stack_healthy
        },
        "vector_store": agent_system.vector_store_id if agent_system else None,
        "rag_enabled": agent_system.vector_store_id is not None if agent_system else False,
        "agents": list(agent_system.agents.keys()) if agent_system else [],
        "mcp_initialized": mcp_manager._initialized if mcp_manager else False,
        "mcp_servers": mcp_servers
    }

    if errors:
        response_data["errors"] = errors
        raise HTTPException(status_code=503, detail=response_data)

    return response_data


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "showroom-assistant-backend",
        "version": "2.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level="info",
        timeout_keep_alive=75
    )
