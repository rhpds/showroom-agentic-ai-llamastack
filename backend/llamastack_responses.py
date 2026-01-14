#!/usr/bin/env python3
"""
LlamaStack Responses API integration
Functions for using LlamaStack's modern Responses API with MCP tools and RAG
"""
import asyncio
import json
import logging
import re
from typing import List, Dict, Optional, AsyncGenerator

logger = logging.getLogger(__name__)


async def stream_response(
    client,
    model: str,
    user_message: str,
    instructions: str,
    tools: Optional[List[Dict]] = None,
    previous_response_id: Optional[str] = None,
    vector_store_id: Optional[str] = None
) -> AsyncGenerator[Dict, None]:
    """Create a streaming response using the Responses API

    Args:
        client: LlamaStackClient instance
        model: LLM model to use (e.g., "openai/gpt-4o")
        user_message: User's message/input
        instructions: System instructions/prompt
        tools: List of tool configurations (MCP tools, RAG tools, etc.)
        previous_response_id: ID of previous response for multi-turn conversations
        vector_store_id: Vector store ID for fetching file attributes if missing

    Yields: Response events from the Responses API
    """
    try:
        # Build the request parameters
        request_params = {
            "model": model,
            "input": user_message,
            "instructions": instructions,
            "stream": True
        }

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            logger.info(f"Configuring response with {len(tools)} tools")

        # Add previous response ID for multi-turn conversations
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id

        # Create the streaming response
        logger.info(f"Creating Responses API request with model: {model}")
        if previous_response_id:
            logger.info(f"Using previous_response_id for conversation continuity: {previous_response_id}")
        response = client.responses.create(**request_params)

        # Stream the response
        for event in response:
            yield event
            # Force event loop to yield and flush
            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"Error creating response: {e}")
        import traceback
        traceback.print_exc()
        raise


def fetch_file_attributes(client, vector_store_id: str, file_id: str) -> Optional[Dict]:
    """Fetch file attributes from LlamaStack API

    This is a workaround for a bug where attributes aren't returned in file_search
    results from the Responses API.

    Args:
        client: LlamaStackClient instance
        vector_store_id: Vector store ID
        file_id: File ID

    Returns:
        Dictionary of attributes or None if fetch fails
    """
    try:
        logger.info(f"Fetching file attributes via API: vector_store_id={vector_store_id}, file_id={file_id}")
        file_obj = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id
        )

        # Extract attributes from the file object
        attributes = getattr(file_obj, 'attributes', None)

        if attributes:
            logger.info(f"Successfully fetched attributes: {attributes}")
            return attributes
        else:
            logger.warning(f"File object has no attributes: {file_obj}")
            return None

    except Exception as e:
        logger.error(f"Failed to fetch file attributes: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_response_event_for_sse(chunk, client=None, vector_store_id: Optional[str] = None) -> Optional[str]:
    """Format a Responses API event for Server-Sent Events (SSE)

    The Responses API returns OpenAI-compatible streaming events.
    This function converts them to the SSE format expected by the frontend.

    Args:
        chunk: Event from Responses API
        client: LlamaStackClient instance (optional, for fetching file attributes)
        vector_store_id: Vector store ID (optional, for fetching file attributes)
    """
    try:
        # The chunk is the event itself with a 'type' field
        event_type = getattr(chunk, 'type', None)

        if not event_type:
            logger.debug(f"Event has no type field, skipping: {type(chunk).__name__}")
            return None

        logger.debug(f"Processing event type: {event_type}")

        # Handle reasoning text deltas (thinking/reasoning content)
        if event_type == 'response.reasoning_text.delta':
            delta = getattr(chunk, 'delta', None)
            if delta:
                return json.dumps({'reasoning': delta})

        # Handle output text deltas (final response content)
        elif event_type == 'response.output_text.delta':
            delta = getattr(chunk, 'delta', None)
            if delta:
                return json.dumps({'content': delta})

        # Handle response creation
        elif event_type == 'response.created':
            # Capture response_id for conversation continuity
            # The response_id is in the nested 'response' object
            response_id = None
            response_obj = getattr(chunk, 'response', None)
            if response_obj:
                response_id = getattr(response_obj, 'id', None)

            if response_id:
                logger.debug(f"Captured response_id for conversation continuity: {response_id}")
                return json.dumps({'status': 'Response started...', 'response_id': response_id})
            return json.dumps({'status': 'Response started...'})

        # Handle response completion
        elif event_type == 'response.done':
            return json.dumps({'status': 'Complete'})

        # Handle response failure
        elif event_type == 'response.failed':
            # Log the entire failed event for debugging
            logger.error(f"Response failed event received")
            logger.error(f"  Event type: {type(chunk).__name__}")
            if hasattr(chunk, 'model_dump'):
                logger.error(f"  Event data: {chunk.model_dump()}")
            else:
                logger.error(f"  Event repr: {repr(chunk)}")

            # Extract error information - check both chunk.error and chunk.response.error
            error_info = getattr(chunk, 'error', None)

            # If not found at top level, check in the nested response object
            if not error_info:
                response_obj = getattr(chunk, 'response', None)
                if response_obj:
                    error_info = getattr(response_obj, 'error', None)

            if error_info:
                error_code = getattr(error_info, 'code', 'unknown_error')
                error_message = getattr(error_info, 'message', 'An error occurred')

                logger.error(f"Response failed: {error_code} - {error_message}")

                # Create a helpful error message for the user
                user_message = f"Error: {error_message}\n\nThis may be due to a conversation context mismatch. Please try resetting your conversation and asking again."

                return json.dumps({'error': user_message})
            else:
                logger.error("No error info found in failed event - this is unexpected!")
                return json.dumps({'error': 'Response failed with unknown error. Please try resetting your conversation and asking again.'})

        # Handle output item start (e.g., message output starting)
        elif event_type == 'response.output_item.added':
            # Check if this is a tool call being added
            item = getattr(chunk, 'item', None)
            if item:
                item_type = getattr(item, 'type', None)
                if item_type == 'mcp_call':
                    # MCP tool call started
                    tool_id = getattr(item, 'id', None)
                    tool_name = getattr(item, 'name', None)

                    logger.info(f"MCP tool call added - id: {tool_id}, name: {tool_name}")

                    if tool_id and tool_name:
                        return json.dumps({
                            'tool_call': {
                                'id': tool_id,
                                'name': tool_name,
                                'state': 'input-available',
                                'input': None,
                                'output': None
                            }
                        })
                elif item_type == 'file_search_call':
                    # File search tool call started
                    tool_id = getattr(item, 'id', None)

                    # Extract search queries (it's an array)
                    queries = getattr(item, 'queries', [])

                    # Get the first query if available
                    query = None
                    if queries and len(queries) > 0:
                        query_raw = queries[0]
                        # The query might be JSON-encoded like '{"query": "search text"}'
                        if isinstance(query_raw, str):
                            try:
                                # Try to parse as JSON
                                query_obj = json.loads(query_raw)
                                # Extract the actual query text
                                query = query_obj.get('query', query_raw)
                            except:
                                # Not JSON, use as-is
                                query = query_raw
                        else:
                            query = str(query_raw)

                    logger.info(f"File search tool call added - id: {tool_id}, query: {query}")

                    if tool_id:
                        return json.dumps({
                            'tool_call': {
                                'id': tool_id,
                                'name': 'file_search',
                                'state': 'input-available',
                                'input': {'query': query} if query else None,
                                'output': None
                            }
                        })

            return json.dumps({'status': 'Generating response...'})

        # Handle content part boundaries
        elif event_type == 'response.content_part.added':
            # A new content part is being added (e.g., reasoning vs output)
            part = getattr(chunk, 'part', None)
            part_type = getattr(part, 'type', None) if part else None
            logger.debug(f"Content part added: {part_type}")
            # Don't send status for this - just structural marker
            return None

        elif event_type == 'response.content_part.done':
            # Content part completed
            part = getattr(chunk, 'part', None)
            part_type = getattr(part, 'type', None) if part else None
            logger.debug(f"Content part done: {part_type}")
            # Don't send status for this - just structural marker
            return None

        # Handle MCP events
        elif event_type == 'response.mcp_list_tools.in_progress':
            return json.dumps({'status': 'Loading tools...'})

        elif event_type == 'response.mcp_list_tools.completed':
            return json.dumps({'status': 'Tools loaded'})

        elif event_type == 'response.mcp_call.in_progress':
            # MCP tool call started - send tool call info to frontend
            call_id = getattr(chunk, 'call_id', None)
            tool_name = getattr(chunk, 'name', None)

            # Try to extract from nested mcp_call object if not found at top level
            if not call_id or not tool_name:
                mcp_call = getattr(chunk, 'mcp_call', None)
                if mcp_call:
                    call_id = call_id or getattr(mcp_call, 'call_id', None) or getattr(mcp_call, 'id', None)
                    tool_name = tool_name or getattr(mcp_call, 'name', None) or getattr(mcp_call, 'tool_name', None)

            logger.info(f"MCP call in progress - call_id: {call_id}, tool_name: {tool_name}")

            if call_id and tool_name:
                return json.dumps({
                    'tool_call': {
                        'id': call_id,
                        'name': tool_name,
                        'state': 'input-available',
                        'input': None,
                        'output': None
                    }
                })
            return json.dumps({'status': 'Calling tool...'})

        elif event_type == 'response.mcp_call.completed':
            # MCP tool call completed - send result to frontend
            call_id = getattr(chunk, 'call_id', None)
            tool_name = getattr(chunk, 'name', None)
            output = None

            # Try to extract from nested mcp_call object
            mcp_call = getattr(chunk, 'mcp_call', None)
            if mcp_call:
                call_id = call_id or getattr(mcp_call, 'call_id', None) or getattr(mcp_call, 'id', None)
                tool_name = tool_name or getattr(mcp_call, 'name', None) or getattr(mcp_call, 'tool_name', None)
                output = getattr(mcp_call, 'output', None) or getattr(mcp_call, 'result', None)

            logger.info(f"MCP call completed - call_id: {call_id}, tool_name: {tool_name}, has_output: {output is not None}")

            if call_id:
                return json.dumps({
                    'tool_call': {
                        'id': call_id,
                        'name': tool_name,
                        'state': 'output-available',
                        'output': str(output) if output else None
                    }
                })
            return json.dumps({'status': 'Tool execution complete'})

        # Handle file search events
        elif event_type == 'response.file_search_call.in_progress':
            # File search started - send tool call info
            call_id = getattr(chunk, 'call_id', None)

            if call_id:
                return json.dumps({
                    'tool_call': {
                        'id': call_id,
                        'name': 'file_search',
                        'state': 'input-available',
                        'input': None,
                        'output': None
                    }
                })
            return json.dumps({'status': 'Searching knowledge base...'})

        elif event_type == 'response.file_search_call.searching':
            return json.dumps({'status': 'Searching documents...'})

        elif event_type == 'response.file_search_call.completed':
            # File search completed - extract sources
            logger.info(f"File search completed, extracting sources")
            logger.info(f"Event data: {chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk}")

            # Extract results from the completed file search
            sources = []
            seen_sources = set()  # Track unique sources by URL to avoid duplicates

            # The results should be in chunk.results or chunk.file_search_call.results
            results = None
            if hasattr(chunk, 'results'):
                results = chunk.results
            elif hasattr(chunk, 'file_search_call'):
                file_search_call = chunk.file_search_call
                if hasattr(file_search_call, 'results'):
                    results = file_search_call.results

            if results:
                logger.info(f"Found {len(results) if isinstance(results, list) else 1} file search results")

                # Process each result
                if isinstance(results, list):
                    for result in results[:10]:  # Limit to 10 sources
                        # Debug: Log full result structure
                        logger.info(f"Processing file search result:")
                        logger.info(f"  Result type: {type(result).__name__}")
                        if hasattr(result, 'model_dump'):
                            logger.info(f"  Result data: {result.model_dump()}")
                        else:
                            logger.info(f"  Result repr: {repr(result)}")

                        # Extract metadata from attributes field
                        attributes = getattr(result, 'attributes', None)
                        logger.info(f"  Extracted attributes: {attributes}")

                        # If no attributes, try fetching from API (workaround for Responses API bug)
                        if not attributes and client and vector_store_id:
                            file_id = getattr(result, 'file_id', None)
                            if file_id:
                                logger.info(f"Attributes missing, fetching via API for file_id={file_id}")
                                attributes = fetch_file_attributes(client, vector_store_id, file_id)

                        if attributes:
                            # Read metadata from attributes (set during vector store upload)
                            title = attributes.get('title', 'Unknown Document')
                            source_url = attributes.get('source_url', '#')
                            content_type = attributes.get('content_type', 'workshop-content')

                            logger.info(f"Extracted metadata from attributes: title='{title}', url='{source_url}'")

                            # Only add if we haven't seen this source URL before
                            if source_url not in seen_sources:
                                sources.append({
                                    'title': title,
                                    'url': source_url,
                                    'content_type': content_type
                                })
                                seen_sources.add(source_url)
                        else:
                            # Fallback to filename/file_id if no attributes
                            file_name = getattr(result, 'filename', None) or getattr(result, 'name', 'Unknown')
                            file_id = getattr(result, 'file_id', None)

                            # Create source entry
                            source_url = f'/files/{file_id}' if file_id else '#'

                            # Only add if we haven't seen this source URL before
                            if source_url not in seen_sources:
                                sources.append({
                                    'title': file_name,
                                    'url': source_url,
                                    'content_type': 'file-search-result'
                                })
                                seen_sources.add(source_url)

            if sources:
                logger.info(f"Extracted {len(sources)} sources from file_search_call.completed")
                return json.dumps({'sources': sources})
            else:
                logger.warning("File search completed but no results found in event")
                return None

        # Handle output item done (contains file search results and tool call results)
        elif event_type == 'response.output_item.done':
            item = getattr(chunk, 'item', None)
            if not item:
                return None

            item_type = getattr(item, 'type', None)

            # Check if this is an MCP tool call completion
            if item_type == 'mcp_call':
                tool_id = getattr(item, 'id', None)
                tool_name = getattr(item, 'name', None)
                tool_arguments = getattr(item, 'arguments', None)
                tool_output = getattr(item, 'output', None)

                logger.info(f"MCP tool call completed - id: {tool_id}, name: {tool_name}, has_output: {tool_output is not None}")

                if tool_id:
                    # Parse arguments JSON if available
                    input_data = None
                    if tool_arguments:
                        try:
                            input_data = json.loads(tool_arguments)
                        except:
                            input_data = tool_arguments

                    return json.dumps({
                        'tool_call': {
                            'id': tool_id,
                            'name': tool_name,
                            'state': 'output-available',
                            'input': input_data,
                            'output': tool_output
                        }
                    })

            # Check if this is a file search result
            elif item_type == 'file_search_call':
                tool_id = getattr(item, 'id', None)
                results = getattr(item, 'results', [])

                # Extract search queries (it's an array)
                queries = getattr(item, 'queries', [])

                # Get the first query if available
                query = None
                if queries and len(queries) > 0:
                    query_raw = queries[0]
                    # In the .done event, queries are already parsed (not JSON strings)
                    if isinstance(query_raw, str):
                        query = query_raw
                    else:
                        query = str(query_raw)

                logger.info(f"File search completed - id: {tool_id}, query: {query}, result_count: {len(results) if results else 0}")

                # Build response with both tool completion and sources
                response_data = {}

                # Add tool call completion
                if tool_id:
                    result_count = len(results) if results else 0
                    response_data['tool_call'] = {
                        'id': tool_id,
                        'name': 'file_search',
                        'state': 'output-available',
                        'input': {'query': query} if query else None,
                        'output': f'Found {result_count} results' if result_count > 0 else 'No results found'
                    }

                # Extract sources if available
                if results:
                    logger.info(f"Found {len(results)} file search results in output_item.done")
                    sources = []
                    seen_sources = set()  # Track unique sources by URL to avoid duplicates

                    for result in results[:10]:  # Limit to 10 sources
                        # Debug: Log full result structure
                        logger.info(f"Processing file search result (output_item.done):")
                        logger.info(f"  Result type: {type(result).__name__}")
                        if hasattr(result, 'model_dump'):
                            logger.info(f"  Result data: {result.model_dump()}")
                        else:
                            logger.info(f"  Result repr: {repr(result)}")

                        # Extract metadata from attributes field
                        attributes = getattr(result, 'attributes', None)
                        logger.info(f"  Extracted attributes: {attributes}")

                        # If no attributes, try fetching from API (workaround for Responses API bug)
                        if not attributes and client and vector_store_id:
                            file_id = getattr(result, 'file_id', None)
                            if file_id:
                                logger.info(f"Attributes missing, fetching via API for file_id={file_id}")
                                attributes = fetch_file_attributes(client, vector_store_id, file_id)

                        if attributes:
                            # Read metadata from attributes (set during vector store upload)
                            title = attributes.get('title', 'Unknown Document')
                            source_url = attributes.get('source_url', '#')
                            content_type = attributes.get('content_type', 'workshop-content')

                            logger.info(f"Extracted metadata from attributes: title='{title}', url='{source_url}'")

                            # Only add if we haven't seen this source URL before
                            if source_url not in seen_sources:
                                sources.append({
                                    'title': title,
                                    'url': source_url,
                                    'content_type': content_type
                                })
                                seen_sources.add(source_url)
                        else:
                            # No attributes found - fall back to filename/file_id
                            logger.warning(f"No attributes found in result, falling back to filename")

                            file_name = getattr(result, 'filename', None) or getattr(result, 'name', None)
                            file_id = getattr(result, 'file_id', None)

                            if file_name:
                                # Clean up the file name to create a title
                                # Remove .txt extension and convert underscores/hyphens to spaces
                                title = file_name.replace('.txt', '').replace('_', ' ').replace('-', ' ').strip()
                                if not title:
                                    title = 'Unknown Document'

                                # Try to determine the source URL
                                # Check if this looks like a PDF-related file
                                if 'pdf' in file_name.lower() or (file_id and 'pdf' in file_id.lower()):
                                    # Assume it's from techdocs
                                    pdf_name = file_name.replace('.txt', '.pdf')
                                    source_url = f"/_/techdocs/{pdf_name}"
                                    content_type = 'pdf-documentation'
                                else:
                                    # Workshop content - use file_id if available
                                    source_url = f'/files/{file_id}' if file_id else '#'
                                    content_type = 'workshop-content'

                                logger.info(f"Using fallback metadata: title='{title}', url='{source_url}'")

                                # Only add if we haven't seen this source URL before
                                if source_url not in seen_sources:
                                    sources.append({
                                        'title': title,
                                        'url': source_url,
                                        'content_type': content_type
                                    })
                                    seen_sources.add(source_url)
                            else:
                                # No filename available either - skip this result
                                logger.warning(f"No filename or attributes found in result, skipping")
                                continue

                    if sources:
                        logger.info(f"Extracted {len(sources)} sources from file search")
                        response_data['sources'] = sources

                # Return combined response with both tool completion and sources
                if response_data:
                    return json.dumps(response_data)

            return None

        # Handle function calls (generic)
        elif event_type == 'response.function_call_arguments.delta':
            # Tool being called - could show this to user
            return json.dumps({'status': 'Using tools...'})

        elif event_type == 'response.function_call_arguments.done':
            # Generic function call complete
            function_call_id = getattr(chunk, 'call_id', None)
            function_name = getattr(chunk, 'name', None)
            logger.debug(f"Function call completed: {function_name} ({function_call_id})")

        # Handle errors
        elif event_type == 'error':
            error_msg = getattr(chunk, 'error', {}).get('message', 'Unknown error')
            return json.dumps({'error': error_msg})

        # Log unknown event types for debugging
        else:
            logger.debug(f"Unhandled event type: {event_type}")

        return None

    except Exception as e:
        logger.warning(f"Error formatting event: {e}")
        import traceback
        traceback.print_exc()
        return None
