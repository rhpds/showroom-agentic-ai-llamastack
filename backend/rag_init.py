#!/usr/bin/env python3
"""
RAG Initialization Module
Loads documents and uploads them to LlamaStack vector store
"""
import logging
import tempfile
import re
import json
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


async def initialize_vector_store(client, content_dir: str, pdf_dir: str, min_chunk_size: int = 100, embedding_model: str = "text-embedding-3-small"):
    """
    Initialize LlamaStack vector store with workshop documents

    Args:
        client: LlamaStackClient instance
        content_dir: Path to AsciiDoc content directory
        pdf_dir: Path to PDF documentation directory
        min_chunk_size: Minimum size for document chunks
        embedding_model: Embedding model to use (e.g., "text-embedding-3-small")

    Returns:
        vector_store_id: ID of created/existing vector store
    """
    vector_store_id = "workshop-docs"

    # First, collect all expected documents to know how many we should have
    logger.info("Scanning documents to determine expected file count...")
    all_content = []

    # Load AsciiDoc content
    content_path = Path(content_dir)
    if content_path.exists():
        adoc_content = _load_asciidoc_content(content_path, min_chunk_size)
        all_content.extend(adoc_content)
        logger.info(f"Found {len(adoc_content)} AsciiDoc documents")

    # Load PDF content
    pdf_path = Path(pdf_dir)
    if pdf_path.exists():
        pdf_content = _load_pdf_content(pdf_path, min_chunk_size)
        all_content.extend(pdf_content)
        logger.info(f"Found {len(pdf_content)} PDF documents")

    if not all_content:
        logger.warning("No documents found to upload")
        # Still create an empty vector store if it doesn't exist
        try:
            stores = client.vector_stores.list()
            existing_store = next((s for s in stores.data if s.id == vector_store_id or s.name == vector_store_id), None)
            if existing_store:
                return existing_store.id
            else:
                vector_store = client.vector_stores.create(name=vector_store_id)
                return vector_store.id
        except Exception as e:
            raise RuntimeError(f"Failed to create empty vector store: {e}") from e

    expected_file_count = len(all_content)
    logger.info(f"Expected file count: {expected_file_count}")

    # Check if vector store already exists and has the correct number of files
    try:
        stores = client.vector_stores.list()
        existing_store = next((s for s in stores.data if s.id == vector_store_id or s.name == vector_store_id), None)

        if existing_store:
            existing_store_id = existing_store.id
            logger.info(f"Found existing vector store: {existing_store_id}")

            # Check how many files are in the vector store
            try:
                vector_store_files = client.vector_stores.files.list(vector_store_id=existing_store_id)
                actual_file_count = len(list(vector_store_files.data)) if hasattr(vector_store_files, 'data') else len(list(vector_store_files))
                logger.info(f"Existing vector store has {actual_file_count} files")

                if actual_file_count == expected_file_count:
                    logger.info(f"✓ Vector store is up-to-date with {actual_file_count} files, skipping re-upload")
                    return existing_store_id
                else:
                    logger.warning(f"Vector store file count mismatch: expected {expected_file_count}, found {actual_file_count}")
                    logger.info("Deleting and recreating vector store...")
                    client.vector_stores.delete(vector_store_id=existing_store_id)
            except Exception as e:
                logger.warning(f"Could not query files in existing vector store: {e}")
                logger.info("Deleting and recreating vector store to be safe...")
                try:
                    client.vector_stores.delete(vector_store_id=existing_store_id)
                except Exception as delete_error:
                    logger.warning(f"Could not delete vector store: {delete_error}")
    except Exception as e:
        logger.warning(f"Error checking existing vector stores: {e}")

    # Create vector store with configured embedding model
    logger.info(f"Creating vector store '{vector_store_id}' with embedding model: {embedding_model}...")
    try:
        vector_store = client.vector_stores.create(
            name=vector_store_id
            # extra_body={
            #     "embedding_model": embedding_model
            # }
        )
        logger.info(f"Created vector store: {vector_store.id}")
        vector_store_id = vector_store.id
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise RuntimeError(f"Critical failure: could not create vector store: {e}") from e

    # Load and upload documents
    try:
        # Upload each document individually to preserve metadata for citations
        uploaded_count = 0
        for doc in all_content:
            try:
                # Create temporary file with just the content (no metadata header)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(doc['content'])
                    temp_file = f.name

                try:
                    # Step 1: Upload file to get file_id
                    with open(temp_file, 'rb') as f:
                        uploaded_file = client.files.create(
                            file=f,
                            purpose='assistants'
                        )
                        logger.info(f"Uploaded file: {uploaded_file.id} ({doc['title']})")

                    # Determine content type based on file path
                    source_url = doc['file_path']
                    if source_url.endswith('.pdf') or '/_/techdocs/' in source_url:
                        content_type = 'pdf-documentation'
                    else:
                        content_type = 'workshop-content'

                    # Step 2: Attach file to vector store with metadata attributes
                    attributes = {
                        'title': doc['title'],
                        'source_url': source_url,
                        'module': doc['module'],
                        'content_type': content_type
                    }

                    logger.info(f"Attaching file to vector store:")
                    logger.info(f"  vector_store_id: {vector_store_id}")
                    logger.info(f"  file_id: {uploaded_file.id}")
                    logger.info(f"  attributes: {attributes}")

                    file_result = client.vector_stores.files.create(
                        vector_store_id=vector_store_id,
                        file_id=uploaded_file.id,
                        attributes=attributes
                    )
                    uploaded_count += 1
                    logger.info(f"✓ Successfully attached with metadata: {doc['module']} - {doc['title']}")
                finally:
                    # Clean up temp file
                    Path(temp_file).unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Error uploading document '{doc['title']}': {e}")
                continue

        logger.info(f"Successfully uploaded {uploaded_count}/{len(all_content)} documents individually")

        # Ensure at least some documents were uploaded successfully
        if uploaded_count == 0 and len(all_content) > 0:
            raise RuntimeError(f"Critical failure: failed to upload any documents to vector store (0/{len(all_content)} succeeded)")

        return vector_store_id

    except Exception as e:
        logger.error(f"Error uploading documents to vector store: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Critical failure during document upload: {e}") from e


def _load_asciidoc_content(content_path: Path, min_chunk_size: int) -> List[dict]:
    """
    Load RAG-optimized content files exported by Antora extension.
    These files have resolved AsciiDoc attributes and metadata headers.
    """
    documents = []

    # Check if this is the new RAG content directory (flat structure with .txt files)
    # or the old content directory (modules/ROOT/pages structure with .adoc files)
    txt_files = list(content_path.glob("*.txt"))

    if txt_files:
        # New RAG content format - files with metadata headers
        logger.info(f"Loading RAG-optimized content from {content_path}")
        for exported_file in txt_files:
            # Skip special files
            if exported_file.stem in ['attrs-page', 'ai-chatbot', 'nav']:
                continue

            try:
                content = exported_file.read_text(encoding='utf-8')

                # Parse metadata and content
                if content.startswith('---\nMETADATA:\n'):
                    parts = content.split('---\n', 2)
                    if len(parts) >= 3:
                        metadata_json = parts[1].replace('METADATA:\n', '')
                        metadata = json.loads(metadata_json)
                        actual_content = parts[2].strip()

                        # Clean AsciiDoc markup from the content
                        cleaned_content = _clean_asciidoc(actual_content)

                        if len(cleaned_content.strip()) > min_chunk_size:
                            documents.append({
                                'title': metadata.get('title', exported_file.stem),
                                'content': cleaned_content,
                                'file_path': metadata.get('url', metadata.get('originalPath', str(exported_file))),
                                'module': f"{metadata.get('component', 'modules')} - {metadata.get('module', 'ROOT')}"
                            })
                else:
                    logger.warning(f"File {exported_file} doesn't have expected metadata format")
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing metadata in {exported_file}: {e}")
            except Exception as e:
                logger.warning(f"Error loading {exported_file}: {e}")
    else:
        # Fallback to old format - raw .adoc files
        logger.info(f"Loading raw AsciiDoc content from {content_path}")
        modules_dir = content_path / "modules" / "ROOT" / "pages"

        if not modules_dir.exists():
            logger.warning(f"No content found at {content_path} or {modules_dir}")
            return documents

        for adoc_file in modules_dir.glob("**/*.adoc"):
            # Skip special files
            if adoc_file.name in ['ai-chatbot.adoc', 'nav.adoc', 'attrs-page.adoc']:
                continue

            try:
                content = adoc_file.read_text(encoding='utf-8')
                title = _extract_title(content, adoc_file.stem)
                cleaned = _clean_asciidoc(content)

                if len(cleaned.strip()) > min_chunk_size:
                    documents.append({
                        'title': title,
                        'content': cleaned,
                        'file_path': str(adoc_file),
                        'module': _extract_module(str(adoc_file))
                    })
            except Exception as e:
                logger.warning(f"Error loading {adoc_file}: {e}")

    return documents


def _load_pdf_content(pdf_path: Path, min_chunk_size: int) -> List[dict]:
    """Load PDF files"""
    documents = []

    for pdf_file in pdf_path.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_file))
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if len(text.strip()) > min_chunk_size:
                title = pdf_file.stem.replace('_', ' ').replace('-', ' ').title()
                # Convert filesystem path to Antora URL path
                # /app/content/modules/ROOT/assets/techdocs/file.pdf -> /_/techdocs/file.pdf
                pdf_url = f"/_/techdocs/{pdf_file.name}"
                documents.append({
                    'title': title,
                    'content': text,
                    'file_path': pdf_url,
                    'module': 'PDF Documentation'
                })
        except Exception as e:
            logger.warning(f"Error loading PDF {pdf_file}: {e}")

    return documents


def _extract_title(content: str, fallback: str) -> str:
    """Extract title from AsciiDoc"""
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('= ') and not line.startswith('== '):
            return line[2:].strip()
    return fallback.replace('-', ' ').title()


def _clean_asciidoc(content: str) -> str:
    """Clean AsciiDoc markup"""
    lines = []
    in_header = True

    for line in content.split('\n'):
        stripped = line.strip()

        if in_header and (stripped.startswith(':') or stripped.startswith('//') or stripped == ''):
            continue
        in_header = False

        # Remove markup
        cleaned = re.sub(r'^=+\s+', '', line)
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
        cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        cleaned = re.sub(r'link:([^\[]+)\[([^\]]*)\]', r'\2', cleaned)
        cleaned = re.sub(r'image::?[^\[]+\[[^\]]*\]', '', cleaned)

        if cleaned.strip():
            lines.append(cleaned.strip())

    return '\n'.join(lines)


def _extract_module(file_path: str) -> str:
    """Extract module name from path"""
    match = re.search(r'module[-_](\d+)', file_path)
    if match:
        return f"Module {match.group(1)}"
    return "General"


if __name__ == "__main__":
    """
    Standalone script to pre-build vector embeddings for GitHub Actions

    Requirements:
        Python 3.12+ (required by llama-stack-client>=0.3.0)

    Usage:
        python rag_init.py --llama-stack-url http://localhost:8321 \
                          --content-dir ../rag-content \
                          --pdf-dir ../content/modules/ROOT/assets/techdocs \
                          --embedding-model text-embedding-3-small
    """
    import argparse
    import sys
    from llama_stack_client import LlamaStackClient

    parser = argparse.ArgumentParser(description='Pre-build vector embeddings')
    parser.add_argument('--llama-stack-url', required=True, help='LlamaStack server URL')
    parser.add_argument('--content-dir', required=True, help='Path to content directory')
    parser.add_argument('--pdf-dir', required=True, help='Path to PDF directory')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='Embedding model to use')
    parser.add_argument('--min-chunk-size', type=int, default=100, help='Minimum chunk size')
    parser.add_argument('--timeout', type=float, default=300.0, help='Client timeout in seconds')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting vector store pre-build...")
    logger.info(f"LlamaStack URL: {args.llama_stack_url}")
    logger.info(f"Content directory: {args.content_dir}")
    logger.info(f"PDF directory: {args.pdf_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")

    # Create LlamaStack client
    client = LlamaStackClient(
        base_url=args.llama_stack_url,
        timeout=args.timeout
    )

    # Run the initialization
    async def run():
        try:
            vector_store_id = await initialize_vector_store(
                client,
                args.content_dir,
                args.pdf_dir,
                args.min_chunk_size,
                args.embedding_model
            )
            logger.info(f"✓ Vector store successfully created: {vector_store_id}")
            return 0
        except Exception as e:
            logger.error(f"✗ Failed to create vector store: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Run the async function
    import asyncio
    exit_code = asyncio.run(run())
    sys.exit(exit_code)
