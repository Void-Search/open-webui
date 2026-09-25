"""Gather local candidates under the authenticated caller's existing access rules."""

from urllib.parse import quote

from .transport import authorized_revisions, saved


async def research_sources(user, question, queries):
    hits = await saved(user, question, queries)
    result = []
    for hit in hits:
        source = hit['source']
        document, revision = source['document_id'], source['revision_id']
        identity = f'research:{document}:{revision}'
        link = (
            '/api/v1/ravenous/research/knowledge/documents/'
            + quote(document, safe='')
            + '/download?revision_id='
            + quote(revision, safe='')
        )
        result.append(
            {
                'source': {'id': identity, 'name': source.get('title') or document, 'type': 'research'},
                'document': [hit['text']],
                'metadata': [
                    {
                        **source,
                        'source': identity,
                        'link': link,
                        'research_kind': 'saved',
                    'verified_at': hit.get('verified_at'),
                    'chunk_id': hit.get('id'),
                        'source_url': hit.get('source_url'),
                    }
                ],
            }
        )
    return result


async def native_sources(
    request, user, question, queries, attachments, emit, *, selected_only=False, model=None, metadata=None
):
    from open_webui.models.config import Config
    from open_webui.models.groups import Groups
    from open_webui.models.knowledge import Knowledges
    from open_webui.retrieval.utils import get_sources_from_items
    from open_webui.utils.tools import get_attached_knowledge

    groups = [group.id for group in await Groups.get_groups_by_member_id(user.id)]
    items = [*attachments, *get_attached_knowledge(model or {}, metadata or {})]
    offset, count = 0, 0
    while not selected_only:
        page = await Knowledges.search_knowledge_bases(
            user.id,
            filter={'user_id': user.id, 'group_ids': groups},
            skip=offset,
            limit=100,
            raise_on_error=True,
        )
        for knowledge in page.items:
            if (knowledge.meta or {}).get('source') == 'external':
                continue
            items.append({'id': knowledge.id, 'name': knowledge.name, 'type': 'collection'})
            count += 1
        await emit({'stage': 'local', 'store': 'native', 'status': 'searching', 'collections': count})
        if len(page.items) < 100:
            break
        offset += 100
    config = await Config.get_many('rag.hybrid_bm25_weight', 'rag.enable_hybrid_search')
    sources = await get_sources_from_items(
        request=request,
        items=items,
        queries=queries,
        rerank_query=question,
        embedding_function=lambda values, prefix: request.app.state.EMBEDDING_FUNCTION(
            values, prefix=prefix, user=user
        ),
        k=75,
        reranking_function=None,
        k_reranker=75,
        r=0,
        hybrid_bm25_weight=config.get('rag.hybrid_bm25_weight', 0.5),
        hybrid_search=config.get('rag.enable_hybrid_search', True),
        user=user,
        candidate_only=True,
    )
    for source in sources:
        for meta in source.get('metadata', []):
            file_id = meta.get('file_id')
            original = str(meta.get('source') or source['source'].get('id', 'local'))
            if original.startswith(('https://', 'http://')):
                meta.setdefault('source_url', original)
            meta['source'] = 'native:' + str(file_id or original)
            meta['research_kind'] = 'local'
            meta['retrieval_authority'] = {
                'type': source['source'].get('type'),
                'id': source['source'].get('id'),
            }
            if file_id:
                meta['link'] = '/api/v1/files/' + quote(str(file_id), safe='') + '/content'
    return sources


async def native_authorization(user, kind, identifier):
    try:
        if kind == 'collection':
            from open_webui.models.knowledge import Knowledges

            allowed = await Knowledges.check_access_by_user_id(identifier, user.id, permission='read')
        else:
            from open_webui.utils.access_control.files import has_access_to_file

            allowed = await has_access_to_file(identifier, 'read', user)
        return None if allowed else 'access_revoked'
    except Exception:
        return 'authorization_unavailable'


async def authorize_sources(user, selected):
    """Return source failures; text supplied directly by the user needs no stored ACL."""
    failures = {}
    references = {
        (item['metadata']['document_id'], item['metadata']['revision_id'])
        for item in selected
        if item['metadata'].get('research_kind') == 'saved'
        and item['metadata'].get('document_id')
        and item['metadata'].get('revision_id')
    }
    try:
        permitted = await authorized_revisions(user, references)
    except Exception:
        permitted = set()
        for item in selected:
            if item['metadata'].get('research_kind') == 'saved':
                failures[item['source_id']] = 'authorization_unavailable'
    for item in selected:
        meta = item['metadata']
        if meta.get('document_id') and (meta['document_id'], meta.get('revision_id')) not in permitted:
            failures.setdefault(item['source_id'], 'access_revoked')
    checked = {}
    for item in selected:
        authority = item['metadata'].get('retrieval_authority') or {}
        kind, identifier = authority.get('type'), authority.get('id')
        if kind not in ('collection', 'file') or not identifier:
            continue
        key = (kind, identifier)
        if key not in checked:
            checked[key] = await native_authorization(user, kind, identifier)
        if checked[key]:
            failures[item['source_id']] = checked[key]
    return failures
