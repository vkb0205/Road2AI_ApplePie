# --- build generation contexts from the retrieved Hit list ---------------
# Hit carries exactly the fields the generator consumes.
# # __ADAPTIVE_K_PORT__ SCORING-SELECTION  (adaptive-k port: scoring + selection) ----------

def article_key(c):
    law_id = str(c.get('law_id', '')).strip()
    ten = str(c.get('ten_van_ban', '')).strip()
    dieu = str(c.get('dieu_so', '')).strip()
    if law_id and ten and dieu:
        return f'{law_id}|{ten}|{dieu}'
    return ''

def source_kind(source_label):
    label = str(source_label)
    if ':' in label:
        return label.split(':', 1)[1]
    return label

def canonical_article_key(c):
    law_id = str(c.get('law_id', '')).strip()
    dieu = str(c.get('dieu_so', '')).strip()
    if law_id and dieu:
        return f'{law_id}|{dieu}'
    return article_key(c)


def score_candidate_domain(candidate, query_plan=None):
    query_plan = query_plan or {}
    profile = query_plan.get('domain_profile', {}) or {}
    law_id = str(candidate.get('law_id', '')).strip()
    title = str(candidate.get('ten_van_ban', ''))
    hay = anchor_plain(' '.join([law_id, title, str(candidate.get('dieu_so', ''))]))
    score = 0.0
    reasons = []

    preferred_law_ids = set(profile.get('preferred_law_ids', []))
    soft_negative_law_ids = set(profile.get('soft_negative_law_ids', []))
    if law_id in preferred_law_ids:
        score += 0.75
        reasons.append('preferred_law_id')
    if law_id in soft_negative_law_ids:
        score -= 0.70
        reasons.append('soft_negative_law_id')

    for term in profile.get('preferred_title_terms', []):
        if anchor_plain(term) in hay:
            score += 0.25
            reasons.append('preferred_title:' + term)
            break

    for term in profile.get('negative_title_terms', []):
        if anchor_plain(term) in hay and law_id not in preferred_law_ids:
            score -= 0.80
            reasons.append('negative_title:' + term)
            break

    labels = set(profile.get('labels', []))
    if 'copyright_assessment' in labels and ('chuyen giao cong nghe' in hay or 'thuong mai' in hay) and law_id not in preferred_law_ids:
        score -= 0.55
        reasons.append('assessment_domain_drift')
    if 'copyright_software' in labels and 'so huu cong nghiep' in hay and law_id not in preferred_law_ids:
        score -= 0.80
        reasons.append('industrial_property_drift')
    if not reasons:
        reasons.append('neutral')
    return float(score), reasons[:6]

def is_generic_variant_text(text, query_plan=None):
    profile = (query_plan or {}).get('domain_profile', {})
    try:
        return is_generic_atomic_question(text, profile)
    except Exception:
        plain = anchor_plain(text)
        return _plain_has_any(plain, ['tai lieu', 'chung cu', 'ho so', 'don yeu cau', 'hop dong', 'tranh chap'])

def candidate_facet_haystack(candidate):
    pieces = [
        str(candidate.get('law_id', '')),
        str(candidate.get('ten_van_ban', '')),
        str(candidate.get('dieu_so', '')),
        str(candidate.get('article_key', '')),
        str(candidate.get('chunk_text', ''))[:2400],
    ]
    return anchor_plain(' '.join(pieces))

def score_one_facet(candidate, facet, support_variants=None):
    support_variants = support_variants or []
    law_id = str(candidate.get('law_id', '')).strip()
    hay = candidate_facet_haystack(candidate)
    facet_id = str(facet.get('facet_id', '')).strip()
    rank_score = 0.0
    evidence_score = 0.0
    reasons = []
    preferred_law_ids = set(str(x).strip() for x in facet.get('preferred_law_ids', []) if str(x).strip())
    if law_id and law_id in preferred_law_ids:
        rank_score += 1.10
        reasons.append('preferred_law_id:' + law_id)
    for term in facet.get('preferred_title_terms', []):
        if anchor_plain(term) in hay:
            rank_score += 0.20
            reasons.append('preferred_title:' + term)
            break
    anchor_hits = []
    for term in facet.get('anchor_terms', []):
        if anchor_plain(term) in hay:
            anchor_hits.append(term)
    if anchor_hits:
        rank_score += min(0.30, 0.08 * len(set(anchor_plain(x) for x in anchor_hits)))
        evidence_score += min(0.22, 0.06 * len(set(anchor_plain(x) for x in anchor_hits)))
        reasons.append('anchor_terms:' + ','.join(anchor_hits[:4]))
    target_hits = []
    for term in facet.get('target_terms', []):
        if anchor_plain(term) in hay:
            target_hits.append(term)
    if target_hits:
        rank_score += min(0.90, 0.18 * len(set(anchor_plain(x) for x in target_hits)))
        evidence_score += min(1.25, 0.30 * len(set(anchor_plain(x) for x in target_hits)))
        reasons.append('target_terms:' + ','.join(target_hits[:4]))
    if facet_id and any(facet_id in str(v) for v in support_variants):
        rank_score += 0.45
        evidence_score += 0.55
        reasons.append('facet_variant_support')
    for term in facet.get('negative_terms', []):
        if anchor_plain(term) in hay and law_id not in preferred_law_ids:
            rank_score -= 0.45
            evidence_score -= 0.25
            reasons.append('negative_term:' + term)
            break
    return float(rank_score), float(evidence_score), reasons[:6]

def score_candidate_facets(candidate, query_plan=None, support_variants=None):
    query_plan = query_plan or {}
    support_variants = support_variants or []
    facets = query_plan.get('facet_profiles', []) or []
    if not facets:
        return 0.0, [], [], {}, {}
    matched = []
    reasons = []
    score_map = {}
    evidence_map = {}
    best = 0.0
    for facet in facets:
        facet_id = str(facet.get('facet_id', '')).strip()
        score, evidence_score, facet_reasons = score_one_facet(candidate, facet, support_variants=support_variants)
        score_map[facet_id] = float(score)
        evidence_map[facet_id] = float(evidence_score)
        best = max(best, float(score))
        threshold = 0.45 if facet.get('preferred_law_ids') else 0.55
        if evidence_score >= threshold:
            matched.append(facet_id)
            reasons.append({'facet_id': facet_id, 'score': float(score), 'evidence_score': float(evidence_score), 'reasons': facet_reasons})
    return float(best), matched, reasons[:8], score_map, evidence_map

def aggregate_article_candidates_from_variants(variant_results, query_plan=None):
    groups = OrderedDict()
    for vr in variant_results:
        variant = vr['variant']
        kind = str(variant.get('kind', ''))
        weight = float(variant.get('weight', 1.0))
        for rank, c in enumerate(vr.get('reranked', []), start=1):
            full_key = article_key(c)
            key = canonical_article_key(c)
            if not key:
                continue
            if key not in groups:
                groups[key] = {
                    'canonical_article_key': key,
                    'article_key': full_key,
                    'law_id': str(c.get('law_id', '')).strip(),
                    'ten_van_ban': str(c.get('ten_van_ban', '')).strip(),
                    'dieu_so': str(c.get('dieu_so', '')).strip(),
                    'best_context': dict(c),
                    'best_rerank_score': float(c.get('rerank_score', -1e9)),
                    'best_rrf_score': float(c.get('rrf_score', 0.0)),
                    'best_variant_kind': kind,
                    'best_variant_rank': rank,
                    'support_count': 0,
                    'support_rrf_sum': 0.0,
                    'support_variants': set(),
                    'source_labels': [],
                    'per_variant': {},
                    'generic_atomic_support': False,
                }
            g = groups[key]
            score = float(c.get('rerank_score', -1e9))
            g['support_count'] += 1
            g['support_rrf_sum'] += float(c.get('rrf_score', 0.0))
            g['support_variants'].add(kind)
            g['source_labels'].extend(s.get('source', '') for s in c.get('source_hits', []))
            if str(kind).startswith('atomic_') and is_generic_variant_text(variant.get('text', ''), query_plan):
                g['generic_atomic_support'] = True
            prev = g['per_variant'].get(kind)
            if prev is None or score > prev['best_rerank_score']:
                g['per_variant'][kind] = {
                    'best_rank': rank,
                    'best_rerank_score': score,
                    'best_rrf_score': float(c.get('rrf_score', 0.0)),
                    'weight': weight,
                }
            if score > g['best_rerank_score']:
                g['best_context'] = dict(c)
                g['best_rerank_score'] = score
                g['best_rrf_score'] = float(c.get('rrf_score', 0.0))
                g['best_variant_kind'] = kind
                g['best_variant_rank'] = rank
                g['article_key'] = full_key
                g['ten_van_ban'] = str(c.get('ten_van_ban', '')).strip()

    candidates = []
    for g in groups.values():
        variants = sorted(v for v in g['support_variants'] if v)
        original_support = 'original' in variants
        atomic_coverage_count = len([v for v in variants if v.startswith('atomic_')])
        facet_support_count = len([v for v in variants if v.startswith('facet_')])
        early_bonus = 0.20 / max(int(g['best_variant_rank']), 1)
        original_bonus = 0.24 if original_support else 0.0
        atomic_bonus = 0.18 * min(atomic_coverage_count, 4)
        facet_bonus = 0.12 * min(facet_support_count, 4)
        support_bonus = 0.05 * math.log1p(g['support_count'])
        rrf_bonus = 0.22 * g['support_rrf_sum']
        weak_penalty = -0.20 if variants == ['must_terms'] else 0.0
        domain_score, domain_reasons = score_candidate_domain(g['best_context'], query_plan)
        facet_score, matched_facets, facet_reasons, facet_scores, facet_evidence_scores = score_candidate_facets(g['best_context'], query_plan, support_variants=variants)
        complexity = (query_plan or {}).get('complexity', 'medium')
        domain_weight = 0.0 if complexity == 'simple' else (0.08 if complexity == 'medium' else 0.18)
        facet_weight = 0.0 if complexity == 'simple' else (0.34 if complexity == 'medium' else 0.40)
        generic_penalty = -0.35 if g.get('generic_atomic_support') and not original_support and domain_score < 0.30 and facet_score < 0.90 else 0.0
        article_score_raw = float(g['best_rerank_score'] + early_bonus + original_bonus + atomic_bonus + facet_bonus + support_bonus + rrf_bonus + weak_penalty)
        article_score = float(article_score_raw + domain_weight * domain_score + facet_weight * facet_score + generic_penalty)
        c = dict(g['best_context'])
        c['canonical_article_key'] = g['canonical_article_key']
        c['article_key'] = g['article_key']
        c['article_score'] = article_score
        c['article_score_raw'] = article_score_raw
        c['domain_score'] = float(domain_score)
        c['domain_reasons'] = domain_reasons
        c['facet_score'] = float(facet_score)
        c['best_facet_score'] = float(facet_score)
        c['facet_scores'] = facet_scores
        c['facet_evidence_score'] = float(max(facet_evidence_scores.values()) if facet_evidence_scores else 0.0)
        c['facet_evidence_scores'] = facet_evidence_scores
        c['matched_facets'] = matched_facets
        c['facet_reasons'] = facet_reasons
        c['generic_atomic_support'] = bool(g.get('generic_atomic_support', False))
        c['support_count'] = g['support_count']
        c['support_rrf_sum'] = float(g['support_rrf_sum'])
        c['support_variants'] = variants
        c['source_kinds'] = variants
        c['source_labels'] = sorted(set(g['source_labels']))[:16]
        c['original_support'] = bool(original_support)
        c['atomic_support_count'] = int(atomic_coverage_count)
        c['atomic_coverage_count'] = int(atomic_coverage_count)
        c['facet_support_count'] = int(facet_support_count)
        c['best_variant_kind'] = g['best_variant_kind']
        c['best_variant_rank'] = int(g['best_variant_rank'])
        c['best_variant_rerank_score'] = float(g['best_rerank_score'])
        c['per_variant'] = g['per_variant']
        candidates.append(c)
    candidates.sort(key=lambda x: x['article_score'], reverse=True)
    return candidates



def article_law_family(cand):
    return str(cand.get('law_id', '')).strip()

def add_selected_article(selected, seen, cand, reason, family_counts=None, max_per_family=None):
    key = cand.get('canonical_article_key') or canonical_article_key(cand)
    if not key or key in seen:
        return False
    family = article_law_family(cand)
    count_before = family_counts.get(family, 0) if family_counts is not None else 0
    if family_counts is not None and max_per_family is not None and count_before >= max_per_family:
        return False
    item = dict(cand)
    item.setdefault('selection_reasons', [])
    item['selection_reasons'] = list(item.get('selection_reasons', [])) + [reason]
    item['law_family_count_before_select'] = int(count_before)
    selected.append(item)
    seen.add(key)
    if family_counts is not None:
        family_counts[family] = count_before + 1
    return True

def candidates_for_variant(article_candidates, kind):
    out = [c for c in article_candidates if kind in c.get('per_variant', {})]
    out.sort(key=lambda c: (
        c.get('per_variant', {}).get(kind, {}).get('best_rank', 10**9),
        -float(c.get('best_facet_score', 0.0)),
        -float(c.get('domain_score', 0.0)),
        -float(c.get('per_variant', {}).get(kind, {}).get('best_rerank_score', -1e9)),
    ))
    return out

def facet_profiles_for_selection(query_plan):
    facets = list((query_plan or {}).get('facet_profiles', []) or [])
    facets.sort(key=lambda f: -float(f.get('priority', 1.0) or 1.0))
    return facets

def candidate_allowed_for_selection(cand, complexity, selected_len, min_k, require_facet=False):
    domain_score = float(cand.get('domain_score', 0.0))
    facet_score = float(cand.get('best_facet_score', cand.get('facet_score', 0.0)))
    if complexity == 'simple':
        return True
    if require_facet and facet_score < 0.85:
        return False
    if domain_score <= -0.75 and not cand.get('original_support', False) and facet_score < 1.10 and selected_len >= min_k:
        return False
    if complexity == 'medium':
        return domain_score > -1.00 or cand.get('original_support', False) or facet_score >= 0.85 or selected_len < min_k
    return True

def facet_score_for_candidate(cand, facet):
    facet_id = str(facet.get('facet_id', '')).strip()
    return float((cand.get('facet_scores') or {}).get(facet_id, 0.0))

def facet_evidence_score_for_candidate(cand, facet):
    facet_id = str(facet.get('facet_id', '')).strip()
    return float((cand.get('facet_evidence_scores') or {}).get(facet_id, 0.0))

def candidate_law_preferred_for_facet(cand, facet):
    return str(cand.get('law_id', '')).strip() in set(str(x).strip() for x in facet.get('preferred_law_ids', []) if str(x).strip())

def candidate_matches_facet(cand, facet):
    evidence = facet_evidence_score_for_candidate(cand, facet)
    threshold = 0.45 if facet.get('preferred_law_ids') else 0.55
    return evidence >= threshold

def candidates_for_facet(article_candidates, facet):
    facet_id = str(facet.get('facet_id', '')).strip()
    out = []
    for cand in article_candidates:
        evidence = facet_evidence_score_for_candidate(cand, facet)
        if candidate_matches_facet(cand, facet) or (candidate_law_preferred_for_facet(cand, facet) and evidence >= 0.30):
            out.append(cand)
    out.sort(key=lambda c: (
        -float(candidate_law_preferred_for_facet(c, facet)),
        -facet_evidence_score_for_candidate(c, facet),
        -facet_score_for_candidate(c, facet),
        int(c.get('best_variant_rank', 10**9)),
        -float(c.get('article_score', 0.0)),
    ))
    return out

def preferred_doc_rescue_candidates(article_candidates, query_plan):
    profile = query_plan.get('domain_profile', {}) if query_plan else {}
    preferred = set(profile.get('preferred_law_ids', []))
    if not preferred:
        return []
    out = [c for c in article_candidates if str(c.get('law_id', '')).strip() in preferred]
    out.sort(key=lambda c: (
        -float(c.get('best_facet_score', 0.0)),
        -float(c.get('domain_score', 0.0)),
        -float(c.get('article_score', 0.0)),
        int(c.get('best_variant_rank', 10**9)),
    ))
    return out

def preferred_facet_candidate_exists(article_candidates, facet):
    preferred = set(str(x).strip() for x in facet.get('preferred_law_ids', []) if str(x).strip())
    if not preferred:
        return False
    return any(candidate_law_preferred_for_facet(cand, facet) and candidate_matches_facet(cand, facet) for cand in article_candidates)

def selected_covers_facet(selected, facet, article_candidates=None):
    preferred = set(str(x).strip() for x in facet.get('preferred_law_ids', []) if str(x).strip())
    require_preferred = bool(preferred) and preferred_facet_candidate_exists(article_candidates or [], facet)
    for cand in selected:
        if require_preferred and not candidate_law_preferred_for_facet(cand, facet):
            continue
        if candidate_matches_facet(cand, facet):
            return True
    return False

def selected_covered_facets(selected, query_plan=None, article_candidates=None):
    if query_plan:
        covered = set()
        for facet in facet_profiles_for_selection(query_plan):
            if selected_covers_facet(selected, facet, article_candidates=article_candidates):
                covered.add(str(facet.get('facet_id', '')))
        return covered
    covered = set()
    for cand in selected:
        for facet_id in cand.get('matched_facets', []):
            covered.add(facet_id)
    return covered

def low_value_duplicate_facet(cand, covered_facets):
    matched = set(cand.get('matched_facets', []))
    if not matched:
        return False
    if not matched.issubset(covered_facets):
        return False
    return float(cand.get('best_facet_score', 0.0)) < 1.20 and not cand.get('original_support', False)

def select_facet_coverage(article_candidates, query_plan, complexity, selected, seen, family_counts, max_k, min_k, coverage_debug):
    facets = facet_profiles_for_selection(query_plan)
    for facet in facets:
        if len(selected) >= max_k:
            coverage_debug.append({'facet_id': facet.get('facet_id', ''), 'status': 'skipped', 'reason': 'max_k'})
            break
        facet_id = facet.get('facet_id', '')
        if selected_covers_facet(selected, facet, article_candidates=article_candidates):
            coverage_debug.append({'facet_id': facet_id, 'status': 'covered_before', 'reason': 'already_covered'})
            continue
        added = False
        skip_reasons = []
        facet_candidates = candidates_for_facet(article_candidates, facet)
        for allow_over_cap in ([False, True] if complexity == 'complex' else [True]):
            for cand in facet_candidates:
                key = cand.get('canonical_article_key') or canonical_article_key(cand)
                if key in seen:
                    skip_reasons.append('seen')
                    continue
                if not candidate_matches_facet(cand, facet):
                    skip_reasons.append('low_facet_evidence')
                    continue
                if not candidate_allowed_for_selection(cand, complexity, len(selected), min_k, require_facet=True):
                    skip_reasons.append('domain_or_low_facet')
                    continue
                max_per_family = 2 if (complexity == 'complex' and not allow_over_cap) else None
                if add_selected_article(selected, seen, cand, f'facet_coverage:{facet_id}', family_counts=family_counts, max_per_family=max_per_family):
                    coverage_debug.append({
                        'facet_id': facet_id,
                        'status': 'selected',
                        'article_key': cand.get('article_key', ''),
                        'law_id': cand.get('law_id', ''),
                        'dieu_so': cand.get('dieu_so', ''),
                        'facet_score': facet_score_for_candidate(cand, facet),
                        'facet_evidence_score': facet_evidence_score_for_candidate(cand, facet),
                        'over_family_cap': bool(allow_over_cap),
                    })
                    added = True
                    break
                skip_reasons.append('family_cap')
            if added:
                break
        if not added:
            coverage_debug.append({
                'facet_id': facet_id,
                'status': 'missed',
                'candidate_count': len(facet_candidates),
                'skip_reasons': sorted(set(skip_reasons))[:6],
            })

def select_article_contexts(article_candidates, query_plan, variants=None):
    complexity = query_plan.get('complexity', 'medium')
    bounds = article_bounds_for_complexity(complexity)
    variants = variants or []
    if not article_candidates:
        return [], {'selected_k': 0, 'reason': 'no_candidates', **bounds, 'complexity': complexity, 'coverage_debug': []}

    max_k = min(bounds['max_k'], len(article_candidates))
    min_k = min(bounds['min_k'], max_k)
    selected, seen = [], set()
    family_counts = {}
    coverage_debug = []
    reason = 'facet_coverage_aware_v2'

    if complexity == 'simple':
        add_selected_article(selected, seen, article_candidates[0], 'simple_top1')
        if len(article_candidates) > 1 and len(selected) < max_k:
            top = float(article_candidates[0].get('article_score', 0.0))
            second = float(article_candidates[1].get('article_score', 0.0))
            same_law = article_candidates[0].get('law_id') == article_candidates[1].get('law_id')
            if same_law and second >= top - 0.55:
                add_selected_article(selected, seen, article_candidates[1], 'simple_close_same_law')
    else:
        select_facet_coverage(article_candidates, query_plan, complexity, selected, seen, family_counts, max_k, min_k, coverage_debug)

        original_quota = 1 if complexity == 'medium' else 2
        for cand in candidates_for_variant(article_candidates, 'original')[:original_quota]:
            if len(selected) >= max_k:
                break
            if candidate_allowed_for_selection(cand, complexity, len(selected), min_k):
                add_selected_article(selected, seen, cand, 'original_quota', family_counts=family_counts)

        if complexity == 'complex':
            rescue_added = 0
            for cand in preferred_doc_rescue_candidates(article_candidates, query_plan):
                if len(selected) >= max_k or rescue_added >= 4:
                    break
                if not candidate_allowed_for_selection(cand, complexity, len(selected), min_k):
                    continue
                if low_value_duplicate_facet(cand, selected_covered_facets(selected, query_plan, article_candidates)) and len(selected) >= min_k:
                    continue
                if add_selected_article(selected, seen, cand, 'preferred_doc_rescue', family_counts=family_counts, max_per_family=2):
                    rescue_added += 1

        base_variant_quota = 1 if complexity == 'medium' else 2
        for v in variants:
            kind = str(v.get('kind', ''))
            if not (kind.startswith('atomic_') or kind.startswith('facet_')):
                continue
            added_for_variant = 0
            for cand in candidates_for_variant(article_candidates, kind):
                if len(selected) >= max_k:
                    break
                if not candidate_allowed_for_selection(cand, complexity, len(selected), min_k):
                    continue
                if low_value_duplicate_facet(cand, selected_covered_facets(selected, query_plan, article_candidates)) and len(selected) >= min_k:
                    continue
                if added_for_variant >= base_variant_quota:
                    if not (float(cand.get('best_facet_score', 0.0)) >= 1.10 or cand.get('original_support', False)):
                        continue
                max_per_family = 3 if complexity == 'complex' else None
                if add_selected_article(selected, seen, cand, f'{kind}_quota', family_counts=family_counts, max_per_family=max_per_family):
                    added_for_variant += 1
                if added_for_variant >= base_variant_quota + 1:
                    break

        for cand in article_candidates:
            if len(selected) >= max_k:
                break
            if float(cand.get('domain_score', 0.0)) <= -0.75 and len(selected) >= min_k:
                continue
            if low_value_duplicate_facet(cand, selected_covered_facets(selected, query_plan, article_candidates)) and len(selected) >= min_k:
                continue
            if candidate_allowed_for_selection(cand, complexity, len(selected), min_k):
                add_selected_article(selected, seen, cand, 'score_fill', family_counts=family_counts, max_per_family=3 if complexity == 'complex' else None)

        if len(selected) < min_k:
            for cand in article_candidates:
                if len(selected) >= min_k or len(selected) >= max_k:
                    break
                add_selected_article(selected, seen, cand, 'min_k_backfill', family_counts=family_counts)

    selected.sort(key=lambda x: x.get('article_score', 0.0), reverse=True)
    debug_rows = []
    for rank, c in enumerate(article_candidates[:max(max_k, CFG['candidate_article_debug_topk'])], start=1):
        debug_rows.append({
            'rank': rank,
            'article_key': c.get('article_key', ''),
            'canonical_article_key': c.get('canonical_article_key', ''),
            'article_score': float(c.get('article_score', 0.0)),
            'article_score_raw': float(c.get('article_score_raw', c.get('article_score', 0.0))),
            'domain_score': float(c.get('domain_score', 0.0)),
            'domain_reasons': list(c.get('domain_reasons', [])),
            'facet_score': float(c.get('facet_score', 0.0)),
            'best_facet_score': float(c.get('best_facet_score', 0.0)),
            'facet_evidence_score': float(c.get('facet_evidence_score', 0.0)),
            'facet_evidence_scores': dict(c.get('facet_evidence_scores', {})),
            'matched_facets': list(c.get('matched_facets', [])),
            'facet_reasons': list(c.get('facet_reasons', [])),
            'generic_atomic_support': bool(c.get('generic_atomic_support', False)),
            'rerank_score': float(c.get('rerank_score', 0.0)),
            'best_variant_kind': c.get('best_variant_kind', ''),
            'best_variant_rank': int(c.get('best_variant_rank', 0)),
            'original_support': bool(c.get('original_support', False)),
            'atomic_coverage_count': int(c.get('atomic_coverage_count', 0)),
            'facet_support_count': int(c.get('facet_support_count', 0)),
            'support_variants': list(c.get('support_variants', [])),
            'law_family_count_before_select': c.get('law_family_count_before_select', ''),
        })
    return selected, {
        'complexity': complexity,
        'selected_k': int(len(selected)),
        'reason': reason,
        **bounds,
        'covered_facets': sorted(selected_covered_facets(selected, query_plan, article_candidates)),
        'coverage_debug': coverage_debug,
        'score_debug': debug_rows,
    }


# # __ADAPTIVE_K_PORT__ ADAPTER  (colab List[Hit] -> kaggle article_candidates dict) ----------
def _hit_to_candidate(h, rank, variant_kind, rrf_score=0.0):
    """Convert a retrieval Hit (or its dict form) into a kaggle-style
    candidate dict carrying the keys consumed by score_candidate_domain,
    score_candidate_facets, article_key, canonical_article_key, and the
    generation context builder."""
    if isinstance(h, dict):
        d = h
    else:
        d = h.to_dict() if hasattr(h, 'to_dict') else {
            'row_idx': getattr(h, 'row_idx', 0),
            'score': getattr(h, 'score', 0.0),
            'source': getattr(h, 'source', ''),
            'law_id': getattr(h, 'law_id', '') or '',
            'ten_van_ban': getattr(h, 'ten_van_ban', '') or '',
            'dieu_so': getattr(h, 'dieu_so', '') or '',
            'chunk_id': getattr(h, 'chunk_id', '') or '',
            'doc_uid': getattr(h, 'doc_uid', '') or '',
            'chunk_text': getattr(h, 'chunk_text', '') or '',
        }
    c = {
        'law_id': str(d.get('law_id', '') or '').strip(),
        'ten_van_ban': str(d.get('ten_van_ban', '') or '').strip(),
        'dieu_so': str(d.get('dieu_so', '') or '').strip(),
        'chunk_id': str(d.get('chunk_id', '') or '').strip(),
        'doc_uid': str(d.get('doc_uid', '') or '').strip(),
        'chunk_text': str(d.get('chunk_text', '') or ''),
        'row_idx': int(d.get('row_idx', 0) or 0),
        'source': str(d.get('source', '') or ''),
        'rerank_score': float(d.get('score', 0.0) or 0.0),
        'rrf_score': float(rrf_score or 0.0),
        'source_hits': [{'source': str(d.get('source', '') or '')}],
        'variant_kind': str(variant_kind or ''),
    }
    return c


def build_article_candidates_from_hits(hits, query_plan, variants=None, sub_query_traces=None):
    """Bridge the colab DecomposingHybridRetriever flat Hit list to the
    kaggle article_candidates schema consumed by select_article_contexts.

    The colab retriever already fuses per-sub-query Hit lists by row_idx
    (DecomposingHybridRetriever._fuse_hits), so we treat the single fused
    `hits` list as the 'original' variant ranking. When sub_query_traces
    are available (decomp_retriever.last_sub_query_traces), each trace's
    final_hits becomes an 'atomic_N' variant ranking, giving the aggregator
    genuine per-variant support signals without re-running retrieval.
    """
    variants = variants or build_query_variants(str(query_plan.get('question', '')), query_plan)
    sub_query_traces = sub_query_traces if sub_query_traces is not None else []
    topk = int(CFG.get('candidate_topk', 60))
    rerank_topk = int(CFG.get('rerank_topk', 48))

    variant_results = []

    # 'original' variant = the fused hit list.
    orig_variant = {'kind': 'original', 'weight': 1.0, 'text': str(query_plan.get('question', ''))}
    orig_reranked = []
    for rank, h in enumerate(hits[:rerank_topk], start=1):
        rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)
        c = _hit_to_candidate(h, rank, 'original', rrf_score=rrf)
        orig_reranked.append(c)
    variant_results.append({'variant': orig_variant, 'reranked': orig_reranked})

    # 'atomic_N' variants = per-sub-query traces (the decomposed leg hits).
    for idx, tr in enumerate(sub_query_traces, start=1):
        kind = f'atomic_{idx}'
        text = getattr(tr, 'sub_query_text', '') or (tr.get('sub_query_text') if isinstance(tr, dict) else '')
        final_hits = getattr(tr, 'final_hits', None)
        if final_hits is None and isinstance(tr, dict):
            final_hits = tr.get('final_hits', [])
        final_hits = list(final_hits or [])
        if not final_hits:
            continue
        reranked = []
        for rank, h in enumerate(final_hits[:rerank_topk], start=1):
            rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)
            reranked.append(_hit_to_candidate(h, rank, kind, rrf_score=rrf))
        variant_results.append({'variant': {'kind': kind, 'weight': 0.85, 'text': text}, 'reranked': reranked})

    # 'must_terms' variant = the must-terms query string (re-uses the same
    # fused hits but tags the top rerank_topk so the aggregator can apply its
    # weak_penalty / support logic.
    if CFG.get('use_must_terms_variant', True):
        mq = must_terms_query(query_plan)
        if mq:
            mt_reranked = []
            for rank, h in enumerate(hits[:rerank_topk], start=1):
                rrf = 1.0 / (CFG.get('rrf_k', 60) + rank)
                mt_reranked.append(_hit_to_candidate(h, rank, 'must_terms', rrf_score=rrf))
            variant_results.append({'variant': {'kind': 'must_terms', 'weight': 0.6, 'text': mq}, 'reranked': mt_reranked})

    candidates = aggregate_article_candidates_from_variants(variant_results, query_plan=query_plan)
    return candidates[:topk]


# kaggle dict-based helpers renamed to avoid clobbering the src import.
def build_gen_contexts_from_articles(article_contexts):
    gen_contexts = []
    for c in article_contexts:
        item = dict(c)
        item['chunk_text'] = str(item.get('chunk_text', ''))[:CFG['gen_chunk_char_limit']]
        gen_contexts.append(item)
        if len(gen_contexts) >= CFG['gen_context_topk']:
            break
    return gen_contexts


def make_relevant_lists_from_articles(article_contexts):
    docs, articles = [], []
    seen_docs, seen_articles = set(), set()
    for c in article_contexts:
        law_id = str(c.get('law_id', '')).strip()
        ten = str(c.get('ten_van_ban', '')).strip()
        dieu = str(c.get('dieu_so', '')).strip()
        if law_id and ten:
            d = f'{law_id}|{ten}'
            if d not in seen_docs:
                seen_docs.add(d)
                docs.append(d)
        if law_id and ten and dieu:
            a = f'{law_id}|{ten}|{dieu}'
            if a not in seen_articles:
                seen_articles.add(a)
                articles.append(a)
    return docs, articles

# end # __ADAPTIVE_K_PORT__ ADAPTER ---------------------------------------------------

# end # __ADAPTIVE_K_PORT__ SCORING-SELECTION ------------------------------------------------

def build_gen_contexts(hits, top_k=None):
    top_k = top_k or GEN_CONTEXT_TOPK
    out = []
    for h in hits[:top_k]:
        out.append({
            "law_id": getattr(h, "law_id", "") or "",
            "ten_van_ban": getattr(h, "ten_van_ban", "") or "",
            "dieu_so": getattr(h, "dieu_so", "") or "",
            "chunk_text": getattr(h, "chunk_text", "") or "",
        })
    return out


# --- resolve input / output paths ----------------------------------------
# Falls back to dev_set/questions.json when INPUT_QUERIES_PATH is blank, so
# this cell is runnable out of the box (same behaviour as the original §8).
_INPUT_Q = Path(INPUT_QUERIES_PATH) if str(INPUT_QUERIES_PATH).strip() else None
if _INPUT_Q is None or not _INPUT_Q.exists():
    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"
    _cand = ((DEV / "questions.json") if (DEV / "questions.json").exists()
             else (_DEV_LOCAL / "questions.json"))
    _INPUT_Q = _cand
    print(f"[batch] INPUT_QUERIES_PATH not set / missing -> using {_INPUT_Q}")
else:
    print(f"[batch] input  : {_INPUT_Q}")
print(f"[batch] results: {RESULTS_PATH}")
print(f"[batch] zip    : {SUBMISSION_ZIP_PATH}")

assert _INPUT_Q.exists(), f"input query file not found: {_INPUT_Q}"
Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)


# --- instantiate the generator (only when enabled) -----------------------
import torch
generator = None
if USE_GENERATOR:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    generator = QwenIRACGenerator()
else:
    print("[batch] USE_GENERATOR=False -> answer field left empty (retrieval-only).")

# --- instantiate the query planner (complexity assessment) ------------
# Always built; when USE_LLM_PLANNER=0 (or CPU w/o override) it self-
# disables and falls back to the rule-based planner, so plan() is safe.
planner = LLMQueryPlanner()


# --- progress bar for the batch loop -------------------------------------
# Wraps the query list with a tqdm bar when tqdm is available (it ships with
# Colab); falls back to a plain iterator otherwise so the cell runs unchanged.
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def _progress(iterable, total=None, desc="batch"):
    """Yield from *iterable* with a tqdm progress bar, or as-is if tqdm absent."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit="q",
                 dynamic_ncols=True, leave=True)


# --- batch run: decompose -> retrieve -> generate -> postprocess ---------
records_in = _json.loads(_INPUT_Q.read_text(encoding="utf-8"))
print(f"[batch] loaded {len(records_in)} queries")

records_out, t0 = [], time.time()
_batch_pbar = _progress(records_in, total=len(records_in), desc="batch")
for rec in _batch_pbar:
    qid   = rec.get("id")
    query = rec.get("question") or rec.get("query") or ""
    if not query:
        print(f"  [skip] id={qid} has no question text")
        continue

    hits = decomp_retriever.retrieve(query, fetch_text=True)
    query_plan = planner.plan(str(query))
    complexity = query_plan.get('complexity', 'medium')
    # # __ADAPTIVE_K_PORT__ LOOP-REWIRE  (adaptive-k selection flow) ---------------------------
    # Build per-variant article candidates from the fused Hit list + the
    # decomposed sub-query traces, then run the facet-coverage-aware selector.
    _variants = build_query_variants(str(query), query_plan)
    _traces = getattr(decomp_retriever, 'last_sub_query_traces', None) or []
    _candidates = build_article_candidates_from_hits(hits, query_plan, variants=_variants, sub_query_traces=_traces)
    _article_contexts, _sel_debug = select_article_contexts(_candidates, query_plan, variants=_variants)
    docs, articles = make_relevant_lists_from_articles(_article_contexts)
    contexts = build_gen_contexts_from_articles(_article_contexts)
    # Trim docs to those whose law_id+ten_van_ban match a kept article.
    _keep_doc_keys = {a.rsplit('|', 1)[0] for a in articles if '|' in a}
    docs = [d for d in docs if d in _keep_doc_keys]

    if generator is not None:
        try:
            answer = generator.generate(str(query), contexts, articles)
        except torch.cuda.OutOfMemoryError:
            print(f"  [oom] id={qid}; retrying with shorter contexts")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            short_ctx = []
            for c in contexts[:2]:
                c2 = dict(c)
                c2["chunk_text"] = str(c2["chunk_text"])[:600]
                short_ctx.append(c2)
            answer = generator.generate(str(query), short_ctx, articles[:8])
        answer = postprocess_answer_citations(answer, articles)
    else:
        answer = ""

    out = dict(rec)                       # preserve any extra input fields
    out["id"]                = qid
    out["question"]          = query
    out["answer"]            = answer
    out["relevant_docs"]     = docs
    out["relevant_articles"] = articles
    # # __ADAPTIVE_K_PORT__ RECORD-DEBUG  (optional selection diagnostics) ---------------------
    if DIAG_TOPK > 0:
        out["selection_debug"] = {
            "complexity": _sel_debug.get("complexity", complexity),
            "selected_k": int(_sel_debug.get("selected_k", len(articles))),
            "reason": _sel_debug.get("reason", ""),
            "covered_facets": list(_sel_debug.get("covered_facets", [])),
            "candidate_topk": [
                {k: c.get(k) for k in ("article_key", "canonical_article_key",
                 "article_score", "domain_score", "facet_score",
                 "matched_facets", "best_variant_kind", "support_variants")}
                for c in _candidates[:DIAG_TOPK]
            ],
        }
    records_out.append(out)
    print(f"  [done] id={qid}  c={complexity}  hits={len(hits)}  "
          f"docs={len(docs)}  articles={len(articles)}  ans_len={len(answer)}")

Path(RESULTS_PATH).write_text(_json.dumps(records_out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
print(f"\n[batch] wrote {len(records_out)} records to {RESULTS_PATH} in {time.time()-t0:.1f}s")


# --- (optional) F2 score if a ground-truth file is available -------------
try:
    _DEV_LOCAL = Path(REPO_DIR) / "dev_set"
    _GTPATH = ((DEV / "ground_truth.json") if (DEV / "ground_truth.json").exists()
               else (_DEV_LOCAL / "ground_truth.json"))
    if _GTPATH.exists():
        import sys as _sys
        if str(_DEV_LOCAL.parent) not in _sys.path:
            _sys.path.insert(0, str(_DEV_LOCAL.parent))
        from dev_set.eval import f2_macro
        _gt = _json.loads(_GTPATH.read_text(encoding="utf-8"))
        print(f"\nF2 macro (decomposition + IRAC generation, batch) = "
              f"{f2_macro(records_out, _gt):.4f}  [gt={_GTPATH}]")
    else:
        print("\n(ground_truth.json not found - skipping F2 score)")
except Exception as _e:
    print(f"\n(F2 scoring skipped: {_e})")


# release the query planner (frees its GPU model if loaded)
try:
    del planner
except NameError:
    pass
# release the generator before §9 validation / §10 cleanup
del generator
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()