# --- text normalisation + lexical tokeniser (ported from hybridrag-decomp) ---
# Required by the query-complexity planning machinery below.
_word_re = re.compile(r'\w+', re.UNICODE)
STOPWORDS = {
    'và', 'hoặc', 'của', 'các', 'những', 'một', 'này', 'đó', 'thì', 'là', 'có', 'bị', 'được',
    'phải', 'cho', 'về', 'trong', 'ngoài', 'theo', 'nếu', 'khi', 'như', 'để', 'với', 'từ', 'ra',
    'sao', 'gì', 'nào', 'bao', 'nhiêu', 'trường', 'hợp', 'cần', 'muốn', 'hỏi', 'tôi', 'công', 'ty'
}

def normalize_text(text):
    text = str(text).replace('\u200b', ' ').replace('\ufeff', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def normalize_key(text):
    return normalize_text(text).lower()

def tokenize_lexical(text, keep_stopwords=False):
    toks = _word_re.findall(normalize_text(text).lower())
    toks = [t for t in toks if len(t) >= 2]
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks

# --- LLM query decomposition + complexity planning -----------------------
# Ported from hybridrag-decomp.ipynb §2. Produces a query_plan whose
# 'complexity' (simple/medium/complex) drives the per-tier article quota via
# article_bounds_for_complexity(). LLM planning is optional; when disabled or
# on-CPU-without-override it falls back to the rule-based planner.
QUESTION_TYPES = {
    'procedure', 'condition', 'rights_obligations', 'sanction',
    'support_incentive', 'scenario', 'deadline', 'comparison',
    'definition_listing', 'other'
}
COMPLEXITIES = {'simple', 'medium', 'complex'}


def split_question_clauses_rule(question):
    q = normalize_text(question)
    pieces = re.split(
        r'[;?。]+|\s+(?:đồng thời|ngoài ra|bên cạnh đó|trong trường hợp|nếu|khi|và nếu|và phải|và cần|đặc biệt nếu)\s+',
        q,
        flags=re.IGNORECASE,
    )
    out = []
    for piece in pieces:
        piece = normalize_text(piece.strip(' ,.-:'))
        if 25 <= len(piece) <= 300:
            out.append(piece)
    if len(out) <= 1 and len(q) > 120:
        for piece in re.split(r',\s+| và ', q):
            piece = normalize_text(piece.strip(' ,.-:'))
            if 25 <= len(piece) <= 260:
                out.append(piece)
    seen, deduped = set(), []
    for piece in out:
        key = normalize_key(piece)
        if key not in seen and key != normalize_key(q):
            seen.add(key)
            deduped.append(piece)
    return deduped[:PLANNER_MAX_ATOMIC]


def detect_question_type(question):
    q = normalize_key(question)
    if re.search(r'\b(quy định|liệt kê|bao gồm|gồm|những|các)\b.*\b(nào|gì)\b', q) and not re.search(r'\b(thủ tục|hồ sơ|điều kiện|xử phạt|vi phạm|trách nhiệm|nghĩa vụ|ưu đãi|hỗ trợ)\b', q):
        return 'definition_listing'
    if re.search(r'\b(thủ tục|hồ sơ|tài liệu|chứng cứ|đơn yêu cầu|chuẩn bị)\b', q):
        return 'procedure'
    if re.search(r'\b(điều kiện|yêu cầu|tiêu chí|đáp ứng)\b', q):
        return 'condition'
    if re.search(r'\b(quyền|nghĩa vụ|trách nhiệm)\b', q):
        return 'rights_obligations'
    if re.search(r'\b(phạt|xử phạt|vi phạm|xử lý|khắc phục)\b', q):
        return 'sanction'
    if re.search(r'\b(hỗ trợ|ưu đãi|miễn|giảm)\b', q):
        return 'support_incentive'
    if re.search(r'\b(thời hạn|bao lâu|khi nào|mấy ngày)\b', q):
        return 'deadline'
    if re.search(r'\b(khác gì|khác biệt|so sánh)\b', q):
        return 'comparison'
    if len(tokenize_lexical(question, keep_stopwords=True)) >= 45:
        return 'scenario'
    return 'other'


def is_simple_listing_query(question):
    q = normalize_key(question)
    toks = tokenize_lexical(question, keep_stopwords=True)
    if len(toks) > 16:
        return False
    if re.search(r'\b(thủ tục|hồ sơ|điều kiện|xử phạt|vi phạm|khắc phục|trách nhiệm|nghĩa vụ|ưu đãi|hỗ trợ|đấu thầu|bồi thường)\b', q):
        return False
    listing_patterns = [
        r'\bquy định\s+(?:những|các)?\s*.+\s+nào\b',
        r'\b(?:những|các)\s+.+\s+nào\b',
        r'\b.+\s+bao gồm\s+(?:những|các)?\s+gì\b',
    ]
    return any(re.search(pat, q) for pat in listing_patterns)


def refine_query_plan_heuristics(question, plan):
    plan = dict(plan or {})
    if is_simple_listing_query(question):
        plan['complexity'] = 'simple'
        plan['atomic_questions'] = []
        plan['question_type'] = 'definition_listing'
        note = 'force_simple_listing'
        existing = normalize_text(plan.get('rationale_short', ''))
        plan['rationale_short'] = (existing + '; ' + note).strip('; ')[:260]
    return plan


def fallback_complexity(question, clauses=None):
    toks = tokenize_lexical(question, keep_stopwords=True)
    q = normalize_key(question)
    clauses = clauses if clauses is not None else split_question_clauses_rule(question)
    multi_markers = len(re.findall(r'\b(và|đồng thời|ngoài ra|nếu|khi|hồ sơ|chứng cứ|xử lý|khắc phục|nghĩa vụ|trách nhiệm|thiệt hại|giám định)\b', q))
    domain_markers = 0
    for pat in [
        r'doanh nghiệp nhỏ và vừa|dnnvv',
        r'đấu thầu|lựa chọn nhà thầu',
        r'quyền tác giả|sở hữu trí tuệ|phần mềm|sao chép',
        r'hải quan|nhập khẩu|xuất khẩu',
        r'giám định',
        r'người tiêu dùng|dễ bị tổn thương',
        r'thuế|đất đai|mặt bằng',
        r'lao động|hợp đồng lao động|bằng cấp|chứng chỉ',
    ]:
        if re.search(pat, q, flags=re.IGNORECASE):
            domain_markers += 1
    if len(toks) >= 50 or len(clauses) >= 3 or multi_markers >= 5:
        return 'complex'
    if domain_markers >= 3:
        return 'complex'
    if len(toks) >= 20 or len(clauses) >= 2 or multi_markers >= 2 or domain_markers >= 2:
        return 'medium'
    return 'simple'


def fallback_must_terms(question, max_terms=10):
    toks = tokenize_lexical(question, keep_stopwords=False)
    seen = []
    for tok in toks:
        if tok not in seen:
            seen.append(tok)
    return seen[:max_terms]


def fallback_query_plan(question, reason='rule_fallback'):
    clauses = split_question_clauses_rule(question)
    complexity = fallback_complexity(question, clauses)
    atomic = clauses if complexity != 'simple' else []
    if not atomic and complexity in {'medium', 'complex'}:
        atomic = [normalize_text(question)]
    plan = {
        'complexity': complexity,
        'atomic_questions': atomic[:CFG['planner_max_atomic']],
        'must_have_terms': fallback_must_terms(question),
        'question_type': detect_question_type(question),
        'rationale_short': reason,
        'planner_fallback': True,
        'planner_error': reason,
        'raw_plan_text': '',
    }
    return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))


def extract_json_object(text):
    text = str(text).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('No JSON object found in planner output')
    return json.loads(text[start:end + 1])


def token_overlap_ratio(candidate, original):
    cand = set(tokenize_lexical(candidate, keep_stopwords=False))
    orig = set(tokenize_lexical(original, keep_stopwords=False))
    if not cand or not orig:
        return 0.0
    return len(cand & orig) / max(len(cand), 1)


def has_forbidden_new_citation(text, original):
    original_has_citation = bool(re.search(r'\b(Điều\s+\d+|Luật\s+\d+|Nghị định\s+\d+|Thông tư\s+\d+|Nghị quyết\s+\d+|\d+/\d{4}/[A-ZĐ-]+)\b', original, flags=re.IGNORECASE))
    if original_has_citation:
        return False
    return bool(re.search(r'\b(Điều\s+\d+|\d+/\d{4}/[A-ZĐ-]+)\b', str(text), flags=re.IGNORECASE))


def sanitize_list_of_strings(values, max_items=8, max_chars=220):
    out, seen = [], set()
    if not isinstance(values, list):
        return out
    for value in values:
        text = normalize_text(str(value))[:max_chars].strip(' -')
        key = normalize_key(text)
        if text and key not in seen:
            out.append(text)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


# # __ADAPTIVE_K_PORT__ PLANNER-HELPERS  (adaptive-k port: anchor/domain/facet/variant) -----
import math
from collections import OrderedDict

def anchor_plain(text):
    import unicodedata
    text = unicodedata.normalize('NFD', normalize_text(text).lower())
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text.replace('\u0111', 'd')

def _append_unique(values, value):
    value = normalize_text(value)
    if value and value not in values:
        values.append(value)

def _plain_has_any(plain, needles):
    return any(needle in plain for needle in needles)


def build_rule_domain_profile(question, must_terms=None, raw_anchor_terms=None):
    must_terms = must_terms or []
    raw_anchor_terms = raw_anchor_terms or []
    haystack = ' '.join([question] + list(must_terms) + list(raw_anchor_terms))
    plain = anchor_plain(haystack)
    labels, anchors, preferred_law_ids = [], [], []
    preferred_title_terms, negative_title_terms, soft_negative_law_ids = [], [], []

    def add_label(label):
        if label not in labels:
            labels.append(label)

    def add_anchor(text):
        _append_unique(anchors, text)

    def add_pref_law(law_id):
        if law_id not in preferred_law_ids:
            preferred_law_ids.append(law_id)

    def add_pref_title(text):
        _append_unique(preferred_title_terms, text)

    def add_negative_title(text):
        _append_unique(negative_title_terms, text)

    def add_soft_negative_law(law_id):
        if law_id not in soft_negative_law_ids:
            soft_negative_law_ids.append(law_id)

    has_copyright = _plain_has_any(plain, [
        'quyen tac gia', 'quyen lien quan', 'phan mem', 'chuong trinh may tinh', 'sao chep',
        'xam pham quyen tac gia'
    ])
    has_industrial = _plain_has_any(plain, ['so huu cong nghiep', 'nhan hieu', 'sang che', 'kieu dang cong nghiep'])
    if has_copyright:
        add_label('copyright_software')
        for term in [
            'quyền tác giả', 'phần mềm', 'sao chép', 'cho thuê',
            'chương trình máy tính', 'xâm phạm quyền tác giả'
        ]:
            if anchor_plain(term) in plain:
                add_anchor(term)
        add_pref_law('50/2005/QH11')
        add_pref_law('17/2023/NĐ-CP')
        for term in ['quyền tác giả', 'quyền liên quan', 'sở hữu trí tuệ']:
            add_pref_title(term)
        if not has_industrial:
            add_negative_title('sở hữu công nghiệp')
            add_negative_title('giống cây trồng')
            add_soft_negative_law('65/2023/NĐ-CP')
            add_soft_negative_law('99/2013/NĐ-CP')
            add_soft_negative_law('11/2015/TT-BKHCN')

    if _plain_has_any(plain, ['hai quan', 'nhap khau', 'kiem soat']):
        add_label('customs_control')
        add_anchor('hải quan')
        add_anchor('kiểm soát')
        add_anchor('nhập khẩu')
        add_pref_law('13/2015/TT-BTC')
        add_pref_law('50/2005/QH11')
        add_pref_title('hải quan')
        add_pref_title('kiểm soát')

    if _plain_has_any(plain, ['giam dinh', 'giam dinh vien']):
        add_label('copyright_assessment')
        add_anchor('giám định')
        add_pref_law('15/2012/TT-BVHTTDL')
        add_pref_law('17/2023/NĐ-CP')
        add_pref_law('105/2006/NĐ-CP')
        add_pref_title('giám định quyền tác giả')
        add_negative_title('chuyển giao công nghệ')
        add_negative_title('thương mại')

    if _plain_has_any(plain, ['nguoi tieu dung', 'de bi ton thuong']):
        add_label('consumer_protection')
        add_anchor('người tiêu dùng')
        add_anchor('dễ bị tổn thương')
        add_pref_law('19/2023/QH15')
        add_pref_title('người tiêu dùng')

    if _plain_has_any(plain, ['doanh nghiep nho va vua', 'dnnvv']) and _plain_has_any(plain, ['dau thau', 'lua chon nha thau']):
        add_label('small_business_procurement')
        add_anchor('doanh nghiệp nhỏ và vừa')
        add_anchor('đấu thầu')
        add_pref_law('04/2017/QH14')
        add_pref_law('22/2023/QH15')
        add_pref_title('hỗ trợ doanh nghiệp nhỏ và vừa')
        add_pref_title('đấu thầu')

    for term in raw_anchor_terms:
        add_anchor(term)

    return {
        'labels': labels,
        'anchor_terms': anchors[:12],
        'preferred_law_ids': preferred_law_ids[:12],
        'preferred_title_terms': preferred_title_terms[:12],
        'negative_title_terms': negative_title_terms[:12],
        'soft_negative_law_ids': soft_negative_law_ids[:12],
    }

def is_generic_atomic_question(text, domain_profile=None):
    plain = anchor_plain(text)
    generic_markers = [
        'tai lieu', 'chung cu', 'ho so', 'don yeu cau', 'hop dong', 'tranh chap', 'xu ly'
    ]
    if not _plain_has_any(plain, generic_markers):
        return False
    domain_profile = domain_profile or {}
    anchors = domain_profile.get('anchor_terms', [])
    return not any(anchor_plain(anchor) in plain for anchor in anchors)

def repair_generic_atomic_question(text, domain_profile):
    text = normalize_text(text)
    if not is_generic_atomic_question(text, domain_profile):
        return text
    anchors = [a for a in domain_profile.get('anchor_terms', []) if normalize_text(a)]
    if not anchors:
        return text
    suffix = ' về ' + ', '.join(anchors[:3])
    if anchor_plain(suffix) in anchor_plain(text):
        return text
    if text.endswith('?'):
        return text[:-1].rstrip() + suffix + '?'
    return text + suffix

def lexical_jaccard(a, b):
    aa = set(tokenize_lexical(a, keep_stopwords=False))
    bb = set(tokenize_lexical(b, keep_stopwords=False))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(len(aa | bb), 1)

def atomic_questions_too_similar(items):
    if len(items) < 2:
        return False
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append(lexical_jaccard(items[i], items[j]))
    return bool(pairs) and max(pairs) >= 0.62

def dedupe_atomic_questions(items, max_items=None):
    out = []
    for item in items:
        item = normalize_text(item)
        if not item:
            continue
        if any(normalize_key(item) == normalize_key(x) or lexical_jaccard(item, x) >= 0.78 for x in out):
            continue
        out.append(item)
        if max_items and len(out) >= max_items:
            break
    return out

def facet_profile(
    facet_id,
    label,
    query,
    anchor_terms=None,
    preferred_law_ids=None,
    preferred_title_terms=None,
    target_terms=None,
    negative_terms=None,
    priority=1.0,
):
    return {
        'facet_id': str(facet_id),
        'label': normalize_text(label),
        'query': normalize_text(query),
        'anchor_terms': [normalize_text(x) for x in (anchor_terms or []) if normalize_text(x)],
        'preferred_law_ids': [str(x).strip() for x in (preferred_law_ids or []) if str(x).strip()],
        'preferred_title_terms': [normalize_text(x) for x in (preferred_title_terms or []) if normalize_text(x)],
        'target_terms': [normalize_text(x) for x in (target_terms or []) if normalize_text(x)],
        'negative_terms': [normalize_text(x) for x in (negative_terms or []) if normalize_text(x)],
        'priority': float(priority),
    }

def sanitize_facet_profiles(values, max_items=None):
    max_items = max_items or max(6, CFG.get('planner_max_atomic', 4))
    out, seen = [], set()
    if not isinstance(values, list):
        return out
    for raw in values:
        if not isinstance(raw, dict):
            continue
        facet_id = normalize_key(str(raw.get('facet_id') or raw.get('label') or raw.get('query') or 'facet'))
        facet_id = re.sub(r'[^a-z0-9_]+', '_', facet_id).strip('_')[:64] or 'facet'
        query = normalize_text(raw.get('query') or raw.get('text') or raw.get('label') or '')
        label = normalize_text(raw.get('label') or query or facet_id)
        key = facet_id + '|' + normalize_key(query or label)
        if not query or key in seen:
            continue
        out.append(facet_profile(
            facet_id=facet_id,
            label=label,
            query=query,
            anchor_terms=sanitize_list_of_strings(raw.get('anchor_terms', []), max_items=8, max_chars=80),
            preferred_law_ids=sanitize_list_of_strings(raw.get('preferred_law_ids', []), max_items=8, max_chars=40),
            preferred_title_terms=sanitize_list_of_strings(raw.get('preferred_title_terms', []), max_items=8, max_chars=80),
            target_terms=sanitize_list_of_strings(raw.get('target_terms', []), max_items=12, max_chars=80),
            negative_terms=sanitize_list_of_strings(raw.get('negative_terms', []), max_items=8, max_chars=80),
            priority=float(raw.get('priority', 1.0) or 1.0),
        ))
        seen.add(key)
        if len(out) >= max_items:
            break
    return out

def build_rule_facet_profiles(question, domain_profile):
    plain = anchor_plain(question)
    labels = set(domain_profile.get('labels', []))
    facets = []

    def add(facet):
        key = facet['facet_id']
        if key not in {x['facet_id'] for x in facets}:
            facets.append(facet)

    if 'small_business_procurement' in labels:
        add(facet_profile(
            'small_business_support',
            'Ưu đãi hỗ trợ doanh nghiệp nhỏ và vừa',
            'Ưu đãi, hỗ trợ dành cho doanh nghiệp nhỏ và vừa theo pháp luật hỗ trợ doanh nghiệp nhỏ và vừa?',
            anchor_terms=['doanh nghiệp nhỏ và vừa', 'ưu đãi', 'hỗ trợ'],
            preferred_law_ids=['04/2017/QH14'],
            preferred_title_terms=['hỗ trợ doanh nghiệp nhỏ và vừa'],
            target_terms=['doanh nghiệp nhỏ và vừa', 'hỗ trợ', 'ưu đãi', 'chính sách hỗ trợ'],
            negative_terms=['chuyển giao công nghệ', 'thương mại'],
            priority=1.15,
        ))
        add(facet_profile(
            'procurement_preference',
            'Ưu đãi trong đấu thầu',
            'Ưu đãi đối với doanh nghiệp nhỏ và vừa trong lựa chọn nhà thầu, đấu thầu?',
            anchor_terms=['doanh nghiệp nhỏ và vừa', 'đấu thầu', 'lựa chọn nhà thầu'],
            preferred_law_ids=['22/2023/QH15'],
            preferred_title_terms=['đấu thầu'],
            target_terms=['ưu đãi', 'đấu thầu', 'nhà thầu', 'lựa chọn nhà thầu', 'doanh nghiệp nhỏ và vừa'],
            negative_terms=['chuyển giao công nghệ', 'thương mại'],
            priority=1.14,
        ))

    if 'copyright_software' in labels:
        multi_domain = bool(labels & {'customs_control', 'copyright_assessment', 'consumer_protection'})
        explicit_software = _plain_has_any(plain, ['phan mem', 'chuong trinh may tinh', 'sao chep', 'cho thue'])
        if (not multi_domain) or explicit_software:
            add(facet_profile(
                'software_property_rights',
                'Quyền tài sản với chương trình máy tính',
                'Quyền tác giả và quyền tài sản đối với chương trình máy tính, phần mềm được quy định thế nào?',
                anchor_terms=['quyền tác giả', 'phần mềm', 'chương trình máy tính'],
                preferred_law_ids=['50/2005/QH11'],
                preferred_title_terms=['sở hữu trí tuệ', 'quyền tác giả'],
                target_terms=['quyền tài sản', 'chương trình máy tính', 'phần mềm', 'sao chép', 'cho thuê'],
                negative_terms=['sở hữu công nghiệp', 'nhãn hiệu', 'sáng chế'],
                priority=1.12,
            ))
            if _plain_has_any(plain, ['sao chep', 'cho thue', 'xam pham']):
                add(facet_profile(
                    'copyright_infringement',
                    'Hành vi xâm phạm quyền tác giả phần mềm',
                    'Hành vi sao chép, cho thuê trái phép phần mềm xâm phạm quyền tác giả như thế nào?',
                    anchor_terms=['quyền tác giả', 'phần mềm', 'sao chép', 'cho thuê'],
                    preferred_law_ids=['50/2005/QH11'],
                    preferred_title_terms=['sở hữu trí tuệ', 'quyền tác giả'],
                    target_terms=['xâm phạm', 'sao chép', 'cho thuê', 'quyền tác giả', 'chương trình máy tính'],
                    negative_terms=['sở hữu công nghiệp', 'nhãn hiệu', 'sáng chế'],
                    priority=1.11,
                ))
        if _plain_has_any(plain, ['ton that', 'mat khach hang', 'co hoi kinh doanh', 'thiet hai', 'boi thuong']):
            add(facet_profile(
                'damage_business_opportunity',
                'Thiệt hại và cơ hội kinh doanh',
                'Cách xác định thiệt hại và tổn thất cơ hội kinh doanh do xâm phạm quyền tác giả phần mềm?',
                anchor_terms=['quyền tác giả', 'phần mềm', 'thiệt hại', 'cơ hội kinh doanh'],
                preferred_law_ids=['50/2005/QH11', '17/2023/NĐ-CP'],
                preferred_title_terms=['sở hữu trí tuệ', 'quyền tác giả'],
                target_terms=['thiệt hại', 'tổn thất', 'cơ hội kinh doanh', 'bồi thường', 'mất khách hàng'],
                negative_terms=['sở hữu công nghiệp', 'nhãn hiệu', 'sáng chế'],
                priority=1.10,
            ))
        if _plain_has_any(plain, ['tu bao ve', 'yeu cau xu ly', 'bien phap bao ve', 'xu ly xam pham']):
            add(facet_profile(
                'self_protection_request',
                'Quyền tự bảo vệ và yêu cầu xử lý',
                'Quyền tự bảo vệ, yêu cầu xử lý xâm phạm và biện pháp bảo vệ quyền sở hữu trí tuệ được quy định thế nào?',
                anchor_terms=['quyền tác giả', 'xâm phạm', 'yêu cầu xử lý'],
                preferred_law_ids=['50/2005/QH11'],
                preferred_title_terms=['sở hữu trí tuệ'],
                target_terms=['quyền tự bảo vệ', 'yêu cầu xử lý', 'biện pháp bảo vệ', 'xâm phạm quyền sở hữu trí tuệ'],
                negative_terms=['sở hữu công nghiệp', 'nhãn hiệu', 'sáng chế'],
                priority=1.08,
            ))
        if _plain_has_any(plain, ['tai lieu', 'chung cu', 'don yeu cau', 'xu ly']):
            add(facet_profile(
                'evidence_request_docs',
                'Tài liệu chứng cứ yêu cầu xử lý',
                'Tài liệu, chứng cứ cần chuẩn bị khi yêu cầu xử lý hành vi xâm phạm quyền tác giả đối với phần mềm?',
                anchor_terms=['quyền tác giả', 'phần mềm', 'tài liệu', 'chứng cứ'],
                preferred_law_ids=['17/2023/NĐ-CP', '50/2005/QH11'],
                preferred_title_terms=['quyền tác giả', 'sở hữu trí tuệ'],
                target_terms=['tài liệu', 'chứng cứ', 'đơn yêu cầu', 'yêu cầu xử lý', 'xâm phạm quyền tác giả'],
                negative_terms=['sở hữu công nghiệp', 'nhãn hiệu', 'sáng chế'],
                priority=1.06,
            ))

    if 'customs_control' in labels:
        add(facet_profile(
            'customs_financial_guarantee',
            'Kiểm soát hải quan và bảo đảm tài chính',
            'Nghĩa vụ bảo đảm tài chính khi yêu cầu hải quan kiểm soát hàng hóa nghi xâm phạm quyền tác giả?',
            anchor_terms=['hải quan', 'kiểm soát', 'nhập khẩu', 'bảo đảm tài chính'],
            preferred_law_ids=['13/2015/TT-BTC', '50/2005/QH11'],
            preferred_title_terms=['hải quan', 'sở hữu trí tuệ'],
            target_terms=['hải quan', 'kiểm soát', 'hàng hóa', 'bảo đảm tài chính', 'tạm dừng làm thủ tục'],
            negative_terms=['chuyển giao công nghệ', 'thương mại'],
            priority=1.16,
        ))
    if 'copyright_assessment' in labels:
        add(facet_profile(
            'assessment_contract',
            'Hợp đồng giám định quyền tác giả',
            'Hợp đồng giám định quyền tác giả, quyền liên quan cần có những nội dung chính nào?',
            anchor_terms=['giám định', 'hợp đồng giám định', 'quyền tác giả'],
            preferred_law_ids=['15/2012/TT-BVHTTDL', '17/2023/NĐ-CP', '105/2006/NĐ-CP'],
            preferred_title_terms=['giám định quyền tác giả', 'quyền tác giả'],
            target_terms=['giám định', 'hợp đồng giám định', 'giám định viên', 'nội dung hợp đồng'],
            negative_terms=['chuyển giao công nghệ', 'thương mại'],
            priority=1.15,
        ))
    if 'consumer_protection' in labels:
        add(facet_profile(
            'consumer_dispute',
            'Tranh chấp với người tiêu dùng dễ bị tổn thương',
            'Trách nhiệm giải quyết tranh chấp khi đối tượng bị xâm phạm là người tiêu dùng dễ bị tổn thương?',
            anchor_terms=['người tiêu dùng', 'dễ bị tổn thương', 'tranh chấp'],
            preferred_law_ids=['19/2023/QH15'],
            preferred_title_terms=['người tiêu dùng'],
            target_terms=['người tiêu dùng', 'dễ bị tổn thương', 'tranh chấp', 'trách nhiệm'],
            negative_terms=['chuyển giao công nghệ', 'thương mại'],
            priority=1.14,
        ))

    limit = max(6, CFG.get('planner_max_atomic', 4))
    return sanitize_facet_profiles(facets, max_items=limit)

def build_rule_legal_facets(question, domain_profile):
    return dedupe_atomic_questions(
        [f.get('query', '') for f in build_rule_facet_profiles(question, domain_profile)],
        max_items=max(6, CFG.get('planner_max_atomic', 4)),
    )

def merge_facet_profiles(rule_facets, raw_facets):
    out, seen = [], set()
    for facet in list(rule_facets or []) + list(raw_facets or []):
        if not isinstance(facet, dict):
            continue
        key = facet.get('facet_id', '') + '|' + normalize_key(facet.get('query', ''))
        if not key.strip('|') or key in seen:
            continue
        out.append(facet)
        seen.add(key)
    return out[:max(6, CFG.get('planner_max_atomic', 4))]

def enrich_query_plan(question, plan):
    plan = dict(plan or {})
    must_terms = sanitize_list_of_strings(plan.get('must_have_terms', []), max_items=12, max_chars=80)
    raw_anchors = sanitize_list_of_strings(plan.get('anchor_terms', []), max_items=12, max_chars=80)
    raw_facets = sanitize_facet_profiles(plan.get('facet_profiles', []), max_items=max(6, CFG.get('planner_max_atomic', 4)))
    domain_profile = build_rule_domain_profile(question, must_terms=must_terms, raw_anchor_terms=raw_anchors)
    complexity = plan.get('complexity', 'medium')
    rule_facet_profiles = build_rule_facet_profiles(question, domain_profile)
    facet_profiles = merge_facet_profiles(rule_facet_profiles, raw_facets)
    rule_facets = dedupe_atomic_questions(
        [f.get('query', '') for f in facet_profiles] + sanitize_list_of_strings(plan.get('legal_facets', []), max_items=6, max_chars=220),
        max_items=max(6, CFG.get('planner_max_atomic', 4)),
    )

    atomic = sanitize_list_of_strings(plan.get('atomic_questions', []), max_items=CFG['planner_max_atomic'], max_chars=240)
    if complexity == 'simple':
        atomic = []
    else:
        atomic = [repair_generic_atomic_question(item, domain_profile) for item in atomic]
        labels = set(domain_profile.get('labels', []))
        force_rule_first = (
            complexity == 'complex'
            or 'small_business_procurement' in labels
            or atomic_questions_too_similar(atomic)
            or len(rule_facet_profiles) >= 2
        )
        if rule_facets and force_rule_first:
            atomic = dedupe_atomic_questions(rule_facets + atomic, max_items=CFG['planner_max_atomic'])
        elif rule_facets and len(atomic) < min(2, CFG['planner_max_atomic']):
            atomic = dedupe_atomic_questions(atomic + rule_facets, max_items=CFG['planner_max_atomic'])
        else:
            atomic = dedupe_atomic_questions(atomic, max_items=CFG['planner_max_atomic'])
        if not atomic and complexity in {'medium', 'complex'}:
            atomic = dedupe_atomic_questions(rule_facets or [normalize_text(question)], max_items=CFG['planner_max_atomic'])

    plan['atomic_questions'] = atomic[:CFG['planner_max_atomic']]
    plan['must_have_terms'] = must_terms or fallback_must_terms_inline(question)
    plan['anchor_terms'] = domain_profile.get('anchor_terms', [])
    plan['legal_facets'] = rule_facets
    plan['facet_profiles'] = facet_profiles
    plan['domain_profile'] = domain_profile
    plan['generic_atomic_questions'] = [a for a in atomic if is_generic_atomic_question(a, domain_profile)]
    return plan

def fallback_must_terms_inline(question, max_terms=10):
    toks = tokenize_lexical(question, keep_stopwords=False)
    seen = []
    for tok in toks:
        if tok not in seen:
            seen.append(tok)
    return seen[:max_terms]


SLIDE_REFERENCE_NOTES = {
    1577: {
        'label': 'Dễ',
        'expected': ['16/2012/QH13|Luật Quảng cáo|Điều 17'],
        'note': '1 văn bản, 1 điều; không cần phân rã, không multi-hop.',
    },
    2: {
        'label': 'Trung bình',
        'expected': ['04/2017/QH14|Luật Hỗ trợ DNNVV|Điều 13', '22/2023/QH15|Luật Đấu thầu|Điều 10'],
        'note': 'Cross-doc 2 văn bản: ưu đãi DNNVV + ưu đãi trong đấu thầu.',
    },
    1128: {
        'label': 'Khó',
        'expected': ['Luật SHTT Điều 20, 22, 28, 198, 204, 205', 'NĐ 17/2023 Điều 66, 73, 75, 76'],
        'note': '3 yêu cầu: hành vi xâm phạm, tổn thất/cơ hội kinh doanh, tài liệu/chứng cứ xử lý.',
    },
    1720: {
        'label': 'Tình huống thực tế',
        'expected': ['SHTT', 'kiểm soát hải quan', 'giám định', 'bảo vệ người tiêu dùng'],
        'note': '4 vế đan xen nhiều lĩnh vực; 3 văn bản, 4 điều.',
    },
}

SMOKE_EXPECTED_ARTICLES = {
    1577: [('16/2012/QH13', 'Điều 17')],
    2: [('04/2017/QH14', 'Điều 13'), ('22/2023/QH15', 'Điều 10')],
    1128: [
        ('50/2005/QH11', 'Điều 20'), ('50/2005/QH11', 'Điều 22'), ('50/2005/QH11', 'Điều 28'),
        ('50/2005/QH11', 'Điều 198'), ('50/2005/QH11', 'Điều 204'), ('50/2005/QH11', 'Điều 205'),
        ('17/2023/NĐ-CP', 'Điều 66'), ('17/2023/NĐ-CP', 'Điều 73'),
        ('17/2023/NĐ-CP', 'Điều 75'), ('17/2023/NĐ-CP', 'Điều 76'),
    ],
}

SMOKE_EXPECTED_DOMAINS = {
    1720: {
        'shtt': ['50/2005/QH11', '17/2023/NĐ-CP'],
        'hai_quan': ['13/2015/TT-BTC', 'hải quan', 'kiểm soát'],
        'giam_dinh': ['15/2012/TT-BVHTTDL', 'giám định'],
        'nguoi_tieu_dung': ['19/2023/QH15', 'người tiêu dùng'],
    },
}


def add_variant(variants, text, kind, weight, dense_only=False, lexical_only=False):
    text = normalize_text(text)
    if not text:
        return
    key = normalize_key(text)
    if key in {v['key'] for v in variants}:
        return
    variants.append({
        'text': text,
        'kind': kind,
        'weight': float(weight),
        'dense_only': bool(dense_only),
        'lexical_only': bool(lexical_only),
        'key': key,
    })

def must_terms_query(query_plan):
    terms = [normalize_text(t) for t in query_plan.get('must_have_terms', []) if normalize_text(t)]
    return ' '.join(terms[:16])

def build_query_variants(question, query_plan=None):
    query_plan = query_plan or fallback_query_plan(question)
    complexity = query_plan.get('complexity', 'medium')
    variants = []
    add_variant(variants, question, 'original', 1.00)

    if complexity == 'simple':
        atomic_weights = [0.86]
        facet_weight = 0.00
        must_weight = 0.58
    elif complexity == 'complex':
        atomic_weights = [0.96, 0.93, 0.90, 0.87]
        facet_weight = 0.88
        must_weight = 0.68
    else:
        atomic_weights = [0.94, 0.90, 0.86]
        facet_weight = 0.91
        must_weight = 0.64

    for idx, atomic in enumerate(query_plan.get('atomic_questions', [])[:CFG['planner_max_atomic']]):
        weight = atomic_weights[min(idx, len(atomic_weights) - 1)]
        add_variant(variants, atomic, f'atomic_{idx + 1}', weight)

    if complexity != 'simple':
        for idx, facet in enumerate(query_plan.get('facet_profiles', [])[:max(6, CFG.get('planner_max_atomic', 4))]):
            text = facet.get('query') or facet.get('label') or ''
            facet_id = re.sub(r'[^a-z0-9_]+', '_', normalize_key(facet.get('facet_id', f'facet_{idx + 1}'))).strip('_') or f'facet_{idx + 1}'
            priority = float(facet.get('priority', 1.0) or 1.0)
            weight = min(0.97, facet_weight + 0.02 * max(priority - 1.0, 0.0))
            add_variant(variants, text, f'facet_{idx + 1}_{facet_id}', weight)

    if CFG.get('use_must_terms_variant', True):
        term_query = must_terms_query(query_plan)
        if term_query and normalize_key(term_query) != normalize_key(question):
            add_variant(variants, term_query, 'must_terms', must_weight, lexical_only=True)

    return variants


# end # __ADAPTIVE_K_PORT__ PLANNER-HELPERS ------------------------------------------------

def validate_query_plan(raw_plan, question, raw_text=''):
    fallback = fallback_query_plan(question)
    if not isinstance(raw_plan, dict):
        fallback['raw_plan_text'] = raw_text
        fallback['planner_error'] = 'planner returned non-dict'
        return fallback

    complexity = str(raw_plan.get('complexity', '')).strip().lower()
    if complexity not in COMPLEXITIES:
        complexity = fallback['complexity']

    qtype = str(raw_plan.get('question_type', '')).strip().lower()
    if qtype not in QUESTION_TYPES:
        qtype = fallback['question_type']

    atomic = sanitize_list_of_strings(raw_plan.get('atomic_questions', []), max_items=PLANNER_MAX_ATOMIC)
    clean_atomic = []
    for item in atomic:
        if normalize_key(item) == normalize_key(question):
            continue
        if has_forbidden_new_citation(item, question):
            continue
        if token_overlap_ratio(item, question) < PLANNER_MIN_OVERLAP:
            continue
        clean_atomic.append(item)

    if not clean_atomic and complexity in {'medium', 'complex'}:
        clean_atomic = fallback['atomic_questions']
    clean_atomic = clean_atomic[:PLANNER_MAX_ATOMIC]

    terms = sanitize_list_of_strings(raw_plan.get('must_have_terms', []), max_items=12, max_chars=60)
    terms = [t for t in terms if not has_forbidden_new_citation(t, question)]
    if not terms:
        terms = fallback['must_have_terms']

    # # __ADAPTIVE_K_PORT__ VQP-UPGRADE  (emit anchor_terms/legal_facets/facet_profiles/domain_profile)
    plan = {
        'complexity': complexity,
        'atomic_questions': clean_atomic,
        'must_have_terms': terms,
        'anchor_terms': sanitize_list_of_strings(raw_plan.get('anchor_terms', []), max_items=12, max_chars=80),
        'legal_facets': sanitize_list_of_strings(raw_plan.get('legal_facets', []), max_items=max(6, CFG['planner_max_atomic']), max_chars=220),
        'facet_profiles': sanitize_facet_profiles(raw_plan.get('facet_profiles', []), max_items=max(6, CFG['planner_max_atomic'])),
        'domain_profile': raw_plan.get('domain_profile', {}) if isinstance(raw_plan.get('domain_profile', {}), dict) else {},
        'question_type': qtype,
        'rationale_short': normalize_text(raw_plan.get('rationale_short', ''))[:260],
        'planner_fallback': False,
        'planner_error': '',
        'raw_plan_text': raw_text,
    }
    return enrich_query_plan(question, refine_query_plan_heuristics(question, plan))


class LLMQueryPlanner:
    def __init__(self, model_name=PLANNER_MODEL):
        self.enabled = bool(USE_LLM_PLANNER)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.disabled_reason = ''
        if not self.enabled:
            self.disabled_reason = 'USE_LLM_PLANNER=0'
            print({'planner': 'disabled', 'fallback': True})
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda' and not PLANNER_ALLOW_CPU:
            self.enabled = False
            self.disabled_reason = 'planner_cpu_fallback'
            print({
                'planner': 'cpu_fallback',
                'fallback': True,
                'reason': 'CUDA is not available; refusing to load planner LLM on CPU. Set PLANNER_ALLOW_CPU=1 to override.',
                'planner_model': model_name,
            })
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs = {'trust_remote_code': True}
        if device == 'cuda' and PLANNER_LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs.update({
                'device_map': 'auto',
                'torch_dtype': torch.float16,
                'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                ),
            })
        elif device == 'cuda':
            load_kwargs.update({'torch_dtype': torch.float16, 'device_map': 'auto'})
        else:
            load_kwargs.update({'torch_dtype': torch.float32})
        print({'planner_model': model_name, 'planner_device': device, 'planner_4bit': bool(device == 'cuda' and PLANNER_LOAD_IN_4BIT)})
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def build_prompt(self, question):
        system = (
            'Bạn là bộ phân tích truy vấn cho hệ thống truy hồi văn bản pháp luật Việt Nam. '
            'Chỉ phân rã ý hỏi, không trả lời câu hỏi, không suy đoán số điều, không bịa tên văn bản. '
            'Luôn trả về đúng một JSON object hợp lệ.'
        )
        user = f'''
Câu hỏi:
{question}

Hãy trả về JSON theo schema:
{{
  "complexity": "simple|medium|complex",
  "atomic_questions": ["mệnh đề truy hồi độc lập, tối đa 4"],
  "must_have_terms": ["thuật ngữ bắt buộc lấy từ hoặc bám rất sát câu hỏi"],
  "anchor_terms": ["domain anchors that every atomic question must preserve"],
  "legal_facets": ["separate legal retrieval facets, not paraphrases"],
  "facet_profiles": [{{"facet_id": "stable_id", "label": "legal facet", "anchor_terms": [], "preferred_law_ids": [], "preferred_title_terms": [], "target_terms": [], "negative_terms": [], "priority": 1.0}}],
  "domain_profile": {{"labels": ["domain labels if clear"]}},
  "question_type": "procedure|condition|rights_obligations|sanction|support_incentive|scenario|deadline|comparison|definition_listing|other",
  "rationale_short": "lý do ngắn, không quá 1 câu"
}}

Quy tắc:
- Every atomic question must preserve specific domain anchors from the original question, such as copyright, software, customs, assessment, vulnerable consumers, small and medium enterprises, or procurement.
- Do not shorten an atomic question into a generic form like documents/evidence, contract contents, dispute handling, or request processing when the original question contains a specific domain.
- legal_facets should split legal aspects, not paraphrase the same question multiple times.
- facet_profiles should name the legal facets that need separate slot coverage; keep law/title preferences only when directly implied by the question.
- simple: 1 vấn đề, thường 1 văn bản/1 điều.
- medium: 2 ý hoặc cross-doc nhẹ.
- complex: nhiều mệnh đề, multi-hop, nhiều lĩnh vực/văn bản.
- atomic_questions phải giữ đúng ý gốc, không thêm vấn đề mới.
- Không nêu "Điều X", mã luật, tên văn bản cụ thể nếu câu hỏi không nêu.
- Không dùng markdown, không giải thích ngoài JSON.
'''
        # # __ADAPTIVE_K_PORT__ PROMPT-UPGRADE
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ], tokenize=False, add_generation_prompt=True)
        return system + '\n\n' + user + '\n\nJSON:'

    def plan(self, question):
        question = normalize_text(question)
        if not self.enabled:
            return fallback_query_plan(question, reason=self.disabled_reason or 'planner_disabled')
        try:
            prompt = self.build_prompt(question)
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=PLANNER_MAX_NEW_TOKENS,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.01,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inputs['input_ids'].shape[1]:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            raw_plan = extract_json_object(raw_text)
            return validate_query_plan(raw_plan, question, raw_text=raw_text)
        except Exception as e:
            plan = fallback_query_plan(question, reason='planner_exception')
            plan['planner_error'] = repr(e)
            return plan


def article_bounds_for_complexity(complexity):
    """Per-tier article quota: {min_k, target_k, max_k}.

    simple  -> 1 article,   complex -> up to 12,  medium -> up to 4.
    Each max_k is capped by SUBMIT_ARTICLE_MAX (the hard ceiling).
    """
    if complexity == 'simple':
        return {'min_k': 1, 'target_k': 1, 'max_k': min(SUBMIT_ARTICLE_MAX_SIMPLE, SUBMIT_ARTICLE_MAX)}
    if complexity == 'complex':
        return {'min_k': 4, 'target_k': 10, 'max_k': min(SUBMIT_ARTICLE_MAX_COMPLEX, SUBMIT_ARTICLE_MAX)}
    return {'min_k': 2, 'target_k': 4, 'max_k': min(SUBMIT_ARTICLE_MAX_MEDIUM, SUBMIT_ARTICLE_MAX)}


# --- IRAC answer generator (ported from hybridrag-decomp.ipynb) -----------
# Builds an IRAC-structured prompt from the retrieved Hit contexts + the
# allowed citation list, generates with the Qwen causal LM, and strips any
# prompt echo. Honours GEN_LOAD_IN_4BIT for memory-constrained GPUs.

class QwenIRACGenerator:
    def __init__(self, model_name=GEN_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs = {"trust_remote_code": True}
        if device == "cuda" and GEN_LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
            })
        elif device == "cuda":
            load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
        else:
            load_kwargs.update({"torch_dtype": torch.float32})
        print({"generator_device": device,
               "load_in_4bit": bool(device == "cuda" and GEN_LOAD_IN_4BIT),
               "gen_context_topk": GEN_CONTEXT_TOPK,
               "gen_chunk_char_limit": GEN_CHUNK_CHAR_LIMIT,
               "gen_max_input_tokens": GEN_MAX_INPUT_TOKENS})
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.device = device

    def build_prompt(self, question, contexts, relevant_articles):
        ctx_blocks = []
        for i, c in enumerate(contexts, start=1):
            meta = " | ".join(str(c.get(k, "")) for k in ["law_id", "ten_van_ban", "dieu_so"])
            text = str(c.get("chunk_text", ""))[:GEN_CHUNK_CHAR_LIMIT]
            ctx_blocks.append(f"[{i}] {meta}\n{text}")
        allowed = "\n".join("- " + str(a) for a in relevant_articles[:ARTICLE_CONTEXT_TOPK])
        system = (
            "Bạn là trợ lý pháp lý AI cho doanh nghiệp SME tại Việt Nam. "
            "Chỉ trả lời dựa trên ngữ cảnh và danh sách căn cứ được cung cấp. "
            "Không bịa văn bản, điều luật, khoản hoặc nguồn tham chiếu. "
            "Nếu thiếu căn cứ, nói rõ là chưa đủ căn cứ."
        )
        user = (
            f"Câu hỏi: {question}\n\n"
            "Danh sách điều luật được phép viện dẫn:\n"
            f"{allowed}\n\n"
            "Ngữ cảnh pháp lý:\n"
            f"{chr(10).join(ctx_blocks)}\n\n"
            "Yêu cầu trả lời:\n"
            "- Trả lời bằng tiếng Việt, ngắn gọn nhưng đủ ý.\n"
            "- Dùng cấu trúc IRAC ngắn: Vấn đề, Quy định, Áp dụng, Kết luận.\n"
            "- Luôn nêu Điều X khi có căn cứ trong danh sách được phép viện dẫn.\n"
            "- Không nhắc lại toàn bộ ngữ cảnh, không trích dẫn điều ngoài danh sách.\n"
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], tokenize=False, add_generation_prompt=True)
        return system + "\n\n" + user + "\n\nTrả lời:"

    def generate(self, question, contexts, relevant_articles):
        import torch
        prompt = self.build_prompt(question, contexts, relevant_articles)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=GEN_MAX_INPUT_TOKENS).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                do_sample=GEN_TEMPERATURE > 0,
                repetition_penalty=1.02,
                use_cache=GEN_USE_CACHE,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return strip_leading_prompt_echo(raw_text)

print("[generator] QwenIRACGenerator class ready (instantiated lazily below)")