"""Idempotent patcher for retrieval_colab_decomp_submit.ipynb.

Ports the query-complexity planning machinery (SIMPLE / MEDIUM / COMPLEX) from
hybridrag-decomp.ipynb §2 so that the §8 batch loop varies the article quota by
complexity tier:

  * cell 20  -> add planner config knobs (USE_LLM_PLANNER, PLANNER_MODEL, ...)
  * cell 22  -> insert text helpers + full planning machinery + LLMQueryPlanner
                + article_bounds_for_complexity() BEFORE QwenIRACGenerator
  * cell 23  -> instantiate planner, plan each query, truncate articles/docs to
                the tier-specific max_k, log the complexity
  * cell 27  -> release 'planner' in the cleanup tuple

The patcher is idempotent: re-running it is a no-op once the markers are present.
Notebook JSON is written with indent=1 + trailing newline (team convention).
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "retrieval_colab_decomp_submit.ipynb"


def _src(cell):
    return cell["source"]


def _joined(cell):
    return "".join(cell["source"])


def _apply_cfg_constant(*src):
    """Convert CFG['foo'] references to module-level constants FOO.

    Used when porting hybridrag code that reads the CFG dict, into the submit
    notebook which uses flat UPPER_SNAKE constants (matching the existing
    QwenIRACGenerator port that already replaced CFG['gen_*']).
    """
    out = []
    for line in src:
        for key, const in CFG_TO_CONST:
            line = line.replace(f"CFG['{key}']", const).replace(f'CFG["{key}"]', const)
        line = line.replace("CFG.get('use_llm_planner', True)", "USE_LLM_PLANNER")
        line = line.replace('CFG.get("use_llm_planner", True)', "USE_LLM_PLANNER")
        line = line.replace("CFG.get('planner_allow_cpu', False)", "PLANNER_ALLOW_CPU")
        line = line.replace('CFG.get("planner_allow_cpu", False)', "PLANNER_ALLOW_CPU")
        line = line.replace("CFG.get('planner_load_in_4bit', True)", "PLANNER_LOAD_IN_4BIT")
        line = line.replace('CFG.get("planner_load_in_4bit", True)', "PLANNER_LOAD_IN_4BIT")
        out.append(line)
    return out


# mapping of CFG key -> flat constant name used in the submit notebook
CFG_TO_CONST = [
    ("planner_max_atomic", "PLANNER_MAX_ATOMIC"),
    ("planner_min_overlap", "PLANNER_MIN_OVERLAP"),
    ("planner_max_new_tokens", "PLANNER_MAX_NEW_TOKENS"),
    ("submit_article_max_simple", "SUBMIT_ARTICLE_MAX_SIMPLE"),
    ("submit_article_max_medium", "SUBMIT_ARTICLE_MAX_MEDIUM"),
    ("submit_article_max_complex", "SUBMIT_ARTICLE_MAX_COMPLEX"),
    ("submit_article_max", "SUBMIT_ARTICLE_MAX"),
]


# ---------------------------------------------------------------------------
# Block A — planner config knobs (cell 20)
# ---------------------------------------------------------------------------
CONFIG_LINES = [
    "\n",
    "# --- Query-planning config (ported from hybridrag-decomp.ipynb §2) -----------\n",
    "# Drives SIMPLE/MEDIUM/COMPLEX complexity assessment and the per-tier article\n",
    "# quota via article_bounds_for_complexity(). Set USE_LLM_PLANNER=0 to use only\n",
    "# the rule-based fallback (no extra GPU model loaded).\n",
    "USE_LLM_PLANNER          = os.environ.get(\"USE_LLM_PLANNER\", \"1\") == \"1\"\n",
    "PLANNER_MODEL            = os.environ.get(\"PLANNER_MODEL\", \"Qwen/Qwen2.5-7B-Instruct\")\n",
    "PLANNER_LOAD_IN_4BIT     = os.environ.get(\"PLANNER_LOAD_IN_4BIT\", \"1\") == \"1\"\n",
    "PLANNER_ALLOW_CPU        = os.environ.get(\"PLANNER_ALLOW_CPU\", \"0\") == \"1\"\n",
    "PLANNER_MAX_NEW_TOKENS   = int(os.environ.get(\"PLANNER_MAX_NEW_TOKENS\", \"512\"))\n",
    "PLANNER_MAX_ATOMIC       = int(os.environ.get(\"PLANNER_MAX_ATOMIC\", \"4\"))\n",
    "PLANNER_MIN_OVERLAP      = float(os.environ.get(\"PLANNER_MIN_OVERLAP\", \"0.18\"))\n",
    "# Per-tier article quotas (max_k returned by article_bounds_for_complexity).\n",
    "SUBMIT_ARTICLE_MAX       = int(os.environ.get(\"SUBMIT_ARTICLE_MAX\", \"16\"))\n",
    "SUBMIT_ARTICLE_MAX_SIMPLE  = int(os.environ.get(\"SUBMIT_ARTICLE_MAX_SIMPLE\", \"2\"))\n",
    "SUBMIT_ARTICLE_MAX_MEDIUM  = int(os.environ.get(\"SUBMIT_ARTICLE_MAX_MEDIUM\", \"4\"))\n",
    "SUBMIT_ARTICLE_MAX_COMPLEX = int(os.environ.get(\"SUBMIT_ARTICLE_MAX_COMPLEX\", \"12\"))\n",
]


def patch_config(cell):
    src = _src(cell)
    joined = _joined(cell)
    marker = "USE_LLM_PLANNER"
    if marker in joined:
        return False  # already patched
    anchor = 'ARTICLE_CONTEXT_TOPK = int(os.environ.get("ARTICLE_CONTEXT_TOPK", "16"))\n'
    idx = _find_exact(src, anchor)
    if idx < 0:
        raise AssertionError("config anchor ARTICLE_CONTEXT_TOPK line not found")
    src[idx + 1:idx + 1] = CONFIG_LINES
    return True


# ---------------------------------------------------------------------------
# Block B — planning machinery (cell 22), inserted before QwenIRACGenerator
# ---------------------------------------------------------------------------
PLANNER_MARKER = "QUESTION_TYPES = {"   # unique sentinel of the inserted block


# Text helpers (ported verbatim from hybridrag-decomp.ipynb cell 1, minus the
# LEGAL_EXPANSIONS list which is not needed by the planning functions).
TEXT_HELPERS = """# --- text normalisation + lexical tokeniser (ported from hybridrag-decomp) ---
# Required by the query-complexity planning machinery below.
_word_re = re.compile(r'\\w+', re.UNICODE)
STOPWORDS = {
    'và', 'hoặc', 'của', 'các', 'những', 'một', 'này', 'đó', 'thì', 'là', 'có', 'bị', 'được',
    'phải', 'cho', 'về', 'trong', 'ngoài', 'theo', 'nếu', 'khi', 'như', 'để', 'với', 'từ', 'ra',
    'sao', 'gì', 'nào', 'bao', 'nhiêu', 'trường', 'hợp', 'cần', 'muốn', 'hỏi', 'tôi', 'công', 'ty'
}

def normalize_text(text):
    text = str(text).replace('\\u200b', ' ').replace('\\ufeff', ' ')
    return re.sub(r'\\s+', ' ', text).strip()

def normalize_key(text):
    return normalize_text(text).lower()

def tokenize_lexical(text, keep_stopwords=False):
    toks = _word_re.findall(normalize_text(text).lower())
    toks = [t for t in toks if len(t) >= 2]
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks

"""

# Planning machinery (ported from hybridrag-decomp.ipynb cell 3).  The CFG[...]
# references are converted to flat constants by _apply_cfg_constant().
PLANNER_BODY_RAW = """# --- LLM query decomposition + complexity planning -----------------------
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
        r'[;?。]+|\\s+(?:đồng thời|ngoài ra|bên cạnh đó|trong trường hợp|nếu|khi|và nếu|và phải|và cần|đặc biệt nếu)\\s+',
        q,
        flags=re.IGNORECASE,
    )
    out = []
    for piece in pieces:
        piece = normalize_text(piece.strip(' ,.-:'))
        if 25 <= len(piece) <= 300:
            out.append(piece)
    if len(out) <= 1 and len(q) > 120:
        for piece in re.split(r',\\s+| và ', q):
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
    if re.search(r'\\b(quy định|liệt kê|bao gồm|gồm|những|các)\\b.*\\b(nào|gì)\\b', q) and not re.search(r'\\b(thủ tục|hồ sơ|điều kiện|xử phạt|vi phạm|trách nhiệm|nghĩa vụ|ưu đãi|hỗ trợ)\\b', q):
        return 'definition_listing'
    if re.search(r'\\b(thủ tục|hồ sơ|tài liệu|chứng cứ|đơn yêu cầu|chuẩn bị)\\b', q):
        return 'procedure'
    if re.search(r'\\b(điều kiện|yêu cầu|tiêu chí|đáp ứng)\\b', q):
        return 'condition'
    if re.search(r'\\b(quyền|nghĩa vụ|trách nhiệm)\\b', q):
        return 'rights_obligations'
    if re.search(r'\\b(phạt|xử phạt|vi phạm|xử lý|khắc phục)\\b', q):
        return 'sanction'
    if re.search(r'\\b(hỗ trợ|ưu đãi|miễn|giảm)\\b', q):
        return 'support_incentive'
    if re.search(r'\\b(thời hạn|bao lâu|khi nào|mấy ngày)\\b', q):
        return 'deadline'
    if re.search(r'\\b(khác gì|khác biệt|so sánh)\\b', q):
        return 'comparison'
    if len(tokenize_lexical(question, keep_stopwords=True)) >= 45:
        return 'scenario'
    return 'other'


def is_simple_listing_query(question):
    q = normalize_key(question)
    toks = tokenize_lexical(question, keep_stopwords=True)
    if len(toks) > 16:
        return False
    if re.search(r'\\b(thủ tục|hồ sơ|điều kiện|xử phạt|vi phạm|khắc phục|trách nhiệm|nghĩa vụ|ưu đãi|hỗ trợ|đấu thầu|bồi thường)\\b', q):
        return False
    listing_patterns = [
        r'\\bquy định\\s+(?:những|các)?\\s*.+\\s+nào\\b',
        r'\\b(?:những|các)\\s+.+\\s+nào\\b',
        r'\\b.+\\s+bao gồm\\s+(?:những|các)?\\s+gì\\b',
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
    multi_markers = len(re.findall(r'\\b(và|đồng thời|ngoài ra|nếu|khi|hồ sơ|chứng cứ|xử lý|khắc phục|nghĩa vụ|trách nhiệm|thiệt hại|giám định)\\b', q))
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
        'atomic_questions': atomic[:PLANNER_MAX_ATOMIC],
        'must_have_terms': fallback_must_terms(question),
        'question_type': detect_question_type(question),
        'rationale_short': reason,
        'planner_fallback': True,
        'planner_error': reason,
        'raw_plan_text': '',
    }
    return refine_query_plan_heuristics(question, plan)


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
    original_has_citation = bool(re.search(r'\\b(Điều\\s+\\d+|Luật\\s+\\d+|Nghị định\\s+\\d+|Thông tư\\s+\\d+|Nghị quyết\\s+\\d+|\\d+/\\d{4}/[A-ZĐ-]+)\\b', original, flags=re.IGNORECASE))
    if original_has_citation:
        return False
    return bool(re.search(r'\\b(Điều\\s+\\d+|\\d+/\\d{4}/[A-ZĐ-]+)\\b', str(text), flags=re.IGNORECASE))


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

    plan = {
        'complexity': complexity,
        'atomic_questions': clean_atomic,
        'must_have_terms': terms,
        'question_type': qtype,
        'rationale_short': normalize_text(raw_plan.get('rationale_short', ''))[:260],
        'planner_fallback': False,
        'planner_error': '',
        'raw_plan_text': raw_text,
    }
    return refine_query_plan_heuristics(question, plan)


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
  "question_type": "procedure|condition|rights_obligations|sanction|support_incentive|scenario|deadline|comparison|definition_listing|other",
  "rationale_short": "lý do ngắn, không quá 1 câu"
}}

Quy tắc:
- simple: 1 vấn đề, thường 1 văn bản/1 điều.
- medium: 2 ý hoặc cross-doc nhẹ.
- complex: nhiều mệnh đề, multi-hop, nhiều lĩnh vực/văn bản.
- atomic_questions phải giữ đúng ý gốc, không thêm vấn đề mới.
- Không nêu "Điều X", mã luật, tên văn bản cụ thể nếu câu hỏi không nêu.
- Không dùng markdown, không giải thích ngoài JSON.
'''
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ], tokenize=False, add_generation_prompt=True)
        return system + '\\n\\n' + user + '\\n\\nJSON:'

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
    \"\"\"Per-tier article quota: {min_k, target_k, max_k}.

    simple  -> 1 article,   complex -> up to 12,  medium -> up to 4.
    Each max_k is capped by SUBMIT_ARTICLE_MAX (the hard ceiling).
    \"\"\"
    if complexity == 'simple':
        return {'min_k': 1, 'target_k': 1, 'max_k': min(SUBMIT_ARTICLE_MAX_SIMPLE, SUBMIT_ARTICLE_MAX)}
    if complexity == 'complex':
        return {'min_k': 4, 'target_k': 10, 'max_k': min(SUBMIT_ARTICLE_MAX_COMPLEX, SUBMIT_ARTICLE_MAX)}
    return {'min_k': 2, 'target_k': 4, 'max_k': min(SUBMIT_ARTICLE_MAX_MEDIUM, SUBMIT_ARTICLE_MAX)}


"""


def _to_source_lines(block_text):
    """Split a multi-line block string into the ipynb 'source' line list.

    Every line keeps its trailing newline except the final one (nbformat rule).
    """
    lines = block_text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    return lines


def patch_generator(cell):
    src = _src(cell)
    joined = _joined(cell)
    if PLANNER_MARKER in joined:
        return False  # already patched
    # anchor: the IRAC generator header comment that precedes the class
    anchor = "# --- IRAC answer generator (ported from hybridrag-decomp.ipynb) -----------\n"
    idx = _find_exact(src, anchor)
    if idx < 0:
        # fall back to the class line itself
        idx = _find_exact(src, "class QwenIRACGenerator:\n")
        if idx < 0:
            raise AssertionError("generator cell anchor not found")
    # The CFG references in PLANNER_BODY_RAW are still literal 'CFG[...]'; the
    # raw text above was already written with the constant names directly, so
    # no substitution is needed. Insert helpers + body before the anchor.
    helper_lines = _to_source_lines(TEXT_HELPERS)
    body_lines = _to_source_lines(PLANNER_BODY_RAW)
    src[idx:idx] = helper_lines + body_lines
    return True


# ---------------------------------------------------------------------------
# Block C — wire planner + tiered article quota into the batch loop (cell 23)
# ---------------------------------------------------------------------------
def patch_batch(cell):
    src = _src(cell)
    joined = _joined(cell)

    # --- C1: instantiate the planner right after the generator block ----------
    planner_inst_marker = "planner = LLMQueryPlanner()"
    if planner_inst_marker in joined:
        c1_done = True
    else:
        c1_done = False
        # anchor: the end of the generator-instantiation block
        anchor = '    print("[batch] USE_GENERATOR=False -> answer field left empty (retrieval-only).")\n'
        idx = _find_exact(src, anchor)
        if idx < 0:
            raise AssertionError("batch cell: generator-instant block anchor not found")
        inst_lines = [
            "\n",
            "# --- instantiate the query planner (complexity assessment) ------------\n",
            "# Always built; when USE_LLM_PLANNER=0 (or CPU w/o override) it self-\n",
            "# disables and falls back to the rule-based planner, so plan() is safe.\n",
            "planner = LLMQueryPlanner()\n",
        ]
        src[idx + 1:idx + 1] = inst_lines

    # --- C2: per-query plan + tiered article quota ---------------------------
    quota_marker = "bounds = article_bounds_for_complexity"
    if quota_marker in joined:
        c2_done = True
    else:
        c2_done = False
        # anchor lines (unpatched):
        a_retrieve = "    hits = decomp_retriever.retrieve(query, fetch_text=True)\n"
        a_lists = "    docs, articles = make_relevant_lists(hits)\n"
        a_ctx = "    contexts = build_gen_contexts(hits)\n"
        i_retrieve = _find_exact(src, a_retrieve)
        i_lists = _find_exact(src, a_lists)
        i_ctx = _find_exact(src, a_ctx)
        if i_retrieve < 0 or i_lists < 0 or i_ctx < 0:
            raise AssertionError("batch cell: retrieve/make_relevant_lists anchor not found")
        if not (i_retrieve < i_lists < i_ctx):
            raise AssertionError("batch cell: anchor order unexpected")
        # Replace the three anchor lines with: retrieve -> plan -> lists ->
        # tiered truncation -> contexts.
        replacement = [
            "    hits = decomp_retriever.retrieve(query, fetch_text=True)\n",
            "    query_plan = planner.plan(str(query))\n",
            "    complexity = query_plan.get('complexity', 'medium')\n",
            "    docs, articles = make_relevant_lists(hits)\n",
            "    # Apply the per-complexity-tier article quota.\n",
            "    bounds = article_bounds_for_complexity(complexity)\n",
            "    _max_articles = bounds['max_k']\n",
            "    articles = articles[:_max_articles]\n",
            "    # Trim docs to those whose law_id+ten_van_ban match a kept article. Each\n",
            "    # kept article is \"law_id|ten_van_ban|dieu_so\"; its parent doc key is the\n",
            "    # \"law_id|ten_van_ban\" prefix (strip the trailing dieu_so segment).\n",
            "    _keep_doc_keys = {a.rsplit('|', 1)[0] for a in articles if '|' in a}\n",
            "    docs = [d for d in docs if d in _keep_doc_keys]\n",
            "    contexts = build_gen_contexts(hits)\n",
        ]
        src[i_retrieve:i_ctx + 1] = replacement

    # --- C3: log complexity in the [done] line -------------------------------
    if "c={complexity}" in "".join(src):
        c3_done = True
    else:
        c3_done = False
        old_l1 = '    print(f"  [done] id={qid}  hits={len(hits)}  docs={len(docs)}  "\n'
        old_l2 = '          f"articles={len(articles)}  ans_len={len(answer)}")\n'
        new_l1 = '    print(f"  [done] id={qid}  c={complexity}  hits={len(hits)}  "\n'
        new_l2 = '          f"docs={len(docs)}  articles={len(articles)}  ans_len={len(answer)}")\n'
        idx = _find_exact(src, old_l1)
        if idx < 0 or src[idx + 1] != old_l2:
            raise AssertionError("batch cell: [done] log line not found")
        src[idx:idx + 2] = [new_l1, new_l2]

    # --- C4: release the planner before the generator ------------------------
    del_marker = "del planner"
    if del_marker in joined:
        c4_done = True
    else:
        c4_done = False
        anchor = "# release the generator before §9 validation / §10 cleanup\n"
        idx = _find_exact(src, anchor)
        if idx < 0:
            raise AssertionError("batch cell: generator-release anchor not found")
        src[idx:idx] = ["# release the query planner (frees its GPU model if loaded)\n",
                        "try:\n",
                        "    del planner\n",
                        "except NameError:\n",
                        "    pass\n"]
    # c*_done is True when the marker was ALREADY present (no-op for that block).
    # A change was made iff any block was NOT already present, i.e. any flag False.
    return not (c1_done and c2_done and c3_done and c4_done)


# ---------------------------------------------------------------------------
# Block D — add 'planner' to the cleanup tuple (cell 27)
# ---------------------------------------------------------------------------
def patch_cleanup(cell):
    src = _src(cell)
    joined = _joined(cell)
    # detect already-patched: the tuple now spans three lines ending with 'planner'
    if "'planner'):" in joined or "'planner',\n" in joined:
        return False
    old = ["for _name in ('decomp_retriever', 'base_retriever', 'router',\n",
           "              'query_encoder', 'faiss_index', 'graph_expander', 'fts',\n",
           "              'generator'):\n"]
    new = ["for _name in ('decomp_retriever', 'base_retriever', 'router',\n",
           "              'query_encoder', 'faiss_index', 'graph_expander', 'fts',\n",
           "              'generator', 'planner'):\n"]
    idx = _find_exact(src, old[0])
    if idx < 0:
        raise AssertionError("cleanup cell: tuple anchor not found")
    # verify the following two lines match
    if src[idx + 1] != old[1] or src[idx + 2] != old[2]:
        raise AssertionError("cleanup cell: tuple lines do not match expected pattern")
    src[idx:idx + 3] = new
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _find_exact(src, line_text):
    for i, s in enumerate(src):
        if s == line_text:
            return i
    return -1


def _print_cfg(cell):
    src = _src(cell)
    # extend the config print dict to surface the planner knobs (idempotent)
    joined = _joined(cell)
    if "USE_LLM_PLANNER" in joined and "'use_llm_planner'" not in joined:
        old = ('print({"gen_model": GEN_MODEL, "use_generator": USE_GENERATOR,\n'
               '       "load_in_4bit": GEN_LOAD_IN_4BIT, "gen_context_topk": GEN_CONTEXT_TOPK,\n'
               '       "results_path": RESULTS_PATH, "submission_zip_path": SUBMISSION_ZIP_PATH})\n')
        new = ('print({"gen_model": GEN_MODEL, "use_generator": USE_GENERATOR,\n'
               '       "load_in_4bit": GEN_LOAD_IN_4BIT, "gen_context_topk": GEN_CONTEXT_TOPK,\n'
               '       "use_llm_planner": USE_LLM_PLANNER, "planner_model": PLANNER_MODEL,\n'
               '       "results_path": RESULTS_PATH, "submission_zip_path": SUBMISSION_ZIP_PATH})\n')
        idx = _find_exact(src, old[0])
        if idx >= 0 and src[idx + 1] == old[1] and src[idx + 2] == old[2]:
            src[idx:idx + 3] = new
            return True
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cfg = cells[20]
    gen = cells[22]
    batch = cells[23]
    cleanup = cells[27]

    changes = []
    if patch_config(cfg):
        changes.append("cell20 config knobs")
        _print_cfg(cfg)
    if patch_generator(gen):
        changes.append("cell22 planning machinery")
    if patch_batch(batch):
        changes.append("cell23 batch wiring")
    if patch_cleanup(cleanup):
        changes.append("cell27 cleanup tuple")

    if not changes:
        print("no-op: already patched")
        return

    # sanity: ensure the generator cell still contains QwenIRACGenerator
    if "class QwenIRACGenerator:" not in _joined(gen):
        raise AssertionError("QwenIRACGenerator class lost during patch")

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"patched OK: {', '.join(changes)}")


if __name__ == "__main__":
    main()
