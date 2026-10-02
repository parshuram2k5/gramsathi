import json
import os
import html

def clean_text(text):
    if not text:
        return ""
    text = str(text).strip()
    return html.unescape(text)

def build_scheme_doc(scheme):
    fields = scheme.get("fields") or {}
    if not fields:
        return None
        
    details = (scheme.get("details") or {}).get("en") or {}
    sc = details.get("schemeContent") or {}
    ec = details.get("eligibilityCriteria") or {}
    ap = details.get("applicationProcess") or []
    
    scheme_id = scheme.get("id", "")
    slug = fields.get("slug", "")
    scheme_name = clean_text(fields.get("schemeName", ""))
    ministry = clean_text(fields.get("nodalMinistryName", ""))
    level = clean_text(fields.get("level", ""))
    
    states = [clean_text(s) for s in (fields.get("beneficiaryState") or []) if s]
    categories = [clean_text(c) for c in (fields.get("schemeCategory") or []) if c]
    tags = [clean_text(t) for t in (fields.get("tags") or []) if t]
    scheme_for = clean_text(fields.get("schemeFor", ""))
    brief_desc = clean_text(fields.get("briefDescription", ""))
    
    detailed_desc = clean_text(sc.get("detailedDescription_md") or brief_desc)
    benefits = clean_text(sc.get("benefits_md") or "")
    eligibility = clean_text(ec.get("eligibilityDescription_md") or "")
    exclusions = clean_text(sc.get("exclusions_md") or "")
    
    proc_parts = []
    if isinstance(ap, list):
        for p in ap:
            if isinstance(p, dict):
                mode = clean_text(p.get("mode", ""))
                proc_md = clean_text(p.get("process_md", ""))
                url = clean_text(p.get("url", ""))
                sub = []
                if mode:
                    sub.append(f"**Mode:** {mode}")
                if url:
                    sub.append(f"**Official Portal/Link:** {url}")
                if proc_md:
                    sub.append(proc_md)
                if sub:
                    proc_parts.append("\n".join(sub))
    application_process = "\n\n".join(proc_parts).strip()
    
    # 1. Structured Markdown for Page Content
    doc_lines = [f"# {scheme_name}\n"]
    meta_info = []
    if ministry:
        meta_info.append(f"**Ministry:** {ministry}")
    if level:
        meta_info.append(f"**Level:** {level}")
    if states:
        meta_info.append(f"**States/UTs:** {', '.join(states)}")
    if categories:
        meta_info.append(f"**Categories:** {', '.join(categories)}")
    if tags:
        meta_info.append(f"**Tags:** {', '.join(tags)}")
    if scheme_for:
        meta_info.append(f"**Target Beneficiaries:** {scheme_for}")
    if meta_info:
        doc_lines.append(" | ".join(meta_info) + "\n")
        
    if detailed_desc:
        doc_lines.append(f"## Description\n{detailed_desc}\n")
    if benefits:
        doc_lines.append(f"## Benefits\n{benefits}\n")
    if eligibility:
        doc_lines.append(f"## Eligibility Criteria\n{eligibility}\n")
    if exclusions:
        doc_lines.append(f"## Exclusions\n{exclusions}\n")
    if application_process:
        doc_lines.append(f"## How to Apply\n{application_process}\n")
        
    page_content = "\n".join(doc_lines).strip()
    
    # 2. Optimized Search Representation for Dense + Lexical retrieval
    search_components = [
        f"Scheme: {scheme_name}",
        f"Ministry: {ministry}" if ministry else "",
        f"Level: {level}" if level else "",
        f"States: {', '.join(states)}" if states else "",
        f"Category: {', '.join(categories)}" if categories else "",
        f"Tags: {', '.join(tags)}" if tags else "",
        f"Target: {scheme_for}" if scheme_for else "",
        f"Brief: {brief_desc[:350]}" if brief_desc else "",
        f"Eligibility: {eligibility[:300]}" if eligibility else "",
        f"Benefits: {benefits[:250]}" if benefits else ""
    ]
    search_text = " | ".join(c for c in search_components if c)
    
    # 3. Metadata for filtering and ranking
    metadata = {
        "id": scheme_id,
        "slug": slug,
        "schemeName": scheme_name,
        "nodalMinistryName": ministry,
        "level": level,
        "schemeCategory": categories,
        "beneficiaryState": states,
        "tags": tags,
        "schemeFor": scheme_for,
        "briefDescription": brief_desc
    }
    
    return {
        "id": scheme_id,
        "metadata": metadata,
        "page_content": page_content,
        "search_text": search_text
    }

def process_schemes(input_file="schemes_detailed.json", output_file="schemes_rag_dataset.jsonl"):
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} schemes. Processing...")
    
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for scheme in data:
            doc = build_scheme_doc(scheme)
            if not doc:
                continue
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed_count += 1
            
    print(f"Successfully processed {processed_count} schemes into {output_file}")

if __name__ == "__main__":
    input_path = "schemes_detailed.json"
    output_path = "schemes_rag_dataset.jsonl"
    
    if os.path.exists(input_path):
        process_schemes(input_path, output_path)
    else:
        print(f"Error: {input_path} not found.")
