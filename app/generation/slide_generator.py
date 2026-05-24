"""Slide generator — creates PowerPoint presentations from PDF content."""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MODEL_NAME = "command-r-plus-08-2024"
OUTPUTS_DIR = Path("data/outputs")

DEFAULT_NUM_SLIDES = 10
DEFAULT_BULLETS = 4
DEFAULT_DEPTH = "detailed"
DEFAULT_STYLE = "academic"

DEPTH_INSTRUCTIONS = {
    "summary": (
        "Write SHORT bullet points — one line each. "
        "Overview only, no deep explanation."
    ),
    "detailed": (
        "Write DETAILED bullet points — 1-2 sentences each. "
        "Include explanations and context."
    ),
    "exam": (
        "Write bullet points as KEY FACTS to memorize. "
        "Include definitions, formulas, and important terms. "
        "Phrase as facts a student should know for an exam."
    )
}

STYLE_INSTRUCTIONS = {
    "academic": "Use formal academic language with precise terminology.",
    "simple": "Use plain simple language. Avoid jargon. Easy to understand.",
    "visual_hints": (
        "After each bullet, add a [VISUAL: ...] hint suggesting "
        "a diagram or image that would illustrate this point."
    )
}


def _get_cohere_client() -> cohere.ClientV2:
    """Get Cohere client."""
    return cohere.ClientV2(api_key=COHERE_API_KEY)


async def _extract_topics(
    chunks: list[dict[str, Any]],
    num_topics: int,
    topic_focus: str
) -> list[dict[str, Any]]:
    """Extract slide topics from chunks using Cohere.

    Args:
        chunks: List of chunk dicts with text and metadata.
        num_topics: Number of topics to extract.
        topic_focus: "all" or specific chapter/topic name.

    Returns:
        List of dicts with keys: topic, chapter, key_concept
    """
    if topic_focus != "all":
        filtered = [
            c for c in chunks
            if topic_focus.lower() in c.get("chapter", "").lower() or
            topic_focus.lower() in c.get("chunk", "").lower()
        ]
        context_chunks = filtered[:20] if filtered else chunks[:20]
    else:
        context_chunks = chunks[:30]

    context = "\n\n".join([
        f"[{c.get('chapter', 'General')} | Page {c.get('page_num', 0)}]\n{c.get('chunk', '')[:300]}"
        for c in context_chunks
    ])

    prompt = f"""
You are analyzing a textbook. Extract exactly {num_topics} slide topics
from this content.

Content:
{context}

Return ONLY a JSON array like this:
[
  {{"topic": "Newton's Second Law", "chapter": "Chapter 3", "key_concept": "F = ma"}},
  {{"topic": "Circular Motion", "chapter": "Chapter 4", "key_concept": "centripetal force"}}
]

Rules:
- Exactly {num_topics} topics
- Topics should cover the most important concepts
- Each topic should be specific enough for one slide
- Return ONLY the JSON array, no other text
"""

    loop = asyncio.get_event_loop()
    client = _get_cohere_client()

    def _call():
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.message.content[0].text

    raw = await loop.run_in_executor(None, _call)

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        topics = json.loads(raw.strip())
        return topics[:num_topics]
    except Exception as e:
        print(f"[slide_generator] Topic parse error: {e}")
        return [
            {"topic": f"Topic {i+1}", "chapter": "General", "key_concept": ""}
            for i in range(num_topics)
        ]


async def _generate_slide_content(
    topic: dict[str, Any],
    chunks: list[dict[str, Any]],
    bullets_per_slide: int,
    depth: str,
    style: str
) -> dict[str, Any]:
    """Generate content for a single slide using Cohere.

    Args:
        topic: Topic dict with topic, chapter, key_concept
        chunks: All chunks for context
        bullets_per_slide: Number of bullet points
        depth: "summary" | "detailed" | "exam"
        style: "academic" | "simple" | "visual_hints"

    Returns:
        Dict with keys: title, bullets, chapter
    """
    relevant = [
        c for c in chunks
        if topic.get("topic", "").lower() in c.get("chunk", "").lower() or
        topic.get("chapter", "").lower() in c.get("chapter", "").lower()
    ][:5]

    context = "\n\n".join([
        c.get("chunk", "")[:400] for c in relevant
    ]) if relevant else "No specific context available."

    depth_inst = DEPTH_INSTRUCTIONS.get(depth, DEPTH_INSTRUCTIONS["detailed"])
    style_inst = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["academic"])

    prompt = f"""
Create content for a presentation slide about: {topic.get('topic', '')}
Chapter: {topic.get('chapter', '')}
Key concept: {topic.get('key_concept', '')}

Context from textbook:
{context}

{depth_inst}
{style_inst}

Return ONLY a JSON object like this:
{{
  "title": "Newton's Second Law",
  "bullets": [
    "Force equals mass times acceleration (F = ma)",
    "Doubling the force doubles the acceleration",
    "Doubling the mass halves the acceleration",
    "Direction of acceleration matches direction of net force"
  ]
}}

Rules:
- Exactly {bullets_per_slide} bullet points
- Title must be concise (max 8 words)
- Base content ONLY on the provided context
- Return ONLY the JSON object, no other text
"""

    loop = asyncio.get_event_loop()
    client = _get_cohere_client()

    def _call():
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.4
        )
        return response.message.content[0].text

    raw = await loop.run_in_executor(None, _call)

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        content = json.loads(raw.strip())
        return {
            "title": content.get("title", topic.get("topic", "")),
            "bullets": content.get("bullets", [])[:bullets_per_slide],
            "chapter": topic.get("chapter", "")
        }
    except Exception as e:
        print(f"[slide_generator] Slide parse error: {e}")
        return {
            "title": topic.get("topic", "Slide"),
            "bullets": [f"Key point {i+1}" for i in range(bullets_per_slide)],
            "chapter": topic.get("chapter", "")
        }


def _build_pptx(
    slides_data: list[dict[str, Any]],
    source_pdf: str,
    include_title_slide: bool,
    include_summary_slide: bool
) -> str:
    """Build a .pptx file from slide content data.

    Args:
        slides_data: List of dicts with title and bullets
        source_pdf: Original PDF filename for title slide
        include_title_slide: Whether to add a title slide
        include_summary_slide: Whether to add a summary slide

    Returns:
        Absolute path to saved .pptx file as string
    """
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    BG_COLOR = RGBColor(0x1A, 0x1A, 0x2E)
    TITLE_COLOR = RGBColor(0xE9, 0x4F, 0x37)
    TEXT_COLOR = RGBColor(0xFF, 0xFF, 0xFF)
    ACCENT_COLOR = RGBColor(0x39, 0x3E, 0x46)
    BULLET_COLOR = RGBColor(0xE0, 0xE0, 0xE0)

    def set_bg(slide, color):
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = color

    if include_title_slide:
        layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(layout)
        set_bg(slide, BG_COLOR)

        title_box = slide.shapes.add_textbox(
            Inches(1), Inches(2.5), Inches(11.33), Inches(1.5)
        )
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = Path(source_pdf).stem.replace("_", " ").title()
        p.font.size = Pt(40)
        p.font.bold = True
        p.font.color.rgb = TITLE_COLOR
        p.alignment = PP_ALIGN.CENTER

        sub_box = slide.shapes.add_textbox(
            Inches(1), Inches(4.2), Inches(11.33), Inches(0.8)
        )
        tf2 = sub_box.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = f"Generated by Online Teacher • {datetime.now().strftime('%B %Y')}"
        p2.font.size = Pt(18)
        p2.font.color.rgb = TEXT_COLOR
        p2.alignment = PP_ALIGN.CENTER

    for slide_data in slides_data:
        layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(layout)
        set_bg(slide, BG_COLOR)

        if slide_data.get("chapter"):
            ch_box = slide.shapes.add_textbox(
                Inches(0.5), Inches(0.2), Inches(12), Inches(0.4)
            )
            ch_tf = ch_box.text_frame
            ch_p = ch_tf.paragraphs[0]
            ch_p.text = slide_data["chapter"]
            ch_p.font.size = Pt(14)
            ch_p.font.color.rgb = ACCENT_COLOR

        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.7), Inches(12.33), Inches(1.2)
        )
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = slide_data["title"]
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = TITLE_COLOR

        line = slide.shapes.add_shape(
            1,
            Inches(0.5), Inches(1.85),
            Inches(12.33), Inches(0.05)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = TITLE_COLOR
        line.line.fill.background()

        content_box = slide.shapes.add_textbox(
            Inches(0.7), Inches(2.1), Inches(11.93), Inches(4.8)
        )
        tf = content_box.text_frame
        tf.word_wrap = True

        for i, bullet in enumerate(slide_data.get("bullets", [])):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"• {bullet}"
            p.font.size = Pt(20)
            p.font.color.rgb = BULLET_COLOR
            p.space_after = Pt(12)

    if include_summary_slide:
        layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(layout)
        set_bg(slide, BG_COLOR)

        title_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(0.7), Inches(12.33), Inches(1.2)
        )
        tf = title_box.text_frame
        p = tf.paragraphs[0]
        p.text = "Key Takeaways"
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = TITLE_COLOR

        content_box = slide.shapes.add_textbox(
            Inches(0.7), Inches(2.1), Inches(11.93), Inches(4.8)
        )
        tf = content_box.text_frame
        tf.word_wrap = True

        for i, slide_data in enumerate(slides_data[:8]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"• {slide_data['title']}"
            p.font.size = Pt(20)
            p.font.color.rgb = BULLET_COLOR
            p.space_after = Pt(8)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{Path(source_pdf).stem}_slides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
    output_path = OUTPUTS_DIR / filename
    prs.save(str(output_path))
    print(f"[slide_generator] Saved: {output_path}")
    return str(output_path.resolve())


async def generate_slides(
    chunks: list[dict[str, Any]],
    source_pdf: str,
    num_slides: int = DEFAULT_NUM_SLIDES,
    bullets_per_slide: int = DEFAULT_BULLETS,
    depth: str = DEFAULT_DEPTH,
    topic_focus: str = "all",
    style: str = DEFAULT_STYLE,
    include_title_slide: bool = True,
    include_summary_slide: bool = True
) -> dict[str, Any]:
    """Generate a PowerPoint presentation from ingested PDF chunks.

    Args:
        chunks: All chunks from Qdrant for the source PDF.
        source_pdf: Original PDF filename.
        num_slides: Number of content slides to generate.
        bullets_per_slide: Bullet points per slide (3-5).
        depth: "summary" | "detailed" | "exam"
        topic_focus: "all" | chapter name | topic name
        style: "academic" | "simple" | "visual_hints"
        include_title_slide: Add title slide at start.
        include_summary_slide: Add summary slide at end.

    Returns:
        Dict with keys:
          - file_path: str (absolute path to .pptx)
          - num_slides: int (actual slides generated)
          - topics: list[str] (slide titles)
          - status: str ("success" or "error")
          - error: str | None
    """
    try:
        print(f"[slide_generator] Generating {num_slides} slides for {source_pdf}")

        topics = await _extract_topics(chunks, num_slides, topic_focus)
        print(f"[slide_generator] Extracted {len(topics)} topics")

        tasks = [
            _generate_slide_content(topic, chunks, bullets_per_slide, depth, style)
            for topic in topics
        ]
        slides_data = await asyncio.gather(*tasks)
        print(f"[slide_generator] Generated content for {len(slides_data)} slides")

        file_path = _build_pptx(
            list(slides_data),
            source_pdf,
            include_title_slide,
            include_summary_slide
        )

        return {
            "file_path": file_path,
            "num_slides": len(slides_data),
            "topics": [s["title"] for s in slides_data],
            "status": "success",
            "error": None
        }

    except Exception as e:
        print(f"[slide_generator] Error: {e}")
        return {
            "file_path": "",
            "num_slides": 0,
            "topics": [],
            "status": "error",
            "error": str(e)
        }