from __future__ import annotations

"""
Gerador de roteiro estruturado (portado do projeto antigo)
SaÃ­da: {"title": "...", "scenes": [{"id": "1", "text": "...", "visualPrompt": "...", "animationType": "..."}]}
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in __import__("sys").path:
    __import__("sys").path.append(str(ROOT.parent))

from video_factory.config import (  # noqa: E402
    ANIMATION_TYPES,
    ASPECT,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    SCENE_COUNTS,
    SCRIPT_CHUNK_ALLOW_SHORTAGE,
    SCRIPT_CHUNK_CONTEXT_SCENES,
    SCRIPT_CHUNK_ENABLED,
    SCRIPT_CHUNK_SCENES,
    SCRIPT_EXPAND_ENABLED,
    SCRIPT_EXPAND_MAX_PASSES,
    SCRIPT_EXPAND_MIN_ADD_WORDS,
    SCRIPT_EXPAND_SCENES_PER_PASS,
    SCRIPT_EXPAND_MAX_ADD_WORDS,
    WORDS_PER_MINUTE,
    NARRATION_POLISH_ENABLED,
    NARRATION_POLISH_MAX_WORDS,
)
from video_factory.openai_compat_client import OpenAICompatClient  # noqa: E402

log = logging.getLogger(__name__)
PUNCT_RE = re.compile(r"[.!?]")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+", flags=re.UNICODE)


# ------------------------ util ------------------------- #
def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_by_words(text: str, max_words: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    max_words = max(8, int(max_words or 26))
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _should_polish_narration(text: str, max_words: int) -> bool:
    clean = _normalize_ws(text)
    if not clean:
        return False
    if not PUNCT_RE.search(clean):
        return True
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    for sent in sentences:
        if _count_words(sent) > int(max_words * 1.6):
            return True
    return False


def _polish_narration_text(text: str, max_words: int) -> str:
    clean = _normalize_ws(text)
    if not clean:
        return clean
    if not _should_polish_narration(clean, max_words):
        return clean

    sentences: List[str] = []
    if not PUNCT_RE.search(clean):
        sentences = _split_by_words(clean, max_words)
    else:
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            end_char = s[-1] if s[-1] in ".!?" else ""
            core = s[:-1].strip() if end_char else s
            if _count_words(core) > int(max_words * 1.6):
                sentences.extend(_split_by_words(core, max_words))
            else:
                sentence = core + end_char if end_char else core
                sentences.append(sentence)

    fixed: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s = s + "."
        fixed.append(s)

    return " ".join(fixed).strip()


def _polish_script_narration(script: Dict[str, Any], max_words: int) -> None:
    cenas = script.get("scenes") or []
    for cena in cenas:
        texto = str(cena.get("text", "") or "")
        cena["text"] = _polish_narration_text(texto, max_words)


# ------------------------ LLM calls ------------------------- #
def _ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"timeout": LLM_TIMEOUT}}
    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def _build_system_prompt(format_ratio: str) -> str:
    anim_list = ", ".join(ANIMATION_TYPES or [])
    return (
        "VocÃª Ã© roteirista e diretor criativo. Gere roteiro em portuguÃªs brasileiro, objetivo e visual.\n"
        "Responda SOMENTE JSON no formato:\n"
        '{"title": "...", "scenes": ['
        '{"id": "1", "text": "...", "visualPrompt": "...", "animationType": "..."}'
        "]}\n"
        f"As cenas devem estar no formato {format_ratio}. "
        f"Use animationType variando entre: {anim_list}. "
        "NÃ£o acrescente comentÃ¡rios fora do JSON."
    )


def _build_user_prompt(
    topic: str,
    duration_minutes: int,
    format_ratio: str,
    num_scenes: int,
    total_words: int,
    min_words_per_scene: int,
    max_words_per_scene: int,
    min_total_words: int,
    max_total_words: int,
) -> str:
    return (
        f"Tema: {topic}\n"
        f"DuraÃ§Ã£o alvo: {duration_minutes} minuto(s).\n"
        f"NÃºmero de cenas: {num_scenes}.\n"
        f"Total de palavras desejado: {min_total_words} a {max_total_words} (alvo {total_words}).\n"
        f"Palavras por cena: {min_words_per_scene} a {max_words_per_scene}.\n"
        "Cada cena deve trazer uma descriÃ§Ã£o visual curta em visualPrompt, com detalhes concretos (objetos, plano, iluminaÃ§Ã£o, clima).\n"
        "Mantenha o texto narrado conciso, claro e com ritmo; evite listas; feche frases com pontuaÃ§Ã£o.\n"
        "Use variedade de planos (aberto/mÃ©dio/close) e Ã¢ngulos quando sugerir visualPrompt.\n"
        "Retorne somente o JSON especificado."
    )


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        if "{" in text and "}" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
            except Exception:
                pass
        raise


def _coerce_structure(raw: Dict[str, Any], topic: str, num_scenes: int) -> Dict[str, Any]:
    scenes = raw.get("scenes") or []
    if not isinstance(scenes, list):
        scenes = []
    # Fallback scenes if missing
    if not scenes:
        scenes = [
            {
                "id": "1",
                "text": topic,
                "visualPrompt": topic,
                "animationType": "kenburns",
            }
        ]
    fixed = []
    anim_cycle = ANIMATION_TYPES or ["kenburns", "zoom_in", "zoom_out", "pan_left", "pan_right"]
    for idx, scene in enumerate(scenes[: num_scenes or len(scenes)]):
        text = _normalize_ws(str(scene.get("text") or "").strip())
        visual = scene.get("visualPrompt") or text or topic
        anim = (scene.get("animationType") or anim_cycle[idx % len(anim_cycle)]).strip()
        fixed.append(
            {
                "id": str(scene.get("id", idx + 1)),
                "text": text or topic,
                "visualPrompt": visual,
                "animationType": anim,
            }
        )
    return {"title": raw.get("title") or topic, "scenes": fixed}


# ------------------------ chunking & expansion helpers ------------------------- #
def _format_previous_scenes(previous_scenes: List[Dict[str, Any]], start_scene_index: int) -> str:
    if not previous_scenes:
        return ""
    lines = []
    first_index = max(0, start_scene_index - len(previous_scenes))
    for i, scene in enumerate(previous_scenes):
        scene_num = first_index + i + 1
        text = str(scene.get("text", "")).strip()
        if len(text) > 360:
            text = text[:360].rstrip() + "..."
        lines.append(f"- Cena {scene_num}: {text}")
    newline = chr(10)
    return "CENAS ANTERIORES (nÃ£o repetir, apenas continuidade):" + newline + newline.join(lines) + newline


def _build_user_prompt_chunk(
    topic: str,
    duration: int,
    total_words: int,
    num_scenes: int,
    words_per_scene: int,
    min_total_words: int,
    max_total_words: int,
    min_words_per_scene: int,
    max_words_per_scene: int,
    part_index: int,
    total_parts: int,
    start_scene_index: int,
    previous_scenes: List[Dict[str, Any]],
    fixed_title: Optional[str],
) -> str:
    extra_depth = ""
    if duration >= 5:
        extra_depth = "- Como Ã© um vÃ­deo mais longo, aprofunde com contexto, exemplos e transiÃ§Ãµes naturais.\n"
    title_hint = f"- Use exatamente este tÃ­tulo: {fixed_title}\n" if fixed_title else ""
    previous_block = _format_previous_scenes(previous_scenes, start_scene_index)
    return f"""PARTE {part_index}/{total_parts}
Crie um roteiro sobre: {topic}

ESTA Ã‰ A PARTE {part_index} DE {total_parts}.
- Gere APENAS {num_scenes} cenas novas, correspondendo Ã s cenas {start_scene_index + 1} a {start_scene_index + num_scenes}.
- NÃ£o repita cenas anteriores. Continue a narrativa.
{title_hint}{previous_block}ESPECIFICAÃ‡Ã•ES DESTA PARTE:
- DuraÃ§Ã£o total prevista do roteiro completo: {duration} minuto(s)
- Total de palavras desta parte: entre {min_total_words} e {max_total_words} (alvo ~{total_words})
- NÃºmero de cenas nesta parte: {num_scenes}
- Palavras por cena: entre {min_words_per_scene} e {max_words_per_scene} (alvo ~{words_per_scene})

IMPORTANTE:
- Cada cena deve ter narraÃ§Ã£o fluida, 1-2 frases, PT-BR.
- Pontue para guiar entonaÃ§Ã£o.
- NÃ£o faÃ§a cenas com menos de {min_words_per_scene} palavras nem mais que {max_words_per_scene}.
- Prompts visuais (visualPrompt) devem ser especÃ­ficos e em inglÃªs.
- Inclua animationType alternando entre os tipos permitidos.
{extra_depth}

RETORNE APENAS O JSON no formato:
{{"title":"...","scenes":[{{"id":"1","text":"...","visualPrompt":"...","animationType":"..."}}]}}"""


def _compute_chunk_word_budget(
    remaining_scenes: int,
    remaining_min_words: int,
    remaining_max_words: int,
    chunk_scenes: int,
    words_per_scene: int,
    min_words_per_scene: int,
    max_words_per_scene: int,
) -> Tuple[int, int, int]:
    remaining_after = max(0, remaining_scenes - chunk_scenes)
    min_possible = max(
        min_words_per_scene * chunk_scenes,
        remaining_min_words - max_words_per_scene * remaining_after,
    )
    max_possible = min(
        max_words_per_scene * chunk_scenes,
        remaining_max_words - min_words_per_scene * remaining_after,
    )
    if max_possible < min_possible:
        min_possible = min_words_per_scene * chunk_scenes
        max_possible = max_words_per_scene * chunk_scenes
    target = words_per_scene * chunk_scenes
    target = min(max_possible, max(min_possible, target))
    return int(min_possible), int(max_possible), int(target)


def _llm_generate(system_prompt: str, user_prompt: str) -> str:
    try:
        return _ollama_generate(system_prompt + "\n\n" + user_prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("Falha Ollama, tentando OpenAI compatÃ­vel: %s", exc)
        client = OpenAICompatClient(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        return client.chat(system_prompt + "\n\n" + user_prompt)


def _expand_scene_text(scene: Dict[str, Any], target_min_words: int, target_max_words: int, add_words: int) -> Optional[str]:
    original = str(scene.get("text", "")).strip()
    if not original:
        return None
    system_prompt = (
        "VocÃª Ã© um editor de roteiro. Expanda o texto mantendo sentido, fluidez e PT-BR. "
        'Retorne APENAS um JSON vÃ¡lido com a chave "text".'
    )
    user_prompt = (
        "TEXTO ATUAL:\n"
        f"{original}\n\n"
        f"Reescreva o texto acima adicionando pelo menos {add_words} palavras. "
        f"O texto final deve ter entre {target_min_words} e {target_max_words} palavras. "
        "NÃ£o repita ideias, mantenha o tom envolvente.\n\n"
        'Retorne APENAS: {"text": "..."}'
    )
    try:
        raw = _llm_generate(system_prompt, user_prompt)
        payload = _parse_json(raw)
        new_text = str(payload.get("text", "")).strip()
        return new_text or None
    except Exception as exc:  # noqa: BLE001
        log.warning("Falha ao expandir cena: %s", exc)
        return None


def _count_words_in_scenes(scenes: List[Dict[str, Any]]) -> int:
    return sum(_count_words(scene.get("text", "")) for scene in scenes)


def _expand_script_to_min_words(
    script: Dict[str, Any],
    min_total_words: int,
    max_total_words: int,
    min_words_per_scene: int,
    max_words_per_scene: int,
) -> None:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return

    max_passes = max(1, int(SCRIPT_EXPAND_MAX_PASSES))
    per_pass = max(1, int(SCRIPT_EXPAND_SCENES_PER_PASS))
    min_add = max(1, int(SCRIPT_EXPAND_MIN_ADD_WORDS))
    max_add = max(1, int(SCRIPT_EXPAND_MAX_ADD_WORDS))

    for _ in range(max_passes):
        counts = [_count_words(scene.get("text", "")) for scene in scenes]
        total = sum(counts)
        deficit = max(0, min_total_words - total)
        remaining_max = max(0, max_total_words - total)
        short_indices = [i for i, w in enumerate(counts) if w < min_words_per_scene]

        if deficit <= 0 and not short_indices:
            return

        candidates = short_indices if short_indices else list(range(len(scenes)))
        candidates = sorted(candidates, key=lambda i: counts[i])
        expanded = 0

        for idx in candidates:
            if expanded >= per_pass:
                break
            current = counts[idx]
            available = max_words_per_scene - current
            if available <= 0 or remaining_max <= 0:
                continue
            needed_for_scene = max(0, min_words_per_scene - current)
            target_add = max(needed_for_scene, min_add)
            if deficit > 0:
                target_add = max(target_add, min(deficit, max_add))
            target_add = min(target_add, available, remaining_max)
            if target_add <= 0:
                continue

            target_min = min(max_words_per_scene, max(min_words_per_scene, current + target_add))
            new_text = _expand_scene_text(
                scene=scenes[idx],
                target_min_words=target_min,
                target_max_words=max_words_per_scene,
                add_words=target_add,
            )
            if not new_text:
                continue

            scenes[idx]["text"] = new_text
            new_count = _count_words(new_text)
            delta = max(0, new_count - current)
            total += delta
            remaining_max = max(0, max_total_words - total)
            deficit = max(0, min_total_words - total)
            counts[idx] = new_count
            expanded += 1

        if expanded == 0:
            break

def generate_scenes(topic: str, duration_minutes: int = 1, format_ratio: Optional[str] = None, num_scenes_override: Optional[int] = None) -> Dict[str, Any]:
    """
    Gera roteiro estruturado com chunking/expansão opcional (multi-pass).
    """
    format_ratio = format_ratio or ASPECT or "16:9"
    if duration_minutes not in SCENE_COUNTS:
        duration_minutes = 1

    base_total_words = duration_minutes * WORDS_PER_MINUTE
    word_buffer = 1.08 if duration_minutes >= 5 else 1.0
    total_words = int(base_total_words * word_buffer)
    num_scenes = num_scenes_override or SCENE_COUNTS.get(duration_minutes, 10)
    words_per_scene = max(1, round(total_words / num_scenes))
    min_words_per_scene = max(18, int(words_per_scene * 0.8))
    max_words_per_scene = max(min_words_per_scene + 5, int(words_per_scene * 1.2))
    if duration_minutes >= 5:
        min_words_per_scene = max(min_words_per_scene, int(words_per_scene * 0.85))
        max_words_per_scene = max(max_words_per_scene, int(words_per_scene * 1.3))
    min_total_words = int(total_words * 0.98)
    max_total_words = int(total_words * 1.15)

    system_prompt = _build_system_prompt(format_ratio)

    def _generate_single() -> Dict[str, Any]:
        user_prompt = _build_user_prompt(
            topic,
            duration_minutes,
            format_ratio,
            num_scenes,
            total_words,
            min_words_per_scene,
            max_words_per_scene,
            min_total_words,
            max_total_words,
        )
        raw = _llm_generate(system_prompt, user_prompt)
        try:
            data = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001
            log.error("Não foi possível converter resposta em JSON: %s | raw=%r", exc, raw[:500])
            data = {"title": topic, "scenes": []}
        return _coerce_structure(data, topic, num_scenes)

    def _generate_chunked() -> Dict[str, Any]:
        chunk_size = max(1, int(SCRIPT_CHUNK_SCENES))
        total_parts = int(math.ceil(num_scenes / chunk_size))
        scenes: List[Dict[str, Any]] = []
        title: Optional[str] = None
        remaining_scenes = num_scenes
        remaining_min_words = min_total_words
        remaining_max_words = max_total_words

        for part_index in range(1, total_parts + 1):
            if remaining_scenes <= 0:
                break
            part_scenes = min(chunk_size, remaining_scenes)
            chunk_min, chunk_max, chunk_target = _compute_chunk_word_budget(
                remaining_scenes=remaining_scenes,
                remaining_min_words=remaining_min_words,
                remaining_max_words=remaining_max_words,
                chunk_scenes=part_scenes,
                words_per_scene=words_per_scene,
                min_words_per_scene=min_words_per_scene,
                max_words_per_scene=max_words_per_scene,
            )
            previous_context = scenes[-SCRIPT_CHUNK_CONTEXT_SCENES:] if SCRIPT_CHUNK_CONTEXT_SCENES > 0 else []
            user_prompt = _build_user_prompt_chunk(
                topic=topic,
                duration=duration_minutes,
                total_words=chunk_target,
                num_scenes=part_scenes,
                words_per_scene=words_per_scene,
                min_total_words=chunk_min,
                max_total_words=chunk_max,
                min_words_per_scene=min_words_per_scene,
                max_words_per_scene=max_words_per_scene,
                part_index=part_index,
                total_parts=total_parts,
                start_scene_index=len(scenes),
                previous_scenes=previous_context,
                fixed_title=title,
            )
            raw = _llm_generate(system_prompt, user_prompt)
            try:
                data = _parse_json(raw)
            except Exception as exc:  # noqa: BLE001
                log.error("Não foi possível converter parte %d em JSON: %s | raw=%r", part_index, exc, raw[:400])
                data = {"title": title or topic, "scenes": []}

            part_coerced = _coerce_structure(data, topic, part_scenes)
            part_scenes_list = part_coerced.get("scenes", [])[:part_scenes]
            if not title:
                title = part_coerced.get("title") or title
            scenes.extend(part_scenes_list)

            part_words = _count_words_in_scenes(part_scenes_list)
            remaining_min_words = max(0, remaining_min_words - part_words)
            remaining_max_words = max(0, remaining_max_words - part_words)
            remaining_scenes -= part_scenes

        script = {"title": title or topic, "scenes": scenes}
        if SCRIPT_EXPAND_ENABLED:
            _expand_script_to_min_words(
                script,
                min_total_words=min_total_words,
                max_total_words=max_total_words,
                min_words_per_scene=min_words_per_scene,
                max_words_per_scene=max_words_per_scene,
            )
        return script

    script = _generate_chunked() if SCRIPT_CHUNK_ENABLED and num_scenes > max(1, int(SCRIPT_CHUNK_SCENES)) else _generate_single()
    if NARRATION_POLISH_ENABLED:
        _polish_script_narration(script, NARRATION_POLISH_MAX_WORDS)
    return script


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gera roteiro usando LLM local ou compatÃ­vel.")
    parser.add_argument("tema", help="Tema ou tÃ³pico do vÃ­deo")
    parser.add_argument("--dur", type=int, default=1, choices=[1, 5], help="DuraÃ§Ã£o em minutos (1 ou 5)")
    parser.add_argument("--aspect", default=None, help="Formato (16:9 ou 9:16)")
    args = parser.parse_args()

    roteiro = generate_scenes(args.tema, duration_minutes=args.dur, format_ratio=args.aspect)
    print(json.dumps(roteiro, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
