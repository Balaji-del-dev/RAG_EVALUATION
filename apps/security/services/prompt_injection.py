"""Rule-based detection of prompt injection and jailbreak attempts in user input.

Rules are fast, cost no memory and explain themselves, but paraphrases will get past them. That is
exactly what the security evaluation measures: guard_false_negative_rate is the share that got through.
"""

import re
from dataclasses import dataclass

INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|context)\b",
    "reveal_prompt": r"\b(reveal|show|print|repeat|output|tell me|what (is|are))\b.{0,40}\b(system prompt|hidden prompt|initial prompt|your (instructions|prompt|rules))\b",
    "new_instructions": r"\b(new|updated|real) (instructions?|system prompt|rules?)\s*[:\-]",
    "role_markers": r"(?:^|\n)\s*(system|assistant)\s*:|<\|?(system|im_start)\|?>|\[/?INST\]",
    "say_exactly": r"\b(respond|reply|answer|say|output)\b.{0,25}\b(only|exactly|just)\b.{0,15}\bwith\b|\bsay exactly\b",
}

JAILBREAK_PATTERNS = {
    "dan_persona": r"\b(DAN|do anything now|developer mode|god mode|jailbreak(ed)?|unfiltered mode)\b",
    "no_restrictions": r"\b(no|without|free (of|from)|remove (all|your))\b.{0,20}\b(restrictions?|filters?|limitations?|censorship|ethical guidelines|safety)\b",
    "pretend_persona": r"\b(pretend|act|behave|roleplay|role-play)\b.{0,30}\b(you are|you're|as|like)\b.{0,40}\b(no rules|evil|unrestricted|without (any )?(rules|limits)|hacker)\b",
    "hypothetical_bypass": r"\b(hypothetically|for a novel|for a story|in a fictional)\b.{0,60}\b(how to|steps to|hack|steal|bypass|exploit)\b",
}


def _compile(patterns: dict[str, str]) -> dict[str, re.Pattern]:
    return {name: re.compile(pattern, re.IGNORECASE | re.DOTALL) for name, pattern in patterns.items()}


COMPILED_INJECTION = _compile(INJECTION_PATTERNS)
COMPILED_JAILBREAK = _compile(JAILBREAK_PATTERNS)


@dataclass
class InjectionVerdict:
    flagged: bool
    category: str = ""
    rule: str = ""


def scan_input(text: str) -> InjectionVerdict:
    for category, patterns in (("prompt_injection", COMPILED_INJECTION), ("jailbreak", COMPILED_JAILBREAK)):
        for rule, pattern in patterns.items():
            if pattern.search(text):
                return InjectionVerdict(True, category, rule)
    return InjectionVerdict(False)
