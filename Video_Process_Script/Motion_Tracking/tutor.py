"""Concept questions for the marimo notebooks: the question, then the answer folded until the reader opens it."""


def question(mo, text: str, answer: str):
    """Question in a callout, answer in a folded block below it."""
    return mo.vstack([mo.callout(mo.md(f"**Question.** {text}"), kind="info"),
                      mo.accordion({"Voir la réponse (réfléchis d'abord)": mo.md(answer)})])
