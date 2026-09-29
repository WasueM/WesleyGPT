# Wesley wrote this
"""Turns in a stitched conversation that depend on earlier turns.

Think-LongContext's stitched conversations never referred back, so it learned
that earlier turns are safe to ignore: recall of a turn-1 fact fell from Think's
9% to 2% after ~6 turns. v2 inserts turns whose right answer is only in the
history:

  fact        "By the way, I drive a Honda Civic." ... "What car do I drive?"
  correction  the same, with "Sorry, I misspoke earlier: ..." in between; the
              answer is the newer value
  unknown     asks for a fact never given: "You haven't told me ..."
  backref     "What was the answer to the last math problem?" / "Which letter
              did you choose for the last multiple-choice question?"
  compute     "I have 40 stamps." ... "If I get 7 more stamps, how many will I
              have?", answered with a <think> block like the math data

At least one real exchange always sits between a fact and its question. The fact
kinds, values and acknowledgements are disjoint from wesleygpt.longcontext_eval's
recall items, so that eval measures transfer rather than repetition.
"""
import re

NAMES = ["Amara", "Theo", "Rosalind", "Kenji", "Fatima", "Callum", "Noelle", "Rafael", "Signe", "Darius", "Imogen", "Mateo"]
CARS = ["Honda Civic", "Toyota Tacoma", "Subaru Outback", "Ford F-150", "Mazda Miata", "Tesla Model 3", "Jeep Wrangler", "Hyundai Elantra"]
INSTRUMENTS = ["cello", "trumpet", "banjo", "clarinet", "piano", "drums", "violin", "ukulele"]
LANGUAGES = ["Portuguese", "Japanese", "Swahili", "Korean", "German", "Italian", "Tagalog", "Arabic"]
ALLERGENS = ["peanuts", "shellfish", "penicillin", "cats", "bee stings", "gluten", "latex", "pollen"]
MAJORS = ["chemical engineering", "art history", "nursing", "economics", "linguistics", "computer science", "marine biology", "accounting"]
PROJECTS = ["Bluebird", "Orchard", "Lighthouse", "Tinderbox", "Paper Moon", "Northstar", "Copperleaf", "Driftwood"]
COUNTRIES = ["Portugal", "Vietnam", "Chile", "Morocco", "Iceland", "Kenya", "Peru", "Greece"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEAMS = ["Chicago Cubs", "Seattle Sounders", "Boston Celtics", "Green Bay Packers", "Utah Jazz", "Denver Broncos", "Toronto Maple Leafs", "LA Galaxy"]
BOOKS = ["Moby-Dick", "Pride and Prejudice", "The Odyssey", "Frankenstein", "Jane Eyre", "Dracula", "Middlemarch", "The Time Machine"]
JOBS = ["a nurse", "an electrician", "a teacher", "an accountant", "a pharmacist", "a welder", "a librarian", "a software engineer"]


def _kind(statement, questions, answers, unknown, values):
    return {"statement": statement, "questions": questions, "answers": answers, "unknown": unknown, "values": values}


FACT_KINDS = [
    _kind("my name is {v}", ["What's my name?", "Do you remember my name?"],
          ["Your name is {v}.", "You told me your name is {v}."], "your name", NAMES),
    _kind("I drive a {v}", ["What car do I drive?", "What kind of car do I have?"],
          ["You drive a {v}.", "You said you drive a {v}."], "what car you drive", CARS),
    _kind("I play the {v}", ["What instrument do I play?", "Which instrument did I say I play?"],
          ["You play the {v}."], "whether you play an instrument", INSTRUMENTS),
    _kind("I'm learning {v}", ["Which language am I learning?", "What language did I say I'm studying?"],
          ["You're learning {v}."], "which language you're learning", LANGUAGES),
    _kind("I'm allergic to {v}", ["What am I allergic to?", "Do you remember my allergy?"],
          ["You're allergic to {v}."], "about any allergies", ALLERGENS),
    _kind("I'm majoring in {v}", ["What's my major?", "What am I studying in school?"],
          ["You're majoring in {v}.", "Your major is {v}."], "your major", MAJORS),
    _kind("my project is called {v}", ["What's my project called?", "What did I say my project was named?"],
          ["Your project is called {v}."], "your project's name", PROJECTS),
    _kind("I'm traveling to {v} next month", ["Where am I traveling next month?", "Which country am I visiting?"],
          ["You're traveling to {v}."], "about any travel plans", COUNTRIES),
    _kind("my birthday is in {v}", ["When is my birthday?", "What month is my birthday?"],
          ["Your birthday is in {v}."], "when your birthday is", MONTHS),
    _kind("my dentist appointment is on {v}", ["What day is my dentist appointment?", "When did I say my dentist appointment was?"],
          ["Your dentist appointment is on {v}."], "about a dentist appointment", WEEKDAYS),
    _kind("I cheer for the {v}", ["Which team do I cheer for?", "What team did I say I support?"],
          ["You cheer for the {v}."], "which team you cheer for", TEAMS),
    _kind("I'm reading {v}", ["What book am I reading?", "Which book did I mention I'm reading?"],
          ["You're reading {v}."], "what you're reading", BOOKS),
    _kind("I work as {v}", ["What do I do for work?", "What's my job?"],
          ["You work as {v}."], "what you do for work", JOBS),
]
THINGS = ["marbles", "stamps", "trading cards", "seeds", "coins", "stickers", "pencils", "postcards", "cupcakes", "tomatoes"]

ACKS = ["Got it.", "Noted!", "Thanks for letting me know.", "Okay, I'll keep that in mind.", "Good to know!", "Noted, thanks."]
CORRECTION_ACKS = ["Thanks for the correction.", "Got it, updated.", "Okay, noted."]
PLANT_ALONE = ["Quick note: {s}.", "By the way, {s}.", "FYI, {s}."]
PLANT_SUFFIX = ["\n\nBy the way, {s}.", "\n\nFYI, {s}.", "\n\nQuick note: {s}."]
BACKREF = {
    "math": (["What was the answer to the last math problem?", "Remind me, what was the final answer to the most recent math question?"],
             ["The answer was {v}.", "The final answer was {v}."]),
    "mmlu": (["Which letter did you choose for the last multiple-choice question?", "What was your answer to the most recent multiple-choice question?"],
             ["I chose {v}.", "My answer was {v}."]),
}
KIND_WEIGHTS = {"fact": 0.45, "backref": 0.25, "compute": 0.20, "correction": 0.05, "unknown": 0.05}


def _text(message):
    c = message["content"]
    return c if isinstance(c, str) else "".join(p["text"] for p in c)


def _answer_of(exchange):
    """('math', final number) or ('mmlu', letter) if the exchange is one, else None."""
    reply = _text(exchange[-1]).strip()
    found = re.search(r"####\s*(\S+)$", reply)
    if found:
        return "math", found.group(1)
    if _text(exchange[0]).startswith("Multiple Choice question") and reply in {"A", "B", "C", "D"}:
        return "mmlu", reply
    return None


def _turn(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class _Plan:
    """Insertions to make, by the index of the segment they follow (-1 = before the first).
    A segment's last reply is what a back-reference reads, its first user turn is where a
    fact can be appended."""

    def __init__(self, exchanges):
        self.exchanges, self.k = exchanges, len(exchanges)
        self.after, self.suffix, self.used = {}, {}, set()

    def insert(self, slot, exchange):
        self.after.setdefault(slot, []).append(exchange)

    def plant(self, rng, statement, allow_suffix=True, last_slot=None):
        """Place a statement no later than `last_slot`; returns the first slot a dependent
        turn may follow."""
        if allow_suffix and self.k >= 3 and rng.random() < 0.5:
            slot = rng.randint(-1, self.k - 3)
            target = slot + 1
            if target not in self.suffix:
                self.suffix[target] = rng.choice(PLANT_SUFFIX).format(s=statement)
                return target + 1
        slot = rng.randint(-1, self.k - 2 if last_slot is None else last_slot)
        self.insert(slot, _turn(rng.choice(PLANT_ALONE).format(s=statement), rng.choice(ACKS)))
        return slot + 1

    def assemble(self):
        out = []
        for slot in range(-1, self.k):
            if slot >= 0:
                exchange = [dict(m) for m in self.exchanges[slot]]
                if slot in self.suffix:
                    exchange[0]["content"] = exchange[0]["content"] + self.suffix[slot]
                out.append(exchange)
            out.extend(self.after.get(slot, []))
        latest = {}
        for exchange in out:  # back-references answer from what actually precedes them
            if exchange[-1]["content"] is None:
                kind, answers = exchange[-1].pop("backref")
                exchange[-1]["content"] = answers.format(v=latest[kind])
            found = _answer_of(exchange)
            if found:
                latest[found[0]] = found[1]
        return [m for exchange in out for m in exchange]


def _free_kind(plan, rng):
    free = [k for k in FACT_KINDS if k["statement"] not in plan.used]
    kind = rng.choice(free)
    plan.used.add(kind["statement"])
    return kind


def _fact(plan, rng):
    kind = _free_kind(plan, rng)
    value = rng.choice(kind["values"])
    first = plan.plant(rng, kind["statement"].format(v=value))
    plan.insert(rng.randint(first, plan.k - 1),
                _turn(rng.choice(kind["questions"]), rng.choice(kind["answers"]).format(v=value)))


def _correction(plan, rng):
    kind = _free_kind(plan, rng)
    old, new = rng.sample(kind["values"], 2)
    first = plan.plant(rng, kind["statement"].format(v=old), allow_suffix=False, last_slot=plan.k - 3)  # room for both
    fix = rng.randint(first, plan.k - 2)
    plan.insert(fix, _turn(f"Sorry, I misspoke earlier: {kind['statement'].format(v=new)}.", rng.choice(CORRECTION_ACKS)))
    plan.insert(rng.randint(fix + 1, plan.k - 1),
                _turn(rng.choice(kind["questions"]), rng.choice(kind["answers"]).format(v=new)))


def _unknown(plan, rng):
    kind = _free_kind(plan, rng)
    plan.insert(rng.randint(0, plan.k - 1), _turn(rng.choice(kind["questions"]), f"You haven't told me {kind['unknown']} yet."))


def _backref(plan, rng):
    first_of = {}
    for i, exchange in enumerate(plan.exchanges):
        found = _answer_of(exchange)
        if found:
            first_of.setdefault(found[0], i)
    if not first_of:
        return _fact(plan, rng)
    kind = rng.choice(sorted(first_of))
    questions, answers = BACKREF[kind]
    plan.insert(rng.randint(first_of[kind], plan.k - 1),
                [{"role": "user", "content": rng.choice(questions)},
                 {"role": "assistant", "content": None, "backref": (kind, rng.choice(answers))}])


def _compute(plan, rng):
    things, n = rng.choice(THINGS), rng.randint(5, 200)
    first = plan.plant(rng, f"I have {n} {things}")
    op = rng.choice(["more", "away", "times"])
    if op == "more":
        m = rng.randint(2, 50)
        question, work, r = f"If I get {m} more {things}, how many will I have?", f"{n} + {m} = {n + m}", n + m
    elif op == "away":
        m = rng.randint(1, n - 1)
        question, work, r = f"If I give away {m} of my {things}, how many will I have left?", f"{n} - {m} = {n - m}", n - m
    else:
        m = rng.randint(2, 5)
        question, work, r = f"If I had {m} times as many {things}, how many would that be?", f"{n} * {m} = {n * m}", n * m
    answer = f"<think>\nYou have {n} {things}.\n{work}\n</think>\n#### {r}"
    plan.insert(rng.randint(first, plan.k - 1), _turn(question, answer))


BUILDERS = {"fact": _fact, "correction": _correction, "unknown": _unknown, "backref": _backref, "compute": _compute}


def insert_dependencies(segments, rng, n, kinds=None):
    """The messages of stitched `segments` (one per dataset draw, as wesleygpt.stitch
    returns them) with `n` dependent turns inserted between segments, never inside
    one: a multi-turn SmolTalk chat's follow-ups refer to its own earlier turns.
    Fewer than two segments come back unchanged.

    kinds: which builders to draw from (default: all, by KIND_WEIGHTS).
    """
    if len(segments) < 2 or n <= 0:
        return [m for segment in segments for m in segment]
    plan = _Plan(segments)
    names = kinds or list(KIND_WEIGHTS)
    weights = [KIND_WEIGHTS[k] for k in names]
    for _ in range(n):
        BUILDERS[rng.choices(names, weights=weights)[0]](plan, rng)
    return plan.assemble()
