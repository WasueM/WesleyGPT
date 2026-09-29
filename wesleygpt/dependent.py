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

A fact's question comes right after it or after real exchanges. The fact
kinds, values and acknowledgements are disjoint from wesleygpt.longcontext_eval's
recall items, so that eval measures transfer rather than repetition.

v2 had 13 hand-written fact kinds. It learned to recall those, even with values
it never saw, but answered "You haven't told me ..." for kinds it had not trained
on: 6% on the held-out recall eval (Think 64%). v3 generates ~160 kinds from
families (names of people and pets, favorites, numbers and codes, colors of
things, ages, jobs, where people live, appointment days and times), so the only
rule that fits them all is to copy what the user said. "unknown" is rarer and
only asks about a sibling of a fact that WAS given (told the cat's name, asked
the brother's), so refusing has to be a check rather than a default.
"""
import random
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


def _kind(statement, questions, answers, unknown, values, family="personal"):
    return {"statement": statement, "questions": questions, "answers": answers, "unknown": unknown,
            "values": values, "family": family}


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
ORIGINAL_KINDS = FACT_KINDS  # v2's kinds; longcontext_eval's trained-kind recall draws from these

# Disjoint from longcontext_eval's PETS, PEOPLE, CITIES and COLORS (tested).
GIVEN_NAMES = ["Juno", "Otis", "Wren", "Felix", "Hazel", "Jasper", "Ivy", "Milo", "Esme", "Rowan", "Tallulah", "Arlo",
               "Odette", "Cyrus", "Maren", "Ezra", "Sabine", "Anselm", "Delphine", "Quincy", "Lark", "Bram", "Cosima",
               "Emeric", "Fern", "Gideon", "Isolde", "Kip", "Leopold", "Nell", "Orla", "Percy", "Romy", "Silas", "Thea"]
OTHER_CITIES = ["Omaha", "Boise", "Savannah", "Lisbon", "Osaka", "Winnipeg", "Cork", "Perth", "Tampa", "Lyon",
                "Seville", "Tacoma", "Duluth", "Albuquerque", "Galway", "Kyoto"]
THING_COLORS = ["silver", "navy blue", "beige", "charcoal gray", "forest green", "burgundy", "cream", "bright orange",
                "sky blue", "gold", "coral", "black", "white", "yellow", "pink", "brown"]
MORE_JOBS = ["a paramedic", "a carpenter", "a dentist", "a chef", "a pilot", "a plumber", "a veterinarian", "an architect",
             "a firefighter", "a journalist", "a mail carrier", "a bank teller", "a physical therapist", "a farmer"]
FAVORITES = {
    "food": ["pizza", "ramen", "tacos", "lasagna", "pad thai", "sushi", "pho", "enchiladas"],
    "movie": ["Jaws", "Up", "Alien", "Casablanca", "Inception", "Coco", "Heat", "Amélie"],
    "song": ["Bohemian Rhapsody", "Hey Jude", "Dancing Queen", "Africa", "Wonderwall", "Clocks", "Hallelujah", "Superstition"],
    "season": ["spring", "summer", "autumn", "winter"],
    "sport": ["tennis", "rugby", "badminton", "lacrosse", "volleyball", "curling", "fencing", "cricket"],
    "animal": ["otter", "giraffe", "octopus", "penguin", "red panda", "axolotl", "hedgehog", "koala"],
    "fruit": ["mango", "kiwi", "papaya", "lychee", "blueberry", "pomegranate", "apricot", "guava"],
    "dessert": ["tiramisu", "cheesecake", "baklava", "crème brûlée", "flan", "churros", "gelato", "key lime pie"],
    "holiday": ["Thanksgiving", "Halloween", "Christmas", "Easter", "Hanukkah", "Diwali", "New Year's Eve", "the Fourth of July"],
    "board game": ["Catan", "Scrabble", "chess", "Monopoly", "Ticket to Ride", "Clue", "Risk", "Pandemic"],
    "band": ["Coldplay", "Queen", "ABBA", "Radiohead", "the Beatles", "Fleetwood Mac", "U2", "Imagine Dragons"],
    "flower": ["tulip", "sunflower", "lilac", "orchid", "peony", "daisy", "marigold", "iris"],
    "school subject": ["chemistry", "geometry", "history", "biology", "Spanish", "physics", "literature", "statistics"],
    "drink": ["lemonade", "hot chocolate", "chai", "root beer", "iced tea", "horchata", "apple cider", "sparkling water"],
    "ice cream flavor": ["pistachio", "mint chip", "rocky road", "butter pecan", "cookie dough", "strawberry", "coffee", "salted caramel"],
    "TV show": ["The Office", "Bluey", "Seinfeld", "Friends", "Planet Earth", "Parks and Recreation", "Survivor", "Jeopardy!"],
    "cereal": ["Cheerios", "Froot Loops", "Raisin Bran", "Lucky Charms", "Frosted Flakes", "Life", "Chex", "Corn Pops"],
    "video game": ["Minecraft", "Tetris", "Zelda", "Mario Kart", "Portal", "Stardew Valley", "Halo", "Celeste"],
    "dinosaur": ["triceratops", "stegosaurus", "velociraptor", "brachiosaurus", "ankylosaurus", "spinosaurus", "T. rex", "iguanodon"],
    "planet": ["Saturn", "Jupiter", "Neptune", "Mars", "Venus", "Mercury", "Uranus"],
    "word": ["serendipity", "petrichor", "mellifluous", "wanderlust", "luminous", "quixotic", "ephemeral", "sonder"],
    "bird": ["robin", "heron", "cardinal", "puffin", "hummingbird", "kingfisher", "magpie", "flamingo"],
    "tree": ["oak", "maple", "birch", "redwood", "willow", "aspen", "cedar", "sycamore"],
}
NAMED = ["cat", "hamster", "horse", "parrot", "goldfish", "rabbit", "turtle", "brother", "cousin", "boss", "roommate",
         "neighbor", "best friend", "grandmother", "grandfather", "uncle", "aunt", "nephew", "niece", "daughter", "son",
         "husband", "wife", "manager", "doctor", "mentor", "landlord", "coach", "piano teacher", "lab partner", "car", "boat"]
RELATIVES = ["mom", "dad", "brother", "aunt", "uncle", "cousin", "best friend", "grandmother", "grandfather", "daughter", "son"]
COLORED = ["car", "house", "bike", "backpack", "front door", "phone case", "couch", "kayak", "raincoat", "umbrella", "suitcase", "scooter"]
AGED = {"brother": (8, 40), "grandmother": (60, 99), "grandfather": (60, 99), "daughter": (1, 30), "son": (1, 30),
        "cat": (1, 18), "niece": (1, 25), "nephew": (1, 25), "cousin": (5, 50), "roommate": (18, 35), "car": (1, 25)}
EVENTS = ["haircut", "job interview", "piano lesson", "yoga class", "team meeting", "doctor's appointment",
          "parent-teacher conference", "book club", "flight home", "eye exam", "oil change", "volunteer shift"]


def _codes(name, make, n=40):
    """Up to `n` distinct generated values, the same every run: codes have no natural
    list. Fewer when the space is small (shoe sizes), instead of looping forever."""
    rng, seen = random.Random(name), []
    for _ in range(n * 50):
        v = make(rng)
        if v not in seen:
            seen.append(v)
        if len(seen) == n:
            break
    return seen


def _letters(rng, n):
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(n))


CODES = {
    "apartment number": lambda r: str(r.randint(100, 999)),
    "hotel room number": lambda r: str(r.randint(100, 1299)),
    "flight number": lambda r: f"{r.choice(['UA', 'DL', 'AA', 'WN', 'AS'])} {r.randint(100, 2999)}",
    "zip code": lambda r: f"{r.randint(10000, 99999)}",
    "bus route": lambda r: str(r.randint(1, 99)),
    "employee ID": lambda r: f"E-{r.randint(10000, 99999)}",
    "table number": lambda r: str(r.randint(1, 60)),
    "seat number": lambda r: f"{r.randint(1, 40)}{r.choice('ABCDEF')}",
    "order number": lambda r: str(r.randint(100000, 999999)),
    "jersey number": lambda r: str(r.randint(0, 99)),
    "parking spot": lambda r: f"{r.choice('ABCDEFG')}{r.randint(1, 60)}",
    "gate number": lambda r: f"{r.choice('ABCDE')}{r.randint(1, 40)}",
    "phone extension": lambda r: str(r.randint(1000, 9999)),
    "library card number": lambda r: str(r.randint(10000000, 99999999)),
    "student ID": lambda r: str(r.randint(100000000, 999999999)),
    "license plate": lambda r: f"{r.randint(1, 9)}{_letters(r, 3)} {r.randint(100, 999)}",
    "shoe size": lambda r: str(r.choice([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13])),
    "bike lock combination": lambda r: f"{r.randint(0, 9)}-{r.randint(0, 9)}-{r.randint(0, 9)}-{r.randint(0, 9)}",
    "house number": lambda r: str(r.randint(10, 9999)),
    "mailbox number": lambda r: str(r.randint(1, 400)),
    "confirmation code": lambda r: _letters(r, 6),
    "ticket number": lambda r: str(r.randint(1000, 99999)),
}


def _times(name):
    return _codes(name, lambda r: f"{r.randint(1, 12)}:{r.choice(['00', '15', '30', '45'])} {r.choice(['am', 'pm'])}", n=24)


def _generated_kinds():
    kinds = []
    for who in NAMED:
        kinds.append(_kind(f"my {who}'s name is {{v}}",
                           [f"What's my {who}'s name?", f"What did I say my {who}'s name is?", f"Do you remember my {who}'s name?"],
                           [f"Your {who}'s name is {{v}}.", f"You told me your {who}'s name is {{v}}."],
                           f"your {who}'s name", GIVEN_NAMES, "names"))
    for what, values in FAVORITES.items():
        kinds.append(_kind(f"my favorite {what} is {{v}}",
                           [f"What's my favorite {what}?", f"Which {what} did I say is my favorite?", f"Do you remember my favorite {what}?"],
                           [f"Your favorite {what} is {{v}}.", f"You said your favorite {what} is {{v}}."],
                           f"your favorite {what}", values, "favorites"))
    for what, make in CODES.items():
        kinds.append(_kind(f"my {what} is {{v}}",
                           [f"What's my {what}?", f"What did I say my {what} was?", f"Remind me, what's my {what}?"],
                           [f"Your {what} is {{v}}.", f"You told me your {what} is {{v}}."],
                           f"your {what}", _codes(what, make), "codes"))
    for what in COLORED:
        kinds.append(_kind(f"my {what} is {{v}}",
                           [f"What color is my {what}?", f"What color did I say my {what} is?"],
                           [f"Your {what} is {{v}}.", f"You said your {what} is {{v}}."],
                           f"what color your {what} is", THING_COLORS, "colors"))
    for who, (low, high) in AGED.items():
        kinds.append(_kind(f"my {who} is {{v}} years old",
                           [f"How old is my {who}?", f"How old did I say my {who} is?"],
                           [f"Your {who} is {{v}} years old."],
                           f"how old your {who} is", [str(a) for a in range(low, high + 1)], "ages"))
    for who in RELATIVES:
        kinds.append(_kind(f"my {who} works as {{v}}",
                           [f"What does my {who} do for work?", f"What's my {who}'s job?"],
                           [f"Your {who} works as {{v}}."],
                           f"what your {who} does for work", MORE_JOBS, "jobs"))
        kinds.append(_kind(f"my {who} lives in {{v}}",
                           [f"Where does my {who} live?", f"Which city did I say my {who} lives in?"],
                           [f"Your {who} lives in {{v}}."],
                           f"where your {who} lives", OTHER_CITIES, "places"))
    kinds.append(_kind("I grew up in {v}", ["Where did I grow up?", "Which city did I say I grew up in?"],
                       ["You grew up in {v}."], "where you grew up", OTHER_CITIES, "places"))
    for what in EVENTS:
        kinds.append(_kind(f"my {what} is on {{v}}",
                           [f"What day is my {what}?", f"When did I say my {what} is?"],
                           [f"Your {what} is on {{v}}."],
                           f"when your {what} is", WEEKDAYS, "days"))
        kinds.append(_kind(f"my {what} is at {{v}}",
                           [f"What time is my {what}?", f"What time did I say my {what} is?"],
                           [f"Your {what} is at {{v}}."],
                           f"what time your {what} is", _times(what), "times"))
    return kinds


FACT_KINDS = ORIGINAL_KINDS + _generated_kinds()
THINGS = ["marbles", "stamps", "trading cards", "seeds", "coins", "stickers", "pencils", "postcards", "cupcakes", "tomatoes"]

ACKS = ["Got it.", "Noted!", "Thanks for letting me know.", "Okay, I'll keep that in mind.", "Good to know!", "Noted, thanks.",
        "Okay!", "Sure, noted.", "Thanks, I'll remember.", "Understood.", "Good to know, thanks!", "Got it, thanks for sharing.",
        "Okay, noted.", "Thanks!"]
CORRECTION_ACKS = ["Thanks for the correction.", "Got it, updated.", "Okay, noted."]
# v2 saw only the first three and recalled only after them, so facts arrive many
# ways, including stated bare ({S} is the statement capitalized). None may match
# longcontext_eval's "Before we start, one thing to remember:" (tested).
PLANT_ALONE = ["Quick note: {s}.", "By the way, {s}.", "FYI, {s}.", "{S}.", "Just so you know, {s}.", "Oh, and {s}.",
               "Something about me: {s}.", "For context, {s}.", "Heads up: {s}.", "Fun fact: {s}.", "I should mention that {s}.",
               "In case it matters, {s}.", "Random, but {s}.", "Also, {s}.", "Can you keep this in mind? {S}.",
               "Worth knowing: {s}.", "A little about me: {s}.", "Keep in mind that {s}.", "Side note: {s}.", "Hey, {s}."]
PLANT_SUFFIX = ["\n\nBy the way, {s}.", "\n\nFYI, {s}.", "\n\nQuick note: {s}.", "\n\nAlso, {s}.", "\n\nOh, and {s}.",
                "\n\n{S}.", "\n\nSide note: {s}.", "\n\nFor context, {s}."]
BACKREF = {
    "math": (["What was the answer to the last math problem?", "Remind me, what was the final answer to the most recent math question?"],
             ["The answer was {v}.", "The final answer was {v}."]),
    "mmlu": (["Which letter did you choose for the last multiple-choice question?", "What was your answer to the most recent multiple-choice question?"],
             ["I chose {v}.", "My answer was {v}."]),
}
ADJACENT = 0.3  # share of recall questions asked right after their fact
KIND_WEIGHTS = {"fact": 0.48, "backref": 0.25, "compute": 0.20, "correction": 0.05, "unknown": 0.02}


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
        self.told = []  # (kind, first slot a question about it may follow)

    def insert(self, slot, exchange):
        self.after.setdefault(slot, []).append(exchange)

    def plant(self, rng, statement, allow_suffix=True, last_slot=None):
        """Place a statement no later than `last_slot`; returns the first slot a dependent
        turn may follow."""
        forms = {"s": statement, "S": statement[0].upper() + statement[1:]}
        if allow_suffix and self.k >= 3 and rng.random() < 0.5:
            slot = rng.randint(-1, self.k - 3)
            target = slot + 1
            if target not in self.suffix:
                self.suffix[target] = rng.choice(PLANT_SUFFIX).format(**forms)
                return target + 1
        slot = rng.randint(-1, self.k - 2 if last_slot is None else last_slot)
        self.insert(slot, _turn(rng.choice(PLANT_ALONE).format(**forms), rng.choice(ACKS)))
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
    plan.told.append((kind, first))
    # Sometimes straight after the fact: v2 always had a real exchange in between
    # and failed when asked right away.
    slot = first - 1 if rng.random() < ADJACENT else rng.randint(first, plan.k - 1)
    plan.insert(slot,
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
    """Ask for a sibling of a fact that was given, so "not told" has to be checked."""
    if not plan.told:
        _fact(plan, rng)
    told, first = rng.choice(plan.told)
    siblings = [k for k in FACT_KINDS if k["family"] == told["family"] and k["statement"] not in plan.used]
    kind = rng.choice(siblings) if siblings else _free_kind(plan, rng)
    plan.used.add(kind["statement"])
    plan.insert(rng.randint(first, plan.k - 1), _turn(rng.choice(kind["questions"]), f"You haven't told me {kind['unknown']} yet."))


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
