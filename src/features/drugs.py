"""Drug-name normalization for DRUG / BIOLOGICAL interventions (exact matching only).

normalize_drug: lowercase; drop bracketed/parenthesized text; drop standalone
numbers and number+unit tokens; drop dose-unit, route, formulation, schedule and
salt words; strip punctuation; collapse whitespace. Alphanumeric code names
(e.g. "ly2189265") are kept intact. Placebo-like and generic names return None.
"""
import re

from src.features.build_features import PLACEBO_RE

DRUG_TYPES = {"DRUG", "BIOLOGICAL"}

UNITS = r"mg|mcg|µg|μg|ug|ng|g|kg|ml|l|iu|units?|u|%|mg/kg|mg/m2|mmol|meq"
NUM_TOKEN = re.compile(rf"^(\d+([.,]\d+)?)({UNITS})?(/(\w+))?$", re.IGNORECASE)
DROP_WORDS = set("""
oral orally po iv i.v intravenous intravenously subcutaneous subcutaneously sc s.c im intramuscular topical
inhaled inhalation intranasal nasal ophthalmic transdermal sublingual buccal rectal vaginal intravitreal
intrathecal intradermal injection injections injectable infusion bolus
tablet tablets tab tabs capsule capsules cap caps solution suspension cream gel ointment lotion patch spray
powder film coated film-coated extended-release extended release er xr xl sr cr ir dr mr modified depot
drops syrup granules sachet vial vials prefilled syringe pen autoinjector emulsion foam suppository
dose doses dosing daily once twice bid qd tid qid weekly monthly qw q2w q4w q3w low high mid
mg mcg µg μg ug ng g kg ml iu unit units mmol
hydrochloride hcl sodium potassium calcium mesylate maleate sulfate sulphate acetate citrate tartrate
besylate fumarate succinate phosphate bromide chloride
application administration administered formulation product given
""".split())
GENERIC = {"drug", "study drug", "study medication", "investigational product", "investigational drug", "active",
           "control", "standard care", "standard of care", "usual care", "chemotherapy", "medication", "treatment",
           "vaccine", "therapy", "drug therapy", "study treatment", "experimental", "active drug", "test product",
           "reference product", "comparator", "active comparator", "background therapy"}


def normalize_drug(name: str):
    if not isinstance(name, str):
        return None
    s = name.lower()
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", " ", s)
    s = re.sub(r"[®™©]", " ", s)
    toks = re.split(r"[\s,;:+/&]+", s)
    keep = []
    for t in toks:
        t = t.strip(".-_'\"*")
        if not t or t in DROP_WORDS or NUM_TOKEN.match(t):
            continue
        keep.append(t)
    s = re.sub(r"[^\w\s-]", " ", " ".join(keep))
    s = " ".join(s.split())
    if not s or s in GENERIC or PLACEBO_RE.search(s):
        return None
    return s


def arm_drugs(intervention_names) -> list:
    """Normalized DRUG/BIOLOGICAL names of one arm ('Drug: X|Biological: Y')."""
    if not isinstance(intervention_names, str):
        return []
    out = []
    for n in intervention_names.split("|"):
        if ":" not in n:
            continue
        typ, nm = n.split(":", 1)
        if typ.strip().upper().replace(" ", "_") not in DRUG_TYPES or PLACEBO_RE.search(nm):
            continue
        d = normalize_drug(nm)
        if d and d not in out:
            out.append(d)
    return out
