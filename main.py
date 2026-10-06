"""
Student Opportunity Hub
=======================
A database-driven web app for discovering and managing student opportunities
(internships, courses, hackathons and projects).

Stack: FastAPI + SQLAlchemy + SQLite + Pydantic. The whole frontend (HTML, CSS,
JavaScript) is served by FastAPI from this one file.

Run it:
    pip install -r requirements.txt
    uvicorn main:app --reload
Then open http://127.0.0.1:8000

Run the built-in database/API checks:
    python main.py --test

File layout (in order):
    1. Imports
    2. Database configuration
    3. Database models
    4. Pydantic schemas (validation)
    5. Database initialisation + sample data
    6. API routes + error handling
    7. HTML
    8. CSS
    9. JavaScript
   10. Self-test and entry point
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    String,
    Text,
    UniqueConstraint,
    case,
    create_engine,
    func,
    inspect,
    or_,
    select,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("opportunity_hub")
# =============================================================================
# 2. DATABASE CONFIGURATION
# =============================================================================
# The SQLite file is created next to main.py the first time the app runs.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "opportunity_hub.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},  # FastAPI uses several threads
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# Allowed values live in one place so the database, the API and the UI agree.
OPP_TYPES = ["Internship", "Course", "Hackathon", "Project"]
MODES = ["Remote", "On-site", "Hybrid"]
TypeName = Literal["Internship", "Course", "Hackathon", "Project"]
ModeName = Literal["Remote", "On-site", "Hybrid"]


def get_db():
    """Give each request its own database session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# 3. DATABASE MODELS
# =============================================================================
def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def sql_list(values) -> str:
    """Turn ['A', 'B'] into "'A', 'B'" for use inside a CHECK constraint."""
    return ", ".join("'" + v + "'" for v in values)


class Base(DeclarativeBase):
    pass


class Opportunity(Base):
    """One row = one opportunity a student can apply to or explore.

    A single table is enough here. Type, mode and category are simple values of
    the opportunity itself, so splitting them into extra tables would only add
    joins without making the app better.
    """

    __tablename__ = "opportunities"
    __table_args__ = (
        # The same opportunity can't be listed twice.
        UniqueConstraint(
            "title", "organization", "opportunity_type", name="uq_opportunity_identity"
        ),
        # The database itself refuses values the app doesn't understand.
        CheckConstraint(
            f"opportunity_type IN ({sql_list(OPP_TYPES)})", name="ck_opportunity_type"
        ),
        CheckConstraint(f"mode IN ({sql_list(MODES)})", name="ck_mode"),
        CheckConstraint("length(trim(title)) > 0", name="ck_title_not_blank"),
        CheckConstraint("length(trim(organization)) > 0", name="ck_org_not_blank"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(120))
    organization: Mapped[str] = mapped_column(String(100))
    opportunity_type: Mapped[str] = mapped_column(String(20), index=True)
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(60), index=True)
    location: Mapped[str] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(10))
    deadline: Mapped[date] = mapped_column(Date, index=True)
    application_link: Mapped[str] = mapped_column(String(500))
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # =============================================================================
# 4. PYDANTIC SCHEMAS (INPUT VALIDATION)
# =============================================================================
FIELD_LABELS = {
    "title": "Title",
    "organization": "Organization",
    "opportunity_type": "Type",
    "description": "Description",
    "category": "Category",
    "location": "Location",
    "mode": "Mode",
    "deadline": "Deadline",
    "application_link": "Application link",
}
FIELD_LIMITS = {
    "title": 120,
    "organization": 100,
    "category": 60,
    "location": 100,
    "description": 1000,
    "application_link": 500,
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def clean_url(value: str) -> str:
    """Accept only real-looking http(s) links."""
    if any(ch.isspace() for ch in value):
        raise ValueError("The link can't contain spaces.")
    try:
        parsed = urlparse(value)
        host = parsed.hostname or ""
    except ValueError:
        raise ValueError("Enter a full link, like https://example.com/apply")
    if parsed.scheme not in ("http", "https") or not host:
        raise ValueError("Enter a full link starting with http:// or https://")
    if "." not in host and host != "localhost":
        raise ValueError("That link doesn't look right. Check the domain name.")
    return value


def parse_deadline(value) -> date:
    """Turn 'YYYY-MM-DD' into a date, or explain what is wrong."""
    if isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Deadline is required.")
        if not DATE_RE.match(text):
            raise ValueError("Enter a valid date in YYYY-MM-DD format.")
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            raise ValueError("That date doesn't exist. Check the day and month.")
    else:
        raise ValueError("Deadline is required.")
    if not 2000 <= parsed.year <= 2100:
        raise ValueError("Enter a year between 2000 and 2100.")
    return parsed


class OpportunityIn(BaseModel):
    """Everything the client may send when creating or editing an opportunity."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    organization: str
    opportunity_type: str
    description: str
    category: str
    location: str
    mode: str
    deadline: date
    application_link: str

    @field_validator(
        "title", "organization", "description", "category", "location", "application_link"
    )
    @classmethod
    def text_fields(cls, value: str, info):
        label = FIELD_LABELS[info.field_name]
        if not value:
            raise ValueError(f"{label} is required.")
        limit = FIELD_LIMITS[info.field_name]
        if len(value) > limit:
            raise ValueError(f"{label} must be {limit} characters or fewer.")
        if info.field_name == "description" and len(value) < 20:
            raise ValueError("Add a little more detail (at least 20 characters).")
        if info.field_name == "application_link":
            return clean_url(value)
        return value

    @field_validator("opportunity_type")
    @classmethod
    def valid_type(cls, value: str):
        if value not in OPP_TYPES:
            raise ValueError("Choose a type: " + ", ".join(OPP_TYPES) + ".")
        return value

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, value: str):
        if value not in MODES:
            raise ValueError("Choose a mode: " + ", ".join(MODES) + ".")
        return value

    @field_validator("deadline", mode="before")
    @classmethod
    def valid_deadline(cls, value):
        return parse_deadline(value)


class OpportunityCreate(OpportunityIn):
    """New opportunities can't have a deadline that has already passed.
    (Editing is looser: an old, closed entry can still be updated.)"""

    @field_validator("deadline")
    @classmethod
    def not_in_past(cls, value: date):
        if value < date.today():
            raise ValueError("The deadline can't be in the past.")
        return value


class OpportunityOut(BaseModel):
    """What the API sends back."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    organization: str
    opportunity_type: str
    description: str
    category: str
    location: str
    mode: str
    deadline: date
    application_link: str
    is_sample: bool
    created_at: datetime
    updated_at: datetime
    # =============================================================================
# 5. DATABASE INITIALISATION + SAMPLE DATA
# =============================================================================
def sample_opportunities():
    """Demo entries shown on first run. Deadlines are relative to today so the
    page never starts out looking expired. Every link uses example.com, which
    is reserved for documentation and never points to a real company."""
    today = date.today()

    def entry(title, org, kind, desc, category, location, mode, days, slug):
        return Opportunity(
            title=title,
            organization=org,
            opportunity_type=kind,
            description=desc,
            category=category,
            location=location,
            mode=mode,
            deadline=today + timedelta(days=days),
            application_link=f"https://example.com/{slug}",
            is_sample=True,
        )

    return [
        entry(
            "Python Backend Internship",
            "Sample Tech Studio",
            "Internship",
            "Build and test REST APIs with FastAPI and SQL databases alongside a small engineering team. Good fit if you have finished a Python course.",
            "Backend Development", "Worldwide", "Remote", 21, "python-backend-internship",
        ),
        entry(
            "Web Development Internship",
            "Demo Digital Agency",
            "Internship",
            "Turn designs into responsive pages with HTML, CSS and JavaScript, and learn how a real project moves from idea to launch.",
            "Web Development", "Berlin, Germany", "Hybrid", 35, "web-development-internship",
        ),
        entry(
            "Data Analyst Internship",
            "Sample Analytics Lab",
            "Internship",
            "Clean datasets, build simple dashboards and present findings to a friendly team. Basic SQL or Python is enough to apply.",
            "Data Science", "Worldwide", "Remote", 12, "data-analyst-internship",
        ),
        entry(
            "AI/ML Hackathon",
            "Demo Campus Innovation Club",
            "Hackathon",
            "A 36-hour build event where small teams prototype a machine learning idea. Mentors are on hand and beginners are welcome.",
            "AI & Machine Learning", "University Main Hall", "On-site", 28, "ai-ml-hackathon",
        ),
        entry(
            "Build for Good Weekend",
            "Sample Civic Tech Collective",
            "Hackathon",
            "Spend a weekend building a useful tool for a local community group. Pick a brief, form a team and demo on Sunday.",
            "Web Development", "Online", "Remote", 9, "build-for-good-weekend",
        ),
        entry(
            "Cloud Computing Course",
            "Sample Learning Academy",
            "Course",
            "Eight self-paced lessons covering virtual machines, storage, networking and deploying a small web app to the cloud.",
            "Cloud Computing", "Online", "Remote", 60, "cloud-computing-course",
        ),
        entry(
            "Intro to SQL and Databases",
            "Demo Open Courseware",
            "Course",
            "Learn tables, joins and constraints by designing a small database from scratch. Includes practice exercises after every lesson.",
            "Data Science", "Online", "Remote", 45, "intro-to-sql",
        ),
        entry(
            "Frontend Development Project Challenge",
            "Sample Dev Community",
            "Project",
            "Recreate a landing page from a design brief, then get feedback from other learners. A solid portfolio piece.",
            "Web Development", "Online", "Remote", 18, "frontend-project-challenge",
        ),
        entry(
            "Open Source Contribution Sprint",
            "Demo Open Source Guild",
            "Project",
            "Make your first open source contribution with guidance. Maintainers pick beginner-friendly issues ahead of time.",
            "Open Source", "Online", "Remote", 5, "open-source-sprint",
        ),
    ]


def init_db(bind) -> None:
    """Create the tables if needed. Sample data is added only on the very first
    run, so entries you delete never come back after a restart."""
    first_run = not inspect(bind).has_table(Opportunity.__tablename__)
    Base.metadata.create_all(bind)
    if first_run:
        with Session(bind) as db:
            db.add_all(sample_opportunities())
            db.commit()


# =============================================================================
# 6. API ROUTES + ERROR HANDLING
# =============================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db(engine)
    except Exception:  # keep the server alive; requests will show a friendly error
        logger.exception("Could not initialise the database")
    yield


app = FastAPI(title="Student Opportunity Hub", lifespan=lifespan)


class FieldError(Exception):
    """Raised when a value is well-formed but not acceptable (e.g. a past date)."""

    def __init__(self, fields: dict):
        self.fields = fields


def error_response(status: int, message: str, fields: Optional[dict] = None):
    # Every error leaves the API in this shape, so the frontend handles one format.
    return JSONResponse(status_code=status, content={"error": message, "fields": fields or {}})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError):
    fields = {}
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", [])]
        if len(loc) < 2 or loc[0] == "path":
            continue
        name = loc[-1]
        kind = err.get("type", "")
        if kind == "json_invalid" or name.isdigit():
            continue
        if kind == "missing":
            label = FIELD_LABELS.get(name, "This field")
            message = f"{label} is required."
        elif kind == "value_error":
            message = err.get("msg", "").removeprefix("Value error, ")
        elif kind == "string_too_long":
            message = "That's too long."
        elif kind.startswith("literal"):
            message = "Choose one of the listed options."
        elif kind == "string_type":
            message = "Enter text for this field."
        else:
            message = "That value isn't valid."
        fields.setdefault(name, message)
    if fields:
        return error_response(422, "Please fix the highlighted fields and try again.", fields)
    return error_response(422, "That request couldn't be understood.")


@app.exception_handler(FieldError)
async def handle_field_error(_request: Request, exc: FieldError):
    return error_response(422, "Please fix the highlighted fields and try again.", exc.fields)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(_request: Request, exc: StarletteHTTPException):
    generic = {404: "We couldn't find that.", 405: "That action isn't supported."}
    if isinstance(exc.detail, str) and exc.detail not in ("Not Found", "Method Not Allowed"):
        message = exc.detail
    else:
        message = generic.get(exc.status_code, "Something went wrong. Please try again.")
    return error_response(exc.status_code, message)


@app.exception_handler(SQLAlchemyError)
async def handle_database_error(_request: Request, exc: SQLAlchemyError):
    logger.error("Database error: %s", exc)  # details stay in the server log
    return error_response(
        503, "We can't reach the database right now. Please try again in a moment."
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, exc: Exception):
    logger.exception("Unexpected error: %s", exc)
    return error_response(500, "Something went wrong. Please try again.")


def get_or_404(db: Session, opp_id: int) -> Opportunity:
    item = db.get(Opportunity, opp_id)
    if item is None:
        raise HTTPException(404, "That opportunity no longer exists. It may have been deleted.")
    return item


def like_pattern(text: str) -> str:
    """Build a LIKE pattern, treating % and _ typed by the user as plain text."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


DUPLICATE_MESSAGE = "This opportunity is already listed (same title, organization and type)."
# ---- READ (list with search, filters and sorting) ---------------------------
@app.get("/api/opportunities", response_model=list[OpportunityOut])
def list_opportunities(
    q: str = Query("", max_length=100),
    opportunity_type: Optional[TypeName] = Query(None, alias="type"),
    category: Optional[str] = Query(None, max_length=60),
    mode: Optional[ModeName] = None,
    sort: Literal["deadline", "newest"] = "deadline",
    db: Session = Depends(get_db),
):
    stmt = select(Opportunity)
    q = q.strip()
    if q:
        pattern = like_pattern(q)
        stmt = stmt.where(
            or_(
                Opportunity.title.ilike(pattern, escape="\\"),
                Opportunity.organization.ilike(pattern, escape="\\"),
            )
        )
    if opportunity_type:
        stmt = stmt.where(Opportunity.opportunity_type == opportunity_type)
    if category:
        stmt = stmt.where(func.lower(Opportunity.category) == category.strip().lower())
    if mode:
        stmt = stmt.where(Opportunity.mode == mode)

    if sort == "newest":
        stmt = stmt.order_by(Opportunity.created_at.desc(), Opportunity.id.desc())
    else:
        # Open deadlines first (soonest on top), closed ones at the end.
        closed = case((Opportunity.deadline < date.today(), 1), else_=0)
        stmt = stmt.order_by(closed, Opportunity.deadline, Opportunity.id)
    return db.scalars(stmt).all()


# ---- READ (numbers for the stats row, category list, closing-soon panel) ----
@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Opportunity.opportunity_type, func.count()).group_by(Opportunity.opportunity_type)
    ).all()
    counts = {kind: 0 for kind in OPP_TYPES}
    counts.update({kind: number for kind, number in rows})

    unique = {}
    for name in db.scalars(select(Opportunity.category)):
        unique.setdefault(name.lower(), name)
    categories = sorted(unique.values(), key=str.lower)

    soon = db.scalars(
        select(Opportunity)
        .where(Opportunity.deadline >= date.today())
        .order_by(Opportunity.deadline, Opportunity.id)
        .limit(3)
    ).all()
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "categories": categories,
        "closing_soon": [OpportunityOut.model_validate(o).model_dump(mode="json") for o in soon],
    }


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok"}


# ---- READ (one) --------------------------------------------------------------
@app.get("/api/opportunities/{opp_id}", response_model=OpportunityOut)
def get_opportunity(opp_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, opp_id)


# ---- CREATE ------------------------------------------------------------------
@app.post("/api/opportunities", response_model=OpportunityOut, status_code=201)
def create_opportunity(payload: OpportunityCreate, db: Session = Depends(get_db)):
    item = Opportunity(**payload.model_dump())
    db.add(item)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, DUPLICATE_MESSAGE)
    db.refresh(item)
    return item


# ---- UPDATE ------------------------------------------------------------------
@app.put("/api/opportunities/{opp_id}", response_model=OpportunityOut)
def update_opportunity(opp_id: int, payload: OpportunityIn, db: Session = Depends(get_db)):
    item = get_or_404(db, opp_id)
    # Allow editing an already-closed entry, but don't let anyone move a deadline into the past.
    if payload.deadline != item.deadline and payload.deadline < date.today():
        raise FieldError({"deadline": "The deadline can't be in the past."})
    for key, value in payload.model_dump().items():
        setattr(item, key, value)
    item.is_sample = False  # once you edit a demo entry it becomes yours
    item.updated_at = utcnow()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, DUPLICATE_MESSAGE)
    db.refresh(item)
    return item


# ---- DELETE ------------------------------------------------------------------
@app.delete("/api/opportunities/{opp_id}")
def delete_opportunity(opp_id: int, db: Session = Depends(get_db)):
    item = get_or_404(db, opp_id)
    db.delete(item)
    db.commit()
    return {"message": "Opportunity deleted.", "id": opp_id}


# ---- The website itself ------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE
# =============================================================================
# 7. HTML
# =============================================================================
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Opportunity Hub: internships, courses, hackathons and projects</title>
<meta name="description" content="Find internships, courses, hackathons and projects for students in one place.">
<meta name="theme-color" content="#FAF7FC">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%23B5316A'/%3E%3Ccircle cx='16' cy='16' r='7' fill='none' stroke='white' stroke-width='3'/%3E%3Ccircle cx='23' cy='9' r='3' fill='white'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Figtree:wght@400..700&display=swap" rel="stylesheet">
<style>
/*__CSS__*/
</style>
</head>
<body>
<a class="skip" href="#browse">Skip to opportunities</a>

<header class="nav" id="nav">
  <div class="wrap nav-inner">
    <a class="brand" href="#top" aria-label="Opportunity Hub home">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#B5316A"/><circle cx="16" cy="16" r="7" fill="none" stroke="#fff" stroke-width="3"/><circle cx="23" cy="9" r="3" fill="#fff"/></svg>
      <span>Opportunity Hub</span>
    </a>
    <button class="icon-btn menu-btn" id="menuBtn" type="button" aria-expanded="false" aria-controls="navLinks" aria-label="Open menu">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
    </button>
    <nav class="nav-links" id="navLinks" aria-label="Main">
      <a href="#browse">Browse</a>
      <a href="#soon">Closing soon</a>
      <button class="btn btn-primary" id="addBtn" type="button">Add opportunity</button>
    </nav>
  </div>
</header>

<main id="top">
  <section class="hero wrap">
    <div class="hero-copy">
      <h1>Find your next opportunity.</h1>
      <p class="lead">Internships, courses, hackathons and projects, collected in one place so you can stop hunting across ten tabs.</p>

      <form class="search" id="searchForm" role="search">
        <svg class="search-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <label class="sr" for="q">Search by title or organization</label>
        <input id="q" name="q" type="search" placeholder="Search by title or organization" autocomplete="off" maxlength="100">
        <button class="btn btn-primary" type="submit">Search</button>
      </form>

      <div class="chips" id="chips" role="group" aria-label="Filter by type"></div>
    </div>

    <aside class="soon" id="soon" aria-labelledby="soonTitle">
      <h2 id="soonTitle">Closing soon</h2>
      <ol class="soon-list" id="soonList"></ol>
    </aside>
  </section>

  <section class="browse wrap" id="browse" aria-labelledby="browseTitle">
    <div class="browse-head">
      <h2 id="browseTitle">Browse opportunities</h2>
      <div class="filters">
        <label class="sel"><span class="sr">Category</span>
          <select id="fCategory"><option value="">All categories</option></select>
        </label>
        <label class="sel"><span class="sr">Mode</span>
          <select id="fMode">
            <option value="">Any mode</option>
            <option>Remote</option><option>On-site</option><option>Hybrid</option>
          </select>
        </label>
        <label class="sel"><span class="sr">Sort by</span>
          <select id="fSort">
            <option value="deadline">Soonest deadline</option>
            <option value="newest">Newest added</option>
          </select>
        </label>
      </div>
    </div>

    <div class="results-bar">
      <p id="resultCount" aria-live="polite">Loading</p>
      <button type="button" class="link-btn" id="clearBtn" hidden>Clear filters</button>
    </div>

    <div class="grid" id="grid" aria-live="polite"></div>
  </section>
</main>

<footer class="footer">
  <div class="wrap">
    <p>Built with FastAPI and SQLite. Entries marked Sample are demo data added on first run.</p>
  </div>
</footer>

<!-- Add / edit dialog -->
<dialog id="formDialog" aria-labelledby="formTitle">
  <div class="sheet">
    <header class="sheet-head">
      <h2 id="formTitle">Add an opportunity</h2>
      <button type="button" class="icon-btn" id="formClose" aria-label="Close">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>
      </button>
    </header>
    <form id="oppForm" novalidate>
      <div class="form-grid">
        <div class="field full" data-field="title">
          <label for="f-title">Title</label>
          <input id="f-title" name="title" maxlength="120" autocomplete="off" placeholder="e.g. Python Backend Internship">
          <p class="err"></p>
        </div>
        <div class="field" data-field="organization">
          <label for="f-org">Organization</label>
          <input id="f-org" name="organization" maxlength="100" autocomplete="off" placeholder="Who is running it?">
          <p class="err"></p>
        </div>
        <div class="field" data-field="category">
          <label for="f-cat">Category</label>
          <input id="f-cat" name="category" maxlength="60" list="categoryList" autocomplete="off" placeholder="e.g. Web Development">
          <datalist id="categoryList"></datalist>
          <p class="err"></p>
        </div>
        <div class="field full" data-field="opportunity_type">
          <span class="label" id="lbl-type">Type</span>
          <div class="seg" role="radiogroup" aria-labelledby="lbl-type">
            <label><input type="radio" name="opportunity_type" value="Internship" checked><span>Internship</span></label>
            <label><input type="radio" name="opportunity_type" value="Course"><span>Course</span></label>
            <label><input type="radio" name="opportunity_type" value="Hackathon"><span>Hackathon</span></label>
            <label><input type="radio" name="opportunity_type" value="Project"><span>Project</span></label>
          </div>
          <p class="err"></p>
        </div>
        <div class="field full" data-field="mode">
          <span class="label" id="lbl-mode">Mode</span>
          <div class="seg" role="radiogroup" aria-labelledby="lbl-mode">
            <label><input type="radio" name="mode" value="Remote" checked><span>Remote</span></label>
            <label><input type="radio" name="mode" value="On-site"><span>On-site</span></label>
            <label><input type="radio" name="mode" value="Hybrid"><span>Hybrid</span></label>
          </div>
          <p class="err"></p>
        </div>
        <div class="field" data-field="location">
          <label for="f-loc">Location</label>
          <input id="f-loc" name="location" maxlength="100" autocomplete="off" placeholder="City, or Online">
          <p class="err"></p>
        </div>
        <div class="field" data-field="deadline">
          <label for="f-deadline">Deadline</label>
          <input id="f-deadline" name="deadline" type="date">
          <p class="err"></p>
        </div>
        <div class="field full" data-field="application_link">
          <label for="f-link">Application link</label>
          <input id="f-link" name="application_link" type="url" inputmode="url" maxlength="500" autocomplete="off" placeholder="https://">
          <p class="err"></p>
        </div>
        <div class="field full" data-field="description">
          <label for="f-desc">Description</label>
          <textarea id="f-desc" name="description" rows="4" maxlength="1000" placeholder="What is it, who is it for, and what will someone get out of it?"></textarea>
          <div class="field-foot"><p class="err"></p><span class="count" id="descCount">0 / 1000</span></div>
        </div>
      </div>
      <p class="form-error" id="formError" role="alert"></p>
      <footer class="sheet-foot">
        <button type="button" class="btn btn-ghost" id="formCancel">Cancel</button>
        <button type="submit" class="btn btn-primary" id="formSubmit">Add opportunity</button>
      </footer>
    </form>
  </div>
</dialog>

<!-- Delete confirmation -->
<dialog id="confirmDialog" class="small" aria-labelledby="confirmTitle">
  <div class="sheet">
    <h2 id="confirmTitle">Delete this opportunity?</h2>
    <p class="confirm-text">Are you sure you want to delete this opportunity? <strong id="confirmName"></strong> will be removed permanently.</p>
    <p class="form-error" id="confirmError" role="alert"></p>
    <footer class="sheet-foot">
      <button type="button" class="btn btn-ghost" id="confirmCancel">Keep it</button>
      <button type="button" class="btn btn-danger" id="confirmDelete">Delete</button>
    </footer>
  </div>
</dialog>

<div class="toasts" id="toasts"></div>

<script>
/*__JS__*/
</script>
</body>
</html>
"""
# =============================================================================
# 8. CSS
# =============================================================================
CSS = r"""
:root {
  --bg: #FAF7FC;
  --surface: #FFFFFF;
  --ink: #2A1A33;
  --muted: #6B5E75;
  --line: #E8DFEE;
  --line-strong: #D5C8DF;
  --brand: #B5316A;
  --brand-deep: #8F2054;
  --brand-soft: #FBE8F0;
  --danger: #B3261E;
  --danger-soft: #FCEBEA;
  --ok: #1F7A4D;

  --c-internship: #B5316A; --s-internship: #FBE8F0;
  --c-course: #1F7A74;     --s-course: #E1F2F0;
  --c-hackathon: #4B52C9;  --s-hackathon: #E8E9FA;
  --c-project: #9A5F0C;    --s-project: #FBEED6;

  --radius: 16px;
  --shadow: 0 1px 2px rgba(60, 30, 80, .06), 0 10px 28px -14px rgba(60, 30, 80, .22);
  --font-display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --font-body: "Figtree", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 84px; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-body);
  font-size: 16px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { font-family: var(--font-display); margin: 0; letter-spacing: -0.02em; line-height: 1.12; }
p { margin: 0; }
button, input, select, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
a { color: inherit; }
.wrap { max-width: 1120px; margin: 0 auto; padding-left: 22px; padding-right: 22px; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.skip { position: absolute; left: 12px; top: -60px; background: var(--ink); color: #fff; padding: 10px 14px; border-radius: 10px; z-index: 100; }
.skip:focus { top: 12px; }
:focus-visible { outline: 3px solid color-mix(in srgb, var(--brand) 55%, white); outline-offset: 2px; }

/* ---- Buttons ---- */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  border: 1.5px solid transparent; border-radius: 12px;
  padding: 10px 18px; font-weight: 600; font-size: 15px; line-height: 1.2;
  text-decoration: none; white-space: nowrap;
  transition: background-color .15s, border-color .15s, color .15s, transform .1s, box-shadow .15s;
}
.btn:active:not(:disabled) { transform: translateY(1px); }
.btn:disabled { opacity: .6; cursor: not-allowed; }
.btn-primary { background: var(--brand); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--brand-deep); }
.btn-ghost { background: transparent; border-color: var(--line-strong); color: var(--ink); }
.btn-ghost:hover:not(:disabled) { background: #F3ECF7; }
.btn-danger { background: var(--danger); color: #fff; }
.btn-danger:hover:not(:disabled) { background: #8E1D17; }
.btn-tint { background: var(--tint-soft, var(--brand-soft)); color: var(--tint, var(--brand)); padding: 9px 16px; }
.btn-tint:hover { background: var(--tint, var(--brand)); color: #fff; }
.btn.loading { position: relative; color: transparent; pointer-events: none; }
.btn.loading::after {
  content: ""; position: absolute; width: 18px; height: 18px; border-radius: 50%;
  border: 2.5px solid rgba(255, 255, 255, .45); border-top-color: #fff;
  animation: spin .7s linear infinite;
}
.icon-btn {
  display: inline-grid; place-items: center; width: 38px; height: 38px;
  border: 0; border-radius: 10px; background: transparent; color: var(--muted);
  transition: background-color .15s, color .15s;
}
.icon-btn:hover { background: rgba(42, 26, 51, .08); color: var(--ink); }
.icon-btn.danger:hover { background: var(--danger-soft); color: var(--danger); }
.link-btn { background: none; border: 0; padding: 4px 2px; color: var(--brand-deep); font-weight: 600; text-decoration: underline; text-underline-offset: 3px; }
.link-btn:hover { color: var(--brand); }

/* ---- Navigation ---- */
.nav { position: sticky; top: 0; z-index: 30; background: rgba(250, 247, 252, .92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
.nav-inner { display: flex; align-items: center; justify-content: space-between; height: 66px; }
.brand { display: flex; align-items: center; gap: 10px; text-decoration: none; font-family: var(--font-display); font-weight: 700; font-size: 20px; letter-spacing: -0.02em; }
.nav-links { display: flex; align-items: center; gap: 26px; }
.nav-links a { text-decoration: none; font-weight: 500; color: var(--muted); }
.nav-links a:hover { color: var(--ink); }
.menu-btn { display: none; }

/* ---- Hero ---- */
.hero { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, .85fr); gap: 48px; align-items: start; padding-top: 64px; padding-bottom: 44px; animation: rise .5s ease-out both; }
.hero h1 { font-size: clamp(2.5rem, 5.4vw, 4.2rem); font-weight: 700; letter-spacing: -0.035em; line-height: 1.02; max-width: 12ch; }
.lead { margin-top: 20px; font-size: 1.15rem; color: var(--muted); max-width: 46ch; }

.search {
  margin-top: 30px; display: flex; align-items: center; gap: 8px;
  background: var(--surface); border: 1.5px solid var(--line-strong); border-radius: 18px;
  padding: 8px 8px 8px 16px; box-shadow: var(--shadow);
  transition: border-color .15s, box-shadow .15s;
}
.search:focus-within { border-color: var(--brand); box-shadow: 0 0 0 4px rgba(181, 49, 106, .14), var(--shadow); }
.search-icon { color: var(--muted); flex: none; }
.search input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; padding: 10px 4px; font-size: 1.05rem; }
.search input::placeholder { color: #8F829A; }
.search .btn { padding: 12px 22px; }

.chips { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  --tint: var(--ink); --tint-soft: #F0E8F5;
  display: inline-flex; align-items: center; gap: 9px;
  background: var(--surface); border: 1.5px solid var(--line); border-radius: 999px;
  padding: 6px 7px 6px 13px; font-weight: 600; font-size: 14.5px;
  transition: border-color .15s, background-color .15s, color .15s;
}
.chip[data-type="Internship"] { --tint: var(--c-internship); --tint-soft: var(--s-internship); }
.chip[data-type="Course"]     { --tint: var(--c-course);     --tint-soft: var(--s-course); }
.chip[data-type="Hackathon"]  { --tint: var(--c-hackathon);  --tint-soft: var(--s-hackathon); }
.chip[data-type="Project"]    { --tint: var(--c-project);    --tint-soft: var(--s-project); }
.chip-n { min-width: 26px; text-align: center; padding: 1px 8px; border-radius: 999px; background: var(--tint-soft); color: var(--tint); font-size: 14px; }
.chip:hover { border-color: var(--tint); }
.chip[aria-pressed="true"] { background: var(--tint); border-color: var(--tint); color: #fff; }
.chip[aria-pressed="true"] .chip-n { background: rgba(255, 255, 255, .22); color: #fff; }

/* ---- Closing soon panel ---- */
.soon { background: var(--surface); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: var(--shadow); }
.soon h2 { font-size: 1.25rem; margin-bottom: 6px; }
.soon-list { list-style: none; margin: 0; padding: 0; }
.soon-item {
  display: grid; grid-template-columns: 10px 1fr auto; gap: 12px; align-items: center; width: 100%;
  text-align: left; background: none; border: 0; border-top: 1px solid var(--line);
  padding: 14px 6px; border-radius: 0; transition: background-color .15s;
}
.soon-list li:first-child .soon-item { border-top: 0; }
.soon-item:hover { background: #FBF7FD; }
.soon-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--tint); }
.soon-title { display: block; font-weight: 600; line-height: 1.25; }
.soon-org { display: block; font-size: 14px; color: var(--muted); }
.soon-when { font-size: 14px; font-weight: 600; color: var(--tint); background: var(--tint-soft); padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
[data-type="Internship"] { --tint: var(--c-internship); --tint-soft: var(--s-internship); }
[data-type="Course"]     { --tint: var(--c-course);     --tint-soft: var(--s-course); }
[data-type="Hackathon"]  { --tint: var(--c-hackathon);  --tint-soft: var(--s-hackathon); }
[data-type="Project"]    { --tint: var(--c-project);    --tint-soft: var(--s-project); }
.soon-empty { color: var(--muted); padding: 10px 0 4px; font-size: 15px; }

/* ---- Browse section ---- */
.browse { padding-top: 24px; padding-bottom: 80px; }
.browse-head { display: flex; align-items: end; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
.browse-head h2 { font-size: 1.9rem; }
.filters { display: flex; gap: 10px; flex-wrap: wrap; }
.sel select {
  appearance: none; background: var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1.5 6 6.5l5-5' fill='none' stroke='%236B5E75' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E") no-repeat right 14px center;
  border: 1.5px solid var(--line-strong); border-radius: 12px; padding: 10px 38px 10px 14px; font-weight: 500; min-width: 150px;
  transition: border-color .15s;
}
.sel select:hover { border-color: var(--brand); }
.results-bar { display: flex; align-items: center; justify-content: space-between; margin: 18px 0 16px; min-height: 28px; color: var(--muted); font-weight: 500; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 20px; transition: opacity .15s; }
.grid.busy { opacity: .55; pointer-events: none; }

/* ---- Cards ---- */
.card {
  display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); overflow: hidden;
  transition: transform .18s, box-shadow .18s, border-color .18s;
}
.card:hover { transform: translateY(-3px); box-shadow: var(--shadow); border-color: color-mix(in srgb, var(--tint) 45%, white); }
.card-top { display: flex; align-items: center; gap: 8px; padding: 10px 10px 10px 18px; background: var(--tint-soft); }
.pill { font-weight: 700; font-size: 14px; color: var(--tint); }
.sample { font-size: 12px; font-weight: 600; color: var(--muted); background: rgba(255, 255, 255, .75); border: 1px solid var(--line-strong); padding: 1px 8px; border-radius: 999px; }
.card-actions { margin-left: auto; display: flex; gap: 2px; }
.card-actions .icon-btn { width: 34px; height: 34px; color: var(--tint); opacity: .75; }
.card-actions .icon-btn:hover { opacity: 1; background: rgba(255, 255, 255, .75); }
.card-actions .icon-btn.danger:hover { color: var(--danger); }
.card-body { padding: 18px 18px 6px; flex: 1; }
.card h3 { font-size: 1.28rem; line-height: 1.2; }
.org { margin-top: 6px; color: var(--muted); font-weight: 500; }
.desc { margin-top: 12px; color: #4A3D54; font-size: 15px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.facts { list-style: none; margin: 14px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 14px; color: var(--muted); }
.facts li { display: inline-flex; align-items: center; gap: 6px; }
.card-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 18px 18px; }
.deadline { display: flex; flex-direction: column; font-size: 14px; color: var(--muted); line-height: 1.3; }
.deadline strong { color: var(--ink); font-size: 15px; }
.deadline.urgent strong { color: var(--brand-deep); }
.deadline.closed strong { color: var(--muted); text-decoration: line-through; text-decoration-thickness: 1px; }
.deadline span { display: inline-flex; align-items: center; gap: 6px; }

/* ---- States: skeleton, empty, error ---- */
.skeleton { height: 292px; border-radius: var(--radius); border: 1px solid var(--line); background: linear-gradient(100deg, #F3ECF7 30%, #FBF8FD 50%, #F3ECF7 70%); background-size: 300% 100%; animation: shimmer 1.4s linear infinite; }
.state { grid-column: 1 / -1; text-align: center; padding: 56px 20px; background: var(--surface); border: 1.5px dashed var(--line-strong); border-radius: 20px; }
.state svg { color: var(--brand); margin-bottom: 10px; }
.state h3 { font-size: 1.4rem; }
.state p { margin: 8px auto 20px; color: var(--muted); max-width: 42ch; }

/* ---- Dialogs ---- */
dialog { border: 0; padding: 0; background: transparent; width: min(640px, calc(100% - 28px)); max-height: min(92dvh, 900px); color: var(--ink); overflow: visible; }
dialog.small { width: min(440px, calc(100% - 28px)); }
dialog::backdrop { background: rgba(42, 26, 51, .5); backdrop-filter: blur(2px); }
dialog[open] { animation: pop .18s ease-out; }
.sheet { background: var(--surface); border-radius: 22px; padding: 26px; max-height: min(92dvh, 900px); overflow-y: auto; box-shadow: 0 30px 80px -20px rgba(42, 26, 51, .5); }
.sheet h2 { font-size: 1.5rem; }
.sheet-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.sheet-foot { display: flex; justify-content: flex-end; gap: 10px; margin-top: 18px; padding: 14px 0 26px; margin-bottom: -26px; position: sticky; bottom: -26px; background: var(--surface); border-top: 1px solid var(--line); }
.confirm-text { margin-top: 10px; color: var(--muted); }
.confirm-text strong { color: var(--ink); }

.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 14px; }
.field.full { grid-column: 1 / -1; }
.field label, .field .label { display: block; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
.field input:not([type="radio"]), .field textarea {
  width: 100%; background: var(--surface); border: 1.5px solid var(--line-strong); border-radius: 12px; padding: 11px 13px;
  transition: border-color .15s, box-shadow .15s;
}
.field textarea { resize: vertical; min-height: 104px; }
.field input:not([type="radio"]):focus, .field textarea:focus { outline: 0; border-color: var(--brand); box-shadow: 0 0 0 4px rgba(181, 49, 106, .14); }
.field.invalid input:not([type="radio"]), .field.invalid textarea { border-color: var(--danger); background: #FFFAF9; }
.err { color: var(--danger); font-size: 13.5px; font-weight: 500; min-height: 0; }
.err:not(:empty) { margin-top: 6px; }
.field-foot { display: flex; justify-content: space-between; gap: 10px; }
.count { color: var(--muted); font-size: 13px; margin-top: 6px; white-space: nowrap; margin-left: auto; }
.form-error { color: var(--danger); font-weight: 500; margin-top: 16px; }
.form-error:empty { display: none; }

.seg { display: flex; flex-wrap: wrap; gap: 8px; }
.seg label { margin: 0; font-weight: 500; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg span { display: inline-block; padding: 8px 15px; border: 1.5px solid var(--line-strong); border-radius: 999px; transition: background-color .15s, border-color .15s, color .15s; cursor: pointer; }
.seg span:hover { border-color: var(--brand); }
.seg input:checked + span { background: var(--brand-soft); border-color: var(--brand); color: var(--brand-deep); font-weight: 600; }
.seg input:focus-visible + span { outline: 3px solid color-mix(in srgb, var(--brand) 55%, white); outline-offset: 2px; }

/* ---- Toasts ---- */
.toasts { position: fixed; z-index: 200; right: 18px; bottom: 18px; display: grid; gap: 10px; width: min(380px, calc(100% - 36px)); pointer-events: none; }
.toast { pointer-events: auto; display: flex; align-items: flex-start; gap: 10px; background: var(--ink); color: #fff; padding: 13px 16px; border-radius: 14px; box-shadow: 0 14px 34px -12px rgba(42, 26, 51, .55); animation: toast-in .22s ease-out; font-weight: 500; }
.toast::before { content: ""; flex: none; width: 10px; height: 10px; margin-top: 7px; border-radius: 50%; background: #6EDDA3; }
.toast.error { background: #5A1411; }
.toast.error::before { background: #FF9C95; }
.toast.out { opacity: 0; transform: translateY(8px); transition: opacity .2s, transform .2s; }

.footer { border-top: 1px solid var(--line); padding: 26px 0 40px; color: var(--muted); font-size: 14px; }

/* ---- Motion ---- */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes pop { from { opacity: 0; transform: scale(.97) translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes toast-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { to { background-position: -300% 0; } }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
}

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; gap: 36px; padding-top: 40px; }
  .hero h1 { max-width: 14ch; }
}
@media (max-width: 720px) {
  .menu-btn { display: inline-grid; }
  .nav-links {
    display: none; position: absolute; left: 0; right: 0; top: 66px; flex-direction: column; align-items: stretch; gap: 4px;
    background: var(--bg); border-bottom: 1px solid var(--line); padding: 12px 22px 20px; box-shadow: 0 18px 30px -20px rgba(42, 26, 51, .3);
  }
  .nav.open .nav-links { display: flex; }
  .nav-links a { padding: 12px 4px; font-size: 1.05rem; }
  .nav-links .btn { margin-top: 6px; padding: 13px 18px; }
}
@media (max-width: 640px) {
  .wrap { padding-left: 16px; padding-right: 16px; }
  .search { flex-wrap: wrap; padding: 10px; border-radius: 16px; }
  .search-icon { display: none; }
  .search input { flex-basis: 100%; padding: 8px 6px; }
  .search .btn { width: 100%; }
  .filters { width: 100%; }
  .sel { flex: 1 1 140px; }
  .sel select { width: 100%; min-width: 0; }
  .grid { grid-template-columns: 1fr; }
  .form-grid { grid-template-columns: 1fr; }
  dialog { width: 100%; max-width: 100%; margin: auto 0 0; }
  .sheet { border-radius: 22px 22px 0 0; padding: 22px 18px 24px; }
  .sheet-foot { padding-bottom: 24px; margin-bottom: -24px; bottom: -24px; }
  .sheet-foot .btn { padding: 13px 18px; }
  .sheet-foot .btn-primary, .sheet-foot .btn-danger { flex: 1; }
  .soon { padding: 16px 16px 8px; }
  .soon-item > span:nth-child(2) { min-width: 0; }
  .soon-title, .soon-org { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card-actions .icon-btn { width: 40px; height: 40px; }
  .toasts { left: 18px; right: 18px; bottom: 14px; width: auto; }
}
"""
# =============================================================================
# 9. JAVASCRIPT
# =============================================================================
JS = r"""
'use strict';

// ---------- Setup ----------
const TYPES = ['Internship', 'Course', 'Hackathon', 'Project'];
const TYPE_PLURAL = { Internship: 'Internships', Course: 'Courses', Hackathon: 'Hackathons', Project: 'Projects' };
const CTA = { Internship: 'Apply now', Hackathon: 'Register', Course: 'View course', Project: 'View details' };

const ICON = {
  edit: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4"/></svg>',
  trash: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>',
  pin: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.8 12 21 12 21Z"/><circle cx="12" cy="9.5" r="2.5"/></svg>',
  mode: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="12" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  tag: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12V4h8l10 10-8 8L3 12Z"/><circle cx="7.5" cy="8.5" r="1"/></svg>',
  cal: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>',
  inbox: '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 13l3-8h12l3 8v6H3v-6Z"/><path d="M3 13h5l1 3h6l1-3h5"/></svg>'
};

const state = { q: '', type: '', category: '', mode: '', sort: 'deadline' };
let lastStats = null;
let editingId = null;
let pendingDelete = null;
let listRequest = 0;
let lastItems = null;
let searchTimer = null;

const $ = (id) => document.getElementById(id);
const el = {
  grid: $('grid'), chips: $('chips'), soonList: $('soonList'), count: $('resultCount'), clear: $('clearBtn'),
  q: $('q'), fCategory: $('fCategory'), fMode: $('fMode'), fSort: $('fSort'),
  formDialog: $('formDialog'), form: $('oppForm'), formTitle: $('formTitle'), formSubmit: $('formSubmit'),
  formError: $('formError'), descCount: $('descCount'), categoryList: $('categoryList'),
  confirmDialog: $('confirmDialog'), confirmName: $('confirmName'), confirmDelete: $('confirmDelete'), confirmError: $('confirmError'),
  toasts: $('toasts'), nav: $('nav'), menuBtn: $('menuBtn')
};

// ---------- Small helpers ----------
function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function parseDate(s) { const [y, m, d] = s.split('-').map(Number); return new Date(y, m - 1, d); }
function daysLeft(s) { const t = new Date(); t.setHours(0, 0, 0, 0); return Math.round((parseDate(s) - t) / 86400000); }
function fmtDate(s) { return parseDate(s).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }); }
function whenText(n) {
  if (n < 0) return 'Closed';
  if (n === 0) return 'Closes today';
  if (n === 1) return '1 day left';
  return n + ' days left';
}
function safeLink(url) { return /^https?:\/\//i.test(url) ? url : '#'; }
function todayISO() {
  const t = new Date();
  return t.getFullYear() + '-' + String(t.getMonth() + 1).padStart(2, '0') + '-' + String(t.getDate()).padStart(2, '0');
}

function toast(message, kind) {
  const node = document.createElement('div');
  node.className = 'toast' + (kind === 'error' ? ' error' : '');
  node.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  node.textContent = message;
  el.toasts.appendChild(node);
  setTimeout(() => { node.classList.add('out'); setTimeout(() => node.remove(), 250); }, kind === 'error' ? 6000 : 3800);
}

// ---------- Talking to the backend ----------
class ApiError extends Error {
  constructor(message, status, fields) { super(message); this.status = status; this.fields = fields || {}; }
}

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, options));
  } catch (e) {
    throw new ApiError("Can't reach the server. Check that it's still running, then try again.", 0);
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* body wasn't JSON */ }
  if (!res.ok) {
    throw new ApiError((data && data.error) || 'Something went wrong. Please try again.', res.status, data && data.fields);
  }
  return data;
}

// ---------- Rendering ----------
function renderChips() {
  if (!lastStats) return;
  const defs = [['', 'All', lastStats.total]].concat(TYPES.map((t) => [t, TYPE_PLURAL[t], lastStats.counts[t] || 0]));
  el.chips.innerHTML = defs.map(([value, label, n]) =>
    `<button type="button" class="chip" data-type="${value}" aria-pressed="${state.type === value}">` +
    `<span>${label}</span><span class="chip-n">${n}</span></button>`).join('');
}

function renderSoon() {
  const list = lastStats ? lastStats.closing_soon : [];
  if (!list.length) {
    el.soonList.innerHTML = '<li class="soon-empty">No upcoming deadlines yet. Add one and it will show up here.</li>';
    return;
  }
  el.soonList.innerHTML = list.map((o) =>
    `<li><button type="button" class="soon-item" data-type="${esc(o.opportunity_type)}" data-title="${esc(o.title)}">` +
    `<span class="soon-dot"></span>` +
    `<span><span class="soon-title">${esc(o.title)}</span><span class="soon-org">${esc(o.organization)}</span></span>` +
    `<span class="soon-when">${whenText(daysLeft(o.deadline))}</span></button></li>`).join('');
}

function renderCategories() {
  const cats = lastStats ? lastStats.categories : [];
  if (state.category && !cats.some((c) => c.toLowerCase() === state.category.toLowerCase())) state.category = '';
  el.fCategory.innerHTML = '<option value="">All categories</option>' +
    cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
  el.fCategory.value = state.category;
  el.categoryList.innerHTML = cats.map((c) => `<option value="${esc(c)}"></option>`).join('');
}

function cardHTML(o) {
  const left = daysLeft(o.deadline);
  const urgency = left < 0 ? 'closed' : left <= 7 ? 'urgent' : '';
  return `
  <article class="card" data-type="${esc(o.opportunity_type)}" data-id="${o.id}">
    <div class="card-top">
      <span class="pill">${esc(o.opportunity_type)}</span>
      ${o.is_sample ? '<span class="sample" title="Demo entry added with the starter data">Sample</span>' : ''}
      <div class="card-actions">
        <button type="button" class="icon-btn" data-action="edit" title="Edit" aria-label="Edit ${esc(o.title)}">${ICON.edit}</button>
        <button type="button" class="icon-btn danger" data-action="delete" title="Delete" aria-label="Delete ${esc(o.title)}">${ICON.trash}</button>
      </div>
    </div>
    <div class="card-body">
      <h3>${esc(o.title)}</h3>
      <p class="org">${esc(o.organization)}</p>
      <p class="desc">${esc(o.description)}</p>
      <ul class="facts">
        <li>${ICON.mode}<span>${esc(o.mode)}</span></li>
        <li>${ICON.pin}<span>${esc(o.location)}</span></li>
        <li>${ICON.tag}<span>${esc(o.category)}</span></li>
      </ul>
    </div>
    <div class="card-foot">
      <div class="deadline ${urgency}">
        <span>${ICON.cal}${fmtDate(o.deadline)}</span>
        <strong>${whenText(left)}</strong>
      </div>
      <a class="btn btn-tint" href="${esc(safeLink(o.application_link))}" target="_blank" rel="noopener noreferrer">${CTA[o.opportunity_type] || 'View'}</a>
    </div>
  </article>`;
}

function showSkeleton() {
  el.grid.innerHTML = '<div class="skeleton"></div>'.repeat(6);
}

function hasFilters() { return !!(state.q || state.type || state.category || state.mode); }

function renderList(items) {
  el.grid.classList.remove('busy');
  el.clear.hidden = !hasFilters();
  el.count.textContent = items.length + (items.length === 1 ? ' opportunity' : ' opportunities') + (hasFilters() ? ' match your filters' : '');
  if (!items.length) {
    const empty = lastStats && lastStats.total === 0;
    el.grid.innerHTML = `<div class="state">${ICON.inbox}` +
      (empty
        ? '<h3>No opportunities yet</h3><p>Add the first internship, course, hackathon or project to start your list.</p><button type="button" class="btn btn-primary" data-state="add">Add opportunity</button>'
        : '<h3>Nothing matches those filters</h3><p>Try a different keyword, or clear the filters to see everything.</p><button type="button" class="btn btn-ghost" data-state="clear">Clear filters</button>') +
      '</div>';
    return;
  }
  el.grid.innerHTML = items.map(cardHTML).join('');
}

function renderLoadError(message) {
  el.grid.classList.remove('busy');
  el.count.textContent = '';
  el.grid.innerHTML = `<div class="state">${ICON.inbox}<h3>We couldn't load opportunities</h3><p>${esc(message)}</p>` +
    '<button type="button" class="btn btn-primary" data-state="retry">Try again</button></div>';
}

// ---------- Loading data ----------
async function loadStats() {
  lastStats = await api('/api/stats');
  renderChips();
  renderSoon();
  renderCategories();
}

async function loadList(first) {
  const mine = ++listRequest;
  if (first) showSkeleton(); else el.grid.classList.add('busy');
  const params = new URLSearchParams();
  if (state.q) params.set('q', state.q);
  if (state.type) params.set('type', state.type);
  if (state.category) params.set('category', state.category);
  if (state.mode) params.set('mode', state.mode);
  params.set('sort', state.sort);
  try {
    const items = await api('/api/opportunities?' + params.toString());
    if (mine === listRequest) { lastItems = items; renderList(items); }
  } catch (err) {
    if (mine === listRequest) { lastItems = null; renderLoadError(err.message); }
  }
}

async function refreshAll(first) {
  const stats = loadStats().catch((err) => { if (!first) toast(err.message, 'error'); });
  await Promise.all([stats, loadList(first)]);
  // The empty-state wording depends on the totals, so draw the list again once both have arrived.
  if (lastItems) renderList(lastItems);
}

// ---------- Search and filters ----------
function applyFilters() { renderChips(); loadList(false); }

function resetFilters() {
  state.q = ''; state.type = ''; state.category = ''; state.mode = '';
  el.q.value = ''; el.fCategory.value = ''; el.fMode.value = '';
}
function clearFilters() { resetFilters(); applyFilters(); }

// ---------- Add / edit dialog ----------
function lockScroll(on) { document.body.style.overflow = on ? 'hidden' : ''; }

function clearFormErrors() {
  el.formError.textContent = '';
  el.form.querySelectorAll('.field').forEach((f) => { f.classList.remove('invalid'); f.querySelector('.err').textContent = ''; });
}

function showFieldErrors(fields) {
  Object.keys(fields || {}).forEach((name) => {
    const wrap = el.form.querySelector(`[data-field="${name}"]`);
    if (!wrap) return;
    wrap.classList.add('invalid');
    wrap.querySelector('.err').textContent = fields[name];
  });
  const first = el.form.querySelector('.field.invalid input, .field.invalid textarea');
  if (first) first.focus();
}

function updateCounter() { el.descCount.textContent = el.form.elements.description.value.length + ' / 1000'; }

function fillForm(o) {
  const f = el.form.elements;
  f.title.value = o ? o.title : '';
  f.organization.value = o ? o.organization : '';
  f.category.value = o ? o.category : '';
  f.location.value = o ? o.location : '';
  f.deadline.value = o ? o.deadline : '';
  f.application_link.value = o ? o.application_link : '';
  f.description.value = o ? o.description : '';
  f.opportunity_type.value = o ? o.opportunity_type : 'Internship';
  f.mode.value = o ? o.mode : 'Remote';
  // New opportunities can't have a past deadline. Existing ones keep whatever they had.
  f.deadline.min = o ? '' : todayISO();
  updateCounter();
}

function openForm(item) {
  editingId = item ? item.id : null;
  clearFormErrors();
  fillForm(item);
  el.formTitle.textContent = item ? 'Edit opportunity' : 'Add an opportunity';
  el.formSubmit.textContent = item ? 'Save changes' : 'Add opportunity';
  el.formDialog.showModal();
  lockScroll(true);
  el.form.elements.title.focus();
}

async function startEdit(id) {
  try {
    openForm(await api('/api/opportunities/' + id));
  } catch (err) {
    toast(err.message, 'error');
    if (err.status === 404) refreshAll(false);
  }
}

async function submitForm(event) {
  event.preventDefault();
  clearFormErrors();
  const body = Object.fromEntries(new FormData(el.form).entries());
  const editing = editingId !== null;
  el.formSubmit.classList.add('loading');
  el.formSubmit.disabled = true;
  try {
    await api(editing ? '/api/opportunities/' + editingId : '/api/opportunities', {
      method: editing ? 'PUT' : 'POST',
      body: JSON.stringify(body)
    });
    el.formDialog.close();
    toast(editing ? 'Changes saved.' : 'Opportunity added.');
    if (!editing) { resetFilters(); state.sort = 'newest'; el.fSort.value = 'newest'; renderChips(); }
    await refreshAll(false);
  } catch (err) {
    el.formError.textContent = err.message;
    showFieldErrors(err.fields);
    if (err.status === 404) refreshAll(false);
  } finally {
    el.formSubmit.classList.remove('loading');
    el.formSubmit.disabled = false;
  }
}

// ---------- Delete flow ----------
function askDelete(card) {
  pendingDelete = Number(card.dataset.id);
  el.confirmName.textContent = card.querySelector('h3').textContent;
  el.confirmError.textContent = '';
  el.confirmDialog.showModal();
  lockScroll(true);
  $('confirmCancel').focus();
}

async function confirmDelete() {
  el.confirmDelete.classList.add('loading');
  el.confirmDelete.disabled = true;
  try {
    await api('/api/opportunities/' + pendingDelete, { method: 'DELETE' });
    el.confirmDialog.close();
    toast('Opportunity deleted.');
    await refreshAll(false);
  } catch (err) {
    if (err.status === 404) { el.confirmDialog.close(); toast(err.message, 'error'); refreshAll(false); }
    else el.confirmError.textContent = err.message;
  } finally {
    el.confirmDelete.classList.remove('loading');
    el.confirmDelete.disabled = false;
  }
}

// ---------- Events ----------
$('searchForm').addEventListener('submit', (e) => {
  e.preventDefault();
  clearTimeout(searchTimer);
  state.q = el.q.value.trim();
  applyFilters();
  $('browse').scrollIntoView({ behavior: 'smooth' });
});
el.q.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.q = el.q.value.trim(); applyFilters(); }, 250);
});
el.chips.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  state.type = chip.dataset.type;
  applyFilters();
});
el.fCategory.addEventListener('change', () => { state.category = el.fCategory.value; applyFilters(); });
el.fMode.addEventListener('change', () => { state.mode = el.fMode.value; applyFilters(); });
el.fSort.addEventListener('change', () => { state.sort = el.fSort.value; applyFilters(); });
el.clear.addEventListener('click', clearFilters);

el.soonList.addEventListener('click', (e) => {
  const item = e.target.closest('.soon-item');
  if (!item) return;
  state.q = item.dataset.title; el.q.value = state.q;
  applyFilters();
  $('browse').scrollIntoView({ behavior: 'smooth' });
});

el.grid.addEventListener('click', (e) => {
  const stateBtn = e.target.closest('[data-state]');
  if (stateBtn) {
    const action = stateBtn.dataset.state;
    if (action === 'add') openForm(null);
    else if (action === 'clear') clearFilters();
    else refreshAll(true);
    return;
  }
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const card = btn.closest('.card');
  if (btn.dataset.action === 'edit') startEdit(card.dataset.id);
  else askDelete(card);
});

$('addBtn').addEventListener('click', () => { el.nav.classList.remove('open'); openForm(null); });
el.form.addEventListener('submit', submitForm);
el.form.addEventListener('input', (e) => {
  const wrap = e.target.closest('.field');
  if (wrap) { wrap.classList.remove('invalid'); wrap.querySelector('.err').textContent = ''; }
  if (e.target.name === 'description') updateCounter();
});
$('formClose').addEventListener('click', () => el.formDialog.close());
$('formCancel').addEventListener('click', () => el.formDialog.close());
$('confirmCancel').addEventListener('click', () => el.confirmDialog.close());
el.confirmDelete.addEventListener('click', confirmDelete);

// Clicking the dimmed backdrop closes a dialog; unlock scrolling whenever one closes.
[el.formDialog, el.confirmDialog].forEach((dlg) => {
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => lockScroll(false));
});

el.menuBtn.addEventListener('click', () => {
  const open = el.nav.classList.toggle('open');
  el.menuBtn.setAttribute('aria-expanded', String(open));
  el.menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
});
document.querySelectorAll('.nav-links a').forEach((a) => a.addEventListener('click', () => {
  el.nav.classList.remove('open');
  el.menuBtn.setAttribute('aria-expanded', 'false');
}));

// ---------- Go ----------
refreshAll(true);
"""

# The finished page: HTML with the CSS and JavaScript dropped in.
PAGE = HTML.replace("/*__CSS__*/", CSS).replace("/*__JS__*/", JS)
# =============================================================================
# 10. SELF-TEST AND ENTRY POINT
# =============================================================================
def run_self_test() -> int:
    """Exercise every endpoint against a throwaway database.
    Run with:  python main.py --test   (needs `httpx`, listed in requirements.txt)"""
    import tempfile

    from fastapi.testclient import TestClient

    folder = tempfile.mkdtemp()
    db_file = os.path.join(folder, "test.db")
    make_engine = lambda path: create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )
    test_engine = make_engine(db_file)
    SessionLocal.configure(bind=test_engine)
    init_db(test_engine)
    client = TestClient(app, raise_server_exceptions=False)

    failures = []

    def check(name, condition):
        print(("  PASS  " if condition else "  FAIL  ") + name)
        if not condition:
            failures.append(name)

    future = (date.today() + timedelta(days=30)).isoformat()
    good = {
        "title": "Test Internship",
        "organization": "Test Org",
        "opportunity_type": "Internship",
        "description": "A test description that is long enough to pass validation.",
        "category": "Testing",
        "location": "Online",
        "mode": "Remote",
        "deadline": future,
        "application_link": "https://example.com/apply",
    }

    print("Database and pages")
    check("home page is served", "Opportunity Hub" in client.get("/").text)
    listing = client.get("/api/opportunities")
    check("sample data was inserted on first run", listing.status_code == 200 and len(listing.json()) == 9)
    check("sample entries are flagged", all(o["is_sample"] for o in listing.json()))

    print("Create")
    created = client.post("/api/opportunities", json=good)
    check("POST returns 201", created.status_code == 201)
    new_id = created.json().get("id")
    check("new row has an id and created_at", bool(new_id) and bool(created.json().get("created_at")))
    check("duplicate is rejected with 409", client.post("/api/opportunities", json=good).status_code == 409)

    print("Read")
    check("GET one returns the row", client.get(f"/api/opportunities/{new_id}").json()["title"] == "Test Internship")
    check("list now has 10 rows", len(client.get("/api/opportunities").json()) == 10)
    stats = client.get("/api/stats").json()
    check("stats total matches", stats["total"] == 10 and stats["counts"]["Internship"] == 4)

    print("Update")
    changed = dict(good, title="Test Internship (edited)", mode="Hybrid")
    updated = client.put(f"/api/opportunities/{new_id}", json=changed)
    check("PUT returns 200", updated.status_code == 200)
    again = client.get(f"/api/opportunities/{new_id}").json()
    check("changes were saved", again["title"] == "Test Internship (edited)" and again["mode"] == "Hybrid")
    check("PUT on a missing id returns 404", client.put("/api/opportunities/99999", json=good).status_code == 404)

    print("Validation")
    for label, patch in [
        ("empty title", {"title": "   "}),
        ("short description", {"description": "too short"}),
        ("invalid url", {"application_link": "not a url"}),
        ("url without http", {"application_link": "ftp://example.com/x"}),
        ("impossible date", {"deadline": "2030-02-30"}),
        ("wrong date format", {"deadline": "12/05/2030"}),
        ("past date", {"deadline": "2001-01-01"}),
        ("bad type", {"opportunity_type": "Party"}),
        ("bad mode", {"mode": "Teleport"}),
    ]:
        res = client.post("/api/opportunities", json=dict(good, **patch))
        check(f"{label} is rejected", res.status_code == 422 and bool(res.json()["fields"]))
    missing = {k: v for k, v in good.items() if k != "organization"}
    check("missing field is rejected", client.post("/api/opportunities", json=missing).status_code == 422)
    check("nothing invalid was stored", len(client.get("/api/opportunities").json()) == 10)

    print("Search, filter and sort")
    found = client.get("/api/opportunities", params={"q": "python"}).json()
    check("search by title", len(found) == 1 and "Python" in found[0]["title"])
    found = client.get("/api/opportunities", params={"q": "campus innovation"}).json()
    check("search by organization", len(found) == 1)
    check("search treats % as text", client.get("/api/opportunities", params={"q": "%"}).json() == [])
    found = client.get("/api/opportunities", params={"type": "Hackathon"}).json()
    check("filter by type", len(found) == 2 and all(o["opportunity_type"] == "Hackathon" for o in found))
    found = client.get("/api/opportunities", params={"category": "web development"}).json()
    check("filter by category", len(found) == 3)
    found = client.get("/api/opportunities", params={"mode": "On-site"}).json()
    check("filter by mode", len(found) == 1)
    by_deadline = [o["deadline"] for o in client.get("/api/opportunities", params={"sort": "deadline"}).json()]
    check("sort by deadline", by_deadline == sorted(by_deadline))
    newest = client.get("/api/opportunities", params={"sort": "newest"}).json()
    check("sort by newest", newest[0]["id"] == new_id)
    check("bad filter value gives 422", client.get("/api/opportunities", params={"type": "Nope"}).status_code == 422)

    print("Delete")
    check("DELETE returns 200", client.delete(f"/api/opportunities/{new_id}").status_code == 200)
    gone = client.get(f"/api/opportunities/{new_id}")
    check("deleted row is gone (404)", gone.status_code == 404 and "error" in gone.json())
    check("deleting twice returns 404", client.delete(f"/api/opportunities/{new_id}").status_code == 404)

    print("Persistence (simulated restart)")
    kept = client.post("/api/opportunities", json=dict(good, title="Survives restart")).json()["id"]
    restarted = make_engine(db_file)
    SessionLocal.configure(bind=restarted)
    init_db(restarted)
    titles = [o["title"] for o in client.get("/api/opportunities").json()]
    check("data is still there after a restart", "Survives restart" in titles and len(titles) == 10)
    check("sample data is not re-added", len(titles) == 10)

    print("Error handling")
    SessionLocal.configure(bind=make_engine(os.path.join(folder, "missing-folder", "x.db")))
    broken = client.get("/api/opportunities")
    body = broken.text
    check("database failure returns 503", broken.status_code == 503)
    check("database failure message is friendly", "Please try again" in body and "sqlite" not in body.lower() and "Traceback" not in body)
    check("unknown API route is a clean 404", client.get("/api/nope").json().get("error") == "We couldn't find that.")
    SessionLocal.configure(bind=restarted)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_self_test())
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    