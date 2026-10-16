// Pathwise page. Builds every node with textContent, so nothing from an answer is ever parsed as HTML.

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
// Rule checks name the handbook rule they apply as [handbook:6.2]; show that the way a reader would.
const tidy = (t) => t.replace(/\[handbook:([\d.]+)\]/g, "(Handbook $1)");
const KIND = { course: "Catalog", handbook: "Handbook", record: "Student record", check: "Rule check, computed in code" };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

let meta = null;
let student = "";
let busy = false;

async function init() {
  meta = await (await fetch("/api/meta")).json();
  $("program").textContent = `${meta.program}, ${meta.institution}`;
  renderStudents();
  $("ask").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
  });
}

function renderStudents() {
  const box = $("students");
  box.replaceChildren();
  const options = [{ id: "", first: "No record" }, ...meta.students];
  options.forEach((s, i) => {
    const b = el("button", null, s.first);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(s.id === student));
    b.tabIndex = s.id === student ? 0 : -1;
    b.addEventListener("click", () => pick(s.id));
    b.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      pick(options[(i + step + options.length) % options.length].id);
      box.querySelector('[aria-checked="true"]').focus();
    });
    box.append(b);
  });
  const s = meta.students.find((x) => x.id === student);
  $("record").textContent = s ? `${s.name}. ${s.summary}` : "General questions about the catalog and handbook. Pick a student to get answers that apply their record.";
  const chips = $("suggestions");
  chips.replaceChildren();
  for (const q of meta.suggestions[student] || []) {
    const c = el("button", "chip", q);
    c.type = "button";
    c.addEventListener("click", () => { $("question").value = q; ask(); });
    chips.append(c);
  }
}

function pick(id) {
  if (id === student) return;
  student = id;
  renderStudents();
}

async function ask() {
  const question = $("question").value.trim();
  if (busy || question.length < 3) return;
  busy = true;
  $("ask-btn").disabled = true;
  $("ask-btn").textContent = "Asking";
  const card = $("answer");
  card.hidden = false;
  const status = el("div", "status");
  const line = el("span", null, "Searching the catalog and handbook");
  status.append(el("span", "pulse"), line);
  card.replaceChildren(status);
  const later = setTimeout(() => { line.textContent = "Writing a cited answer and checking every quote"; }, 1400);
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, student: student || null }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The question could not be answered.");
    render(body);
  } catch (err) {
    card.replaceChildren(el("p", "error", err.message || "Something went wrong. Try again."));
  } finally {
    clearTimeout(later);
    busy = false;
    $("ask-btn").disabled = false;
    $("ask-btn").textContent = "Ask";
  }
}

function render(a) {
  const card = $("answer");
  const cites = a.statements.flatMap((s) => s.citations);
  const verified = cites.filter((c) => c.verified).length;
  const unsupported = a.statements.filter((s) => !s.supported).length;
  const labels = Object.fromEntries(a.sources.map((s) => [s.id, s.label]));

  const head = el("div", "answer-head");
  head.append(el("h2", null, "Answer"));
  let badge;
  if (!a.answerable) badge = el("span", "badge plain", "Not covered by the sources");
  else if (unsupported) badge = el("span", "badge warn", `${unsupported} unsupported ${unsupported === 1 ? "statement" : "statements"}`);
  else if (verified < cites.length) badge = el("span", "badge warn", `${verified} of ${cites.length} quotes verified`);
  else badge = el("span", "badge ok", `${verified} of ${cites.length} quotes verified`);
  head.append(badge);

  const list = el("ol", "statements");
  a.statements.forEach((s, i) => {
    const li = el("li", s.supported ? null : "unsupported");
    li.append(document.createTextNode(s.text));
    const keys = [];
    const seen = new Set();
    s.citations.forEach((c, j) => {
      if (!c.verified) return;
      keys.push(`${i}-${j}`);
      if (seen.has(c.number)) return;
      seen.add(c.number);
      const fn = el("button", "fn", String(c.number));
      fn.type = "button";
      fn.dataset.n = c.number;
      fn.setAttribute("aria-label", `Source ${c.number}: ${labels[c.source]}`);
      fn.addEventListener("mouseenter", () => heat({ n: c.number, keys }));
      fn.addEventListener("focus", () => heat({ n: c.number, keys }));
      fn.addEventListener("mouseleave", cool);
      fn.addEventListener("blur", cool);
      fn.addEventListener("click", () => jump(c.number, keys));
      li.append(fn);
    });
    const missing = s.citations.length - keys.length;
    if (!s.supported) li.append(el("span", "flag", "No verified source"));
    else if (missing) li.append(el("span", "flag", `${missing} quote not found`));
    li.dataset.keys = keys.join(" ");
    li.addEventListener("mouseenter", () => heat({ keys, row: li }));
    li.addEventListener("mouseleave", cool);
    list.append(li);
  });

  const who = a.student ? meta.students.find((x) => x.id === a.student) : null;
  const metaLine = el("p", "meta",
    `Asked ${who ? `as ${who.first}` : "without a student record"}. ${a.sources.length} of ${a.considered.length} sources cited. ${a.seconds} s.`);
  card.replaceChildren(head, el("p", "question-echo", a.question), list, metaLine);
  renderSources(a);
  if (card.getBoundingClientRect().top > innerHeight - 120) {
    card.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
  }
}

function renderSources(a) {
  const box = $("sources");
  box.replaceChildren();
  $("src-count").textContent = `${a.sources.length} cited, ${a.considered.length} searched`;
  if (!a.sources.length) box.append(el("p", "empty", "No source was cited for this answer."));

  a.sources.forEach((src, idx) => {
    const n = idx + 1;
    const spans = [];
    a.statements.forEach((s, i) => s.citations.forEach((c, j) => {
      if (c.verified && c.source === src.id && c.span) spans.push({ a: c.span[0], b: c.span[1], key: `${i}-${j}` });
    }));
    const card = el("article", "src");
    card.id = `src-${n}`;
    const title = el("div", "src-title");
    const label = el("div", "src-label", src.label);
    label.append(el("span", "kind", KIND[src.kind] || src.kind));
    title.append(el("span", "num", String(n)), label);
    const text = el("p", "src-text");
    // Split the text at every quote boundary; each piece is marked with the citations that cover it.
    const cuts = [...new Set([0, src.text.length, ...spans.flatMap((s) => [s.a, s.b])])].sort((x, y) => x - y);
    for (let k = 0; k < cuts.length - 1; k++) {
      const [from, to] = [cuts[k], cuts[k + 1]];
      const piece = tidy(src.text.slice(from, to));
      const covering = spans.filter((s) => s.a <= from && s.b >= to).map((s) => s.key);
      if (!covering.length) { text.append(document.createTextNode(piece)); continue; }
      const m = el("mark", null, piece);
      m.dataset.keys = covering.join(" ");
      m.addEventListener("mouseenter", () => heat({ keys: covering }));
      m.addEventListener("mouseleave", cool);
      text.append(m);
    }
    card.append(title, text);
    box.append(card);
  });

  const cited = new Set(a.sources.map((s) => s.id));
  const rest = a.considered.filter((s) => !cited.has(s.id));
  if (rest.length) {
    const d = el("details", "also");
    d.append(el("summary", null, `Also searched, not cited (${rest.length})`));
    const ul = el("ul");
    for (const s of rest) ul.append(el("li", null, `${s.label}, ${(KIND[s.kind] || s.kind).toLowerCase()}`));
    d.append(ul);
    box.append(d);
  }
}

// Highlighting links both ways: a statement lights its quotes and sources, a quote lights its statements.
function heat({ keys, n, row }) {
  cool();
  const set = new Set(keys);
  document.querySelectorAll("mark[data-keys]").forEach((m) => {
    if (m.dataset.keys.split(" ").some((k) => set.has(k)) && (n == null || m.closest(".src").id === `src-${n}`)) {
      m.classList.add("hot");
      m.closest(".src").classList.add("hot");
    }
  });
  document.querySelectorAll(".statements li").forEach((li) => {
    if (li === row || (!row && n == null && li.dataset.keys.split(" ").some((k) => set.has(k)))) li.classList.add("hot");
  });
  if (n != null) document.querySelectorAll(`.fn[data-n="${n}"]`).forEach((f) => f.classList.add("hot"));
}

function cool() {
  document.querySelectorAll(".hot").forEach((x) => x.classList.remove("hot"));
}

function jump(n, keys) {
  const card = $(`src-${n}`);
  card.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "nearest" });
  heat({ n, keys });
}

init();
