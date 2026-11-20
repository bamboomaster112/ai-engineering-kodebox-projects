"use strict";
// AI Desk UI: plain JavaScript, no build step. All model text is escaped before it is shown.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data;
}

// Read a text/event-stream response from a POST request and call onEvent for each JSON event.
async function stream(path, body, onEvent) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

// Minimal, safe Markdown: escape first, then headings, bullets, bold, http(s) links and [n] citations.
function md(text, citeHref) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\[(\d{1,3})\]/g, (m, n) => (citeHref ? `<a href="${citeHref(n)}">[${n}]</a>` : m));
  const out = [];
  let list = false;
  for (const raw of String(text || "").split("\n")) {
    const line = raw.trimEnd();
    const h = line.match(/^(#{1,3})\s+(.*)/);
    const li = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
    if (!li && list) { out.push("</ul>"); list = false; }
    if (h) out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`);
    else if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(li[1])}</li>`); }
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}

function busy(button, on, label) {
  button.disabled = on;
  if (label) button.textContent = label;
}

// ---- tabs --------------------------------------------------------------------------------
document.querySelectorAll(".tabs button").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
  if (btn.dataset.tab === "usage") loadUsage();
}));

// ---- status ------------------------------------------------------------------------------
async function loadHealth() {
  try {
    const h = await api("api/health");
    const kb = h.kb || {};
    $("#status").textContent = `${h.provider === "fake" ? "Offline fake model" : "KodeKey"} · main ${h.roles.main} · `
      + `${kb.documents || 0} help articles · ${kb.dense ? "hybrid" : "keyword"} retrieval · search: ${h.search}`;
    $("#health").textContent = JSON.stringify(h, null, 2);
  } catch (e) { $("#status").textContent = `API not reachable: ${e.message}`; }
}

// ---- support bot ---------------------------------------------------------------------------
let threadId = null;
try { threadId = sessionStorage.getItem("aidesk-thread"); } catch { /* storage unavailable */ }

function addMsg(cls, html) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerHTML = html;
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return div;
}

$("#chat-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addMsg("user", esc(text));
  const pending = addMsg("bot", '<span class="muted">Thinking…</span>');
  const btn = ev.submitter || $("#chat-form button");
  busy(btn, true);
  try {
    const out = await api("api/support/chat", { message: text, thread_id: threadId });
    threadId = out.thread_id;
    try { sessionStorage.setItem("aidesk-thread", threadId); } catch { /* ignore */ }
    let html = esc(out.answer);
    if (out.citations && out.citations.length) {
      html += '<div class="cites">' + out.citations.map((c) => `<span class="cite">[${c.n}] ${esc(c.title)} · ${esc(c.section)}</span>`).join("") + "</div>";
    }
    if (out.pii_masked && out.pii_masked.length) html += `<div class="muted">Masked before processing: ${esc(out.pii_masked.join(", "))}</div>`;
    html += `<details><summary>How this answer was made</summary><pre>${esc(JSON.stringify(out.trace, null, 2))}</pre></details>`;
    pending.innerHTML = html;
    if (out.blocked) pending.classList.add("blocked");
    if (out.ticket) loadTickets();
  } catch (e) {
    pending.innerHTML = `Error: ${esc(e.message)}`;
    pending.classList.add("blocked");
  } finally { busy(btn, false); }
});

$("#new-thread").addEventListener("click", () => {
  threadId = null;
  try { sessionStorage.removeItem("aidesk-thread"); } catch { /* ignore */ }
  $("#chat-log").innerHTML = "";
  addMsg("bot", "New conversation started. What can I help with?");
});

async function loadTickets() {
  try {
    const rows = await api("api/tickets");
    $("#tickets").innerHTML = rows.length
      ? rows.map((t) => `<li><strong>#${t.id}</strong> ${esc(t.subject)}<div class="muted">${esc(t.priority)} · ${new Date(t.created * 1000).toLocaleString()}</div></li>`).join("")
      : '<li class="muted">None yet</li>';
  } catch { /* ignore */ }
}
$("#refresh-tickets").addEventListener("click", loadTickets);

// ---- ask the web ---------------------------------------------------------------------------
function describeStep(d) {
  if (!d) return "";
  if (d.step === "agent") return d.tool_calls && d.tool_calls.length
    ? "Model chose: " + d.tool_calls.map((t) => `${t.name}(${esc(t.args.query || t.args.url || "")})`).join(", ")
    : "Model is writing the answer";
  if (d.step === "tools") return "Ran: " + d.calls.map((c) => c.error ? `${c.name} failed (${esc(c.error)})` : c.source_id ? `read source [${c.source_id}]` : `${c.name} → ${c.results} results`).join("; ");
  if (d.step === "finalize") return `Checked citations: ${d.cited.length} valid, ${d.invalid.length} removed, ${d.removed_untrusted_sentences} untrusted sentences dropped`;
  if (d.step === "plan") return "Plan: " + d.subquestions.map(esc).join(" | ");
  if (d.step === "approve") return d.by === "auto" ? "Plan auto-approved" : (d.approved ? "Plan approved" : "Plan rejected");
  if (d.step === "research") return `Researcher read ${d.pages} page(s) for: ${esc(d.subq)}`;
  if (d.step === "collect") return `Collected ${d.sources} distinct sources`;
  if (d.step === "critic") return d.skipped ? "Critic skipped (round limit)" : (d.sufficient ? "Critic: coverage is sufficient" : "Critic found gaps: " + d.gaps.map(esc).join(" | "));
  if (d.step === "write") return "Writer drafted the report";
  if (d.step === "verify") return `Verified citations: ${d.cited} cited, ${d.invalid.length} invalid removed`;
  return esc(d.step);
}

$("#web-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = ev.submitter || $("#web-form button");
  const steps = $("#web-steps"); const answer = $("#web-answer");
  steps.innerHTML = ""; answer.innerHTML = '<p class="muted">Working…</p>';
  busy(btn, true);
  try {
    await stream("api/web/ask", { question: $("#web-input").value.trim() }, (e) => {
      if (e.event === "step") steps.insertAdjacentHTML("beforeend", `<li>${describeStep(e.detail)}</li>`);
      if (e.event === "error") answer.innerHTML = `<p class="msg blocked">${esc(e.message)}</p>`;
      if (e.event === "done") {
        answer.innerHTML = md(e.answer, (n) => `#web-src-${n}`) + (e.sources.length
          ? '<h3>Sources</h3><ol class="sources">' + e.sources.map((s) => `<li id="web-src-${s.n}" value="${s.n}"><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a></li>`).join("") + "</ol>"
          : '<p class="muted">No sources were cited.</p>');
      }
    });
  } catch (e) { answer.innerHTML = `<p class="msg blocked">${esc(e.message)}</p>`; }
  finally { busy(btn, false); }
});

// ---- deep research ---------------------------------------------------------------------------
let researchRun = null;
function researchHandler(e) {
  if (e.event === "started" || e.event === "resumed") researchRun = e.run_id;
  if (e.event === "step") $("#research-steps").insertAdjacentHTML("beforeend", `<li>${describeStep(e.detail)}</li>`);
  if (e.event === "awaiting_approval") {
    $("#plan-text").value = e.plan.join("\n");
    $("#plan-box").hidden = false;
  }
  if (e.event === "error") $("#research-report").innerHTML = `<p class="msg blocked">${esc(e.message)}</p>`;
  if (e.event === "done") {
    $("#research-report").innerHTML = e.report ? md(e.report) : '<p class="muted">The run ended without a report.</p>';
    if (e.report && !e.citations_ok) $("#research-report").insertAdjacentHTML("afterbegin", '<p class="msg blocked">The citation check found problems in this report.</p>');
  }
}

$("#research-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = ev.submitter || $("#research-form button");
  $("#research-steps").innerHTML = ""; $("#research-report").innerHTML = ""; $("#plan-box").hidden = true;
  busy(btn, true);
  try { await stream("api/research/start", { question: $("#research-input").value.trim() }, researchHandler); }
  catch (e) { $("#research-report").innerHTML = `<p class="msg blocked">${esc(e.message)}</p>`; }
  finally { busy(btn, false); }
});

async function resumeResearch(approved) {
  const btn = $("#plan-run");
  const plan = $("#plan-text").value.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 4);
  $("#plan-box").hidden = true;
  busy(btn, true);
  try { await stream("api/research/resume", { run_id: researchRun, plan, approved }, researchHandler); }
  catch (e) { $("#research-report").innerHTML = `<p class="msg blocked">${esc(e.message)}</p>`; }
  finally { busy(btn, false); }
}
$("#plan-run").addEventListener("click", () => resumeResearch(true));
$("#plan-reject").addEventListener("click", () => resumeResearch(false));

// ---- playground ---------------------------------------------------------------------------
async function loadModels() {
  try {
    const models = await api("api/playground/models");
    $("#pg-models").innerHTML = models.map((m, i) => `<label><input type="checkbox" value="${esc(m)}" ${i < 3 ? "checked" : ""}>${esc(m)}</label>`).join("");
  } catch (e) { $("#pg-models").textContent = e.message; }
}
$("#pg-temp").addEventListener("input", () => { $("#pg-temp-out").textContent = $("#pg-temp").value; });

$("#pg-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = ev.submitter || $("#pg-form button[type=submit]");
  const models = [...document.querySelectorAll("#pg-models input:checked")].map((i) => i.value);
  $("#pg-results").innerHTML = ""; $("#pg-summary").textContent = "Running…";
  busy(btn, true);
  try {
    const out = await api("api/playground/compare", {
      prompt: $("#pg-prompt").value, system: $("#pg-system").value || null, models,
      temperature: Number($("#pg-temp").value), max_tokens: Number($("#pg-max").value), runs: Number($("#pg-runs").value),
    });
    $("#pg-summary").textContent = `Fastest: ${out.summary.fastest || "n/a"} · tokens used: ${out.summary.tokens} · errors: ${out.summary.errors}`;
    $("#pg-results").innerHTML = out.results.map((r) => `<div class="result ${r.error ? "err" : ""}">
      <header><strong>${esc(r.model)}${out.settings.runs > 1 ? ` · run ${r.run}` : ""}</strong>
      <span class="muted">${r.latency_ms} ms · ${r.prompt_tokens}+${r.completion_tokens} tok${r.cached ? " · cached" : ""}</span></header>
      <div class="text">${esc(r.error || r.text)}</div></div>`).join("");
  } catch (e) { $("#pg-summary").textContent = e.message; }
  finally { busy(btn, false); }
});

// ---- usage ----------------------------------------------------------------------------------
async function loadUsage() {
  try {
    const u = await api("api/usage");
    $("#usage-total").textContent = `${u.tokens.toLocaleString()} tokens of a ${u.daily_cap.toLocaleString()} daily cap`;
    $("#usage-rows").innerHTML = u.by_model_and_purpose.map((r) => `<tr><td>${esc(r.model)}</td><td>${esc(r.purpose)}</td><td>${r.calls}</td><td>${r.cached_calls}</td><td>${(r.tokens || 0).toLocaleString()}</td></tr>`).join("");
  } catch (e) { $("#usage-total").textContent = e.message; }
  loadHealth();
}
$("#refresh-usage").addEventListener("click", loadUsage);

loadHealth(); loadTickets(); loadModels();
