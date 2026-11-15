/* Shared front-end logic: form submission, drag-and-drop, result rendering. */
"use strict";

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const charts = [];

function verdictClass(v) {
  if (v === "LIKELY FAKE") return "verdict-fake";
  if (v === "LIKELY REAL") return "verdict-real";
  if (v === "UNCERTAIN") return "verdict-unc";
  return "verdict-none";
}
function verdictBadge(v) {
  const cls = v === "LIKELY FAKE" ? "bg-fake" : v === "LIKELY REAL" ? "bg-real" : v === "UNCERTAIN" ? "bg-unc" : "bg-secondary";
  return `<span class="badge badge-verdict ${cls}">${esc(v)}</span>`;
}

function scoreBar(score) {
  if (score === null || score === undefined) return "";
  return `<div class="score-bar mt-2"><div class="marker" style="left:${(score * 100).toFixed(1)}%"></div></div>
          <div class="score-scale mt-1"><span>0 · real-like</span><span>0.35</span><span>0.5</span><span>0.65</span><span>fake-like · 1</span></div>`;
}

function list(items, cls) {
  if (!items || !items.length) return "";
  return `<ul class="mb-0 ${cls}">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}

function metricLine(m) {
  if (!m || m.roc_auc === undefined) return "";
  return `<div class="small text-secondary mt-1">Held-out test of this model: accuracy ${fmt(m.accuracy)}, F1(fake) ${fmt(m.f1_fake)}, ROC-AUC ${fmt(m.roc_auc)} (n=${m.n}).</div>`;
}

function evidenceImg(src, caption) {
  if (!src) return "";
  return `<div class="col-md-6 col-lg-4"><img class="evidence-img" src="${src}" alt="${esc(caption)}"><div class="evidence-caption">${esc(caption)}</div></div>`;
}

/* ---------------------------------------------------------------- panels */
function textPanel(t) {
  const chips = (arr, cls) => (arr || []).map((x) => `<span class="term-chip ${cls}">${esc(x.term)} <span class="mono">${fmt(x.weight, 3)}</span></span>`).join("") || '<span class="text-secondary small">none</span>';
  const s = t.style_stats || {};
  return `<div class="card mb-3"><div class="card-header"><i class="bi bi-fonts me-1"></i>Text analysis <span class="text-secondary fw-normal small">· ${esc(t.model)}</span></div>
    <div class="card-body">
      <div class="d-flex flex-wrap gap-3 align-items-baseline mb-2"><div>Text fake-score <span class="fs-5 fw-bold mono">${fmt(t.fake_score)}</span></div>
      <div class="small text-secondary">${t.known_terms} n-grams matched the vocabulary</div></div>
      ${metricLine(t.test_metrics)}
      <h6 class="mt-3">Word-level explanation</h6>
      <div class="highlight-box">${t.highlighted_html}</div>
      <div class="small text-secondary mt-1"><mark class="hl-fake">red</mark> pushes toward fake, <mark class="hl-real">green</mark> toward real (exact TF-IDF × coefficient contributions of the linear model).</div>
      <div class="row mt-3"><div class="col-md-6"><h6>Terms pushing toward FAKE</h6>${chips(t.terms_toward_fake, "term-fake")}</div>
      <div class="col-md-6"><h6>Terms pushing toward REAL</h6>${chips(t.terms_toward_real, "term-real")}</div></div>
      <h6 class="mt-3">Style indicators</h6>
      <div class="small mono">words ${s.n_words} · exclamations ${s.exclamations} · questions ${s.questions} · ALL-CAPS words ${s.all_caps_words} · uppercase ratio ${fmt(s.uppercase_ratio)} · URL ${s.has_url ? "yes" : "no"}</div>
    </div></div>`;
}

function imagePanel(im) {
  const man = im.manipulation || {}, ctx = im.context || {}, ai = im.ai_generated || {}, md = im.metadata || {};
  const row = (k, v) => `<tr><th class="fw-normal text-secondary">${esc(k)}</th><td class="mono">${v}</td></tr>`;
  const exif = Object.entries(md.exif || {}).map(([k, v]) => row(k, esc(v))).join("");
  const concepts = (im.concepts || []).map((c) => `${esc(c.label)} (${fmt(c.prob)})`).join(", ");
  return `<div class="card mb-3"><div class="card-header"><i class="bi bi-image me-1"></i>Image analysis</div><div class="card-body">
    <div class="row g-3 mb-2">
      <div class="col-sm-6 col-lg-3"><div class="text-secondary small">Authenticity score</div><div class="fs-4 fw-bold mono">${fmt(im.authenticity_score)}</div><div class="small text-secondary">1 − max(technical scores)</div></div>
      <div class="col-sm-6 col-lg-3"><div class="text-secondary small">Tampering (forensic)</div><div class="fs-4 fw-bold mono">${man.available ? fmt(man.score) : "n/a"}</div><div class="small text-secondary">${man.available ? `${fmt((man.suspicious_block_fraction || 0) * 100, 1)}% suspicious blocks` : ""}</div></div>
      <div class="col-sm-6 col-lg-3"><div class="text-secondary small">AI-generated</div><div class="fs-4 fw-bold mono">${ai.available ? fmt(ai.score) : "n/a"}</div><div class="small text-secondary">${ai.available ? "" : "detector not trained"}</div></div>
      <div class="col-sm-6 col-lg-3"><div class="text-secondary small">Misleading-use (content)</div><div class="fs-4 fw-bold mono">${ctx.available ? fmt(ctx.score) : "n/a"}</div><div class="small text-secondary">weak, dataset-specific</div></div>
    </div>
    ${metricLine(man.test_metrics)}
    <h6 class="mt-3">Visual evidence</h6>
    <div class="row g-3">
      ${evidenceImg(im.preview, "Uploaded image")}
      ${evidenceImg(man.heatmap, "Forensic heatmap: red blocks = high tamper probability (blue = consistent)")}
      ${evidenceImg(man.ela, "Error Level Analysis (bright = high re-compression error)")}
      ${evidenceImg(ctx.cam, "Class activation map of the content model (regions driving its score)")}
      ${evidenceImg(ai.cam, "Class activation map of the AI-generated detector")}
    </div>
    <div class="row mt-3"><div class="col-lg-6"><h6>Metadata</h6><table class="table table-sm small mb-0">
      ${row("Format / mode", esc(`${md.format || "?"} / ${md.mode || "?"}`))}${row("Size", esc((md.size || []).join(" × ")))}
      ${row("JPEG quality (est.)", md.jpeg_quality_estimate ?? "–")}${row("Camera EXIF", md.has_camera_exif ? "present" : "absent")}
      ${row("AI-generator tags", esc((md.ai_generator_tags || []).join(", ") || "none"))}${row("Editing software", esc((md.editing_software || []).join(", ") || "none"))}${exif}
    </table></div>
    <div class="col-lg-6"><h6>Recognised content (ImageNet classes)</h6><div class="small">${concepts || "–"}</div>
    <div class="small text-secondary mt-1">Used to describe the image; object labels are not evidence of authenticity.</div></div></div>
  </div></div>`;
}

function videoPanel(v, idx) {
  const md = v.metadata || {};
  const frames = (v.frames || []).map((f) => `<div class="col-6 col-md-4 col-lg-2"><div class="frame-card"><img src="${f.thumb}" alt="frame ${f.frame_index}">
      <div class="meta mono">t=${fmt(f.time_s, 1)}s · tamper ${fmt(f.manipulation)}${f.ai_generated !== null ? ` · AI ${fmt(f.ai_generated)}` : ""} · faces ${f.faces}</div></div></div>`).join("");
  return `<div class="card mb-3"><div class="card-header"><i class="bi bi-camera-video me-1"></i>Video analysis</div><div class="card-body">
    <div class="row g-3 mb-2">
      <div class="col-6 col-lg-3"><div class="text-secondary small">Mean frame tamper score</div><div class="fs-4 fw-bold mono">${fmt(v.manipulation_mean)}</div></div>
      <div class="col-6 col-lg-3"><div class="text-secondary small">Max frame tamper score</div><div class="fs-4 fw-bold mono">${fmt(v.manipulation_max)}</div></div>
      <div class="col-6 col-lg-3"><div class="text-secondary small">Scene discontinuities</div><div class="fs-4 fw-bold mono">${v.temporal?.scene_cuts ?? "–"}</div></div>
      <div class="col-6 col-lg-3"><div class="text-secondary small">Duration · fps · codec</div><div class="mono">${fmt(md.duration_s, 1)} s · ${fmt(md.fps, 1)} · ${esc(md.codec)}</div><div class="small text-secondary">${esc((md.resolution || []).join("×"))}</div></div>
    </div>
    <h6 class="mt-3">Key frames</h6><div class="row g-2">${frames}</div>
    <h6 class="mt-3">Temporal consistency (colour-histogram distance between consecutive samples)</h6>
    <div style="height:180px"><canvas id="temporal-${idx}"></canvas></div>
  </div></div>`;
}

function consistencyPanel(c) {
  if (!c) return "";
  const body = c.available
    ? `<div>Consistency score (${esc(c.backend)}): <span class="fw-bold mono">${fmt(c.score)}</span> <span class="small text-secondary">(0 = image unrelated to text, 1 = strong match)</span></div>`
    : `<div class="small">${esc(c.note || "Not available.")}</div>`;
  const ov = c.concept_overlap && c.concept_overlap.length ? `<div class="small mt-2">Recognised objects also mentioned in the text: ${esc(c.concept_overlap.join(", "))}</div>` : "";
  return `<div class="card mb-3"><div class="card-header"><i class="bi bi-link-45deg me-1"></i>Text–image consistency</div><div class="card-body">${body}${ov}</div></div>`;
}

/* ------------------------------------------------------------- main render */
function renderResult(r, target) {
  charts.splice(0).forEach((c) => c.destroy());
  const f = r.fusion || {};
  const contrib = f.contributions || [];
  const hasContrib = contrib.length > 0;
  let html = `<div class="card mb-3"><div class="card-body">
    <div class="row g-3 align-items-center">
      <div class="col-lg-5"><div class="verdict ${verdictClass(r.verdict)}">
        <div class="small text-uppercase fw-semibold opacity-75">Final assessment${r.id ? ` · #${r.id}` : ""}</div>
        <div class="label">${esc(r.verdict)}</div>
        <div class="mt-1">Model score <span class="mono fw-bold">${fmt(r.score)}</span> · decisiveness <span class="mono fw-bold">${r.confidence === null ? "–" : fmt(r.confidence * 100, 0) + "%"}</span></div>
        ${scoreBar(r.score)}
      </div>
      <div class="small text-secondary mt-2">Decisiveness = distance of the score from 0.5. It is <b>not</b> the probability that the content is fake or true.</div></div>
      <div class="col-lg-7">${hasContrib ? `<h6 class="mb-1">Modality-wise scores and fusion weights</h6><div style="height:${60 + contrib.length * 34}px"><canvas id="contrib-chart"></canvas></div>` : ""}</div>
    </div></div></div>`;

  if ((r.flags || []).length) html += `<div class="box-flag mb-3"><b><i class="bi bi-exclamation-octagon me-1"></i>Evidence flags</b>${list(r.flags, "list-flag mt-1")}</div>`;
  if ((r.reasons || []).length) html += `<div class="card mb-3"><div class="card-header"><i class="bi bi-lightbulb me-1"></i>Why this result</div><div class="card-body">${list(r.reasons, "list-reason")}</div></div>`;
  if ((r.warnings || []).length) html += `<div class="box-warn mb-3"><b><i class="bi bi-exclamation-triangle me-1"></i>Warnings and limitations</b>${list(r.warnings, "list-warn mt-1")}</div>`;

  const p = r.parts || {};
  if (p.text) html += textPanel(p.text);
  if (p.consistency) html += consistencyPanel(p.consistency);
  if (p.image) html += imagePanel(p.image);
  if (p.video) html += videoPanel(p.video, 0);
  html += `<p class="disclaimer mt-3">${esc(r.disclaimer)} <span class="text-nowrap">(analysis time ${fmt(r.elapsed_s, 1)} s)</span></p>`;
  target.innerHTML = html;

  if (hasContrib) {
    const ctx = document.getElementById("contrib-chart");
    charts.push(new Chart(ctx, {
      type: "bar",
      data: {
        labels: contrib.map((c) => `${c.name} (w=${fmt(c.weight)})`),
        datasets: [{ label: "fake-score", data: contrib.map((c) => c.score),
          backgroundColor: contrib.map((c) => (c.score >= 0.65 ? "#d1453b" : c.score <= 0.35 ? "#1f8a5b" : "#8a7fd6")), borderRadius: 4 }],
      },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => `score ${fmt(c.raw)} · weight ${fmt(contrib[c.dataIndex].weight)}` } } },
        scales: { x: { min: 0, max: 1, title: { display: true, text: "modality fake-score (0 real-like … 1 fake-like)" } } } },
    }));
  }
  if (p.video && p.video.temporal) {
    const s = p.video.temporal.series || [];
    charts.push(new Chart(document.getElementById("temporal-0"), {
      type: "line",
      data: { labels: s.map((_, i) => i + 1), datasets: [{ data: s, borderColor: "#2f6fed", pointRadius: 2, tension: 0.2, fill: false }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "sample" } }, y: { min: 0, title: { display: true, text: "distance" } } } },
    }));
  }
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* --------------------------------------------------------- form handling */
async function submitAnalysis(form, url, target, buildBody) {
  const btn = form.querySelector("button[type=submit]");
  const errBox = form.querySelector(".form-error");
  errBox.textContent = "";
  errBox.classList.add("d-none");
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    const { body, headers } = buildBody();
    const res = await fetch(url, { method: "POST", body, headers });
    let data;
    try { data = await res.json(); } catch { throw new Error(`Server returned ${res.status}`); }
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    renderResult(data, target);
  } catch (e) {
    errBox.textContent = e.message;
    errBox.classList.remove("d-none");
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}

function setupDropzone(zone) {
  const input = zone.querySelector("input[type=file]");
  const preview = zone.querySelector(".dz-preview");
  const show = () => {
    const f = input.files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    const isVid = f.type.startsWith("video");
    preview.innerHTML = `<div class="small fw-semibold mt-2">${esc(f.name)} · ${(f.size / 1048576).toFixed(2)} MB</div>` +
      (isVid ? `<video src="${url}" controls muted></video>` : f.type.startsWith("image") ? `<img src="${url}" alt="preview">` : "");
  };
  zone.addEventListener("click", (e) => { if (e.target.tagName !== "VIDEO") input.click(); });
  input.addEventListener("click", (e) => e.stopPropagation());
  input.addEventListener("change", show);
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(); } });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".dropzone").forEach(setupDropzone);
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
    const ta = document.querySelector(b.dataset.target || "#text");
    ta.value = b.dataset.sample;
    if (b.dataset.kind) { const k = document.querySelector(`input[name=kind][value=${b.dataset.kind}]`); if (k) k.checked = true; }
  }));
});
