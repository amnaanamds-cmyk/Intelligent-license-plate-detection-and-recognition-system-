/* PlateVision LPR web app - plain JavaScript, no external libraries, so it
 * also works on sites without internet access. All text coming from the
 * server is inserted with textContent (never innerHTML) to prevent injection. */
"use strict";

const ROLES = ["viewer", "operator", "admin"];
const state = { token: null, user: null, role: null, view: null, lastEventId: 0,
                timers: [], searchOffset: 0, searchQuery: null };

/* ------------------------------------------------------------ helpers -- */
const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return n;
}

/* Like node.replaceChildren(), but skips null/false entries (the native
 * method would insert them as the text "null"/"false"). */
function fill(node, ...kids) {
  node.replaceChildren(...kids.flat().filter(k => k !== null && k !== undefined && k !== false));
}

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
  } catch (_) { return null; }
}

function canDo(role) { return ROLES.indexOf(state.role) >= ROLES.indexOf(role); }

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
                 : d.toLocaleString([], { dateStyle: "short", timeStyle: "short" });
}

function pct(x) { return `${Math.round((x || 0) * 100)}%`; }

function imgUrl(eventId, kind) {
  const t = state.token ? `?token=${encodeURIComponent(state.token)}` : "";
  return `/events/${eventId}/${kind}${t}`;
}

let toastTimer = null;
function toast(msg, critical = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (critical ? " critical" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, critical ? 6000 : 3000);
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { method: opts.method || "GET", headers, body: opts.body });
  if (res.status === 401 && !path.startsWith("/auth/")) {
    logout(false);
    throw new Error("Session expired - please sign in again");
  }
  let data = null;
  const ct = res.headers.get("content-type") || "";
  data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : data;
    throw new Error(typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map(d => d.msg).join("; ") : `HTTP ${res.status}`);
  }
  return data;
}

/* --------------------------------------------------------------- auth -- */
async function boot() {
  state.token = store("lpr_token");
  let me;
  try { me = await api("/auth/me"); }
  catch (e) { showAuth(false, "Cannot reach the server: " + e.message); return; }
  if (me.authenticated && !me.setup_required) return startApp(me);
  if (me.setup_required) return showAuth(true);
  showAuth(false);
}

function showAuth(setup, error = "") {
  $("#app").hidden = true;
  $("#auth-screen").hidden = false;
  $("#login-form").hidden = setup;
  $("#setup-form").hidden = !setup;
  $("#auth-subtitle").textContent = setup ? "Welcome - let's set up the system" : "Sign in to continue";
  $("#auth-error").textContent = error;
}

async function doLogin(username, password) {
  const r = await api("/auth/login", { method: "POST", json: { username, password } });
  state.token = r.token;
  store("lpr_token", r.token);
  startApp({ username: r.username, role: r.role });
}

$("#login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  $("#auth-error").textContent = "";
  try { await doLogin(f.get("username"), f.get("password")); }
  catch (e) { $("#auth-error").textContent = e.message; }
});

$("#setup-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  $("#auth-error").textContent = "";
  try {
    await api("/users", { method: "POST",
      json: { username: f.get("username"), password: f.get("password"), role: "admin" } });
    await doLogin(f.get("username"), f.get("password"));
    toast("Admin account created");
  } catch (e) { $("#auth-error").textContent = e.message; }
});

function logout(callServer = true) {
  if (callServer && state.token) api("/auth/logout", { method: "POST" }).catch(() => {});
  state.token = null;
  store("lpr_token", null);
  state.timers.forEach(clearInterval);
  state.timers = [];
  showAuth(false);
}
$("#logout-btn").addEventListener("click", () => logout());

/* ---------------------------------------------------------- navigation -- */
function startApp(me) {
  state.user = me.username;
  state.role = me.role;
  $("#auth-screen").hidden = true;
  $("#app").hidden = false;
  $("#user-name").textContent = me.username;
  $("#user-role").textContent = me.role;
  document.querySelectorAll("[data-role]").forEach(n => {
    if (!n.classList.contains("view")) n.hidden = !canDo(n.dataset.role);
  });
  buildNav();
  checkHealth();
  state.timers.push(setInterval(checkHealth, 15000));
  state.timers.push(setInterval(() => { if (state.view === "live") refreshLive(); }, 3000));
  state.timers.push(setInterval(() => { if (state.view === "cameras") refreshCameras(); }, 4000));
  const wanted = (location.hash || "").slice(1) || store("lpr_view") ||
                 (canDo("operator") ? "scan" : "live");
  show(document.getElementById("view-" + wanted) && canDo(document.getElementById("view-" + wanted).dataset.role)
       ? wanted : "live");
}

function buildNav() {
  const views = [...document.querySelectorAll(".view")].filter(v => canDo(v.dataset.role));
  for (const navId of ["tabs-desktop", "tabs-mobile"]) {
    const nav = document.getElementById(navId);
    fill(nav, ...views.map(v => {
      const name = v.id.replace("view-", "");
      return el("button", { type: "button", "data-view": name, onclick: () => show(name) },
        navId === "tabs-mobile" ? el("span", { class: "ic", "aria-hidden": "true", text: v.dataset.icon }) : null,
        el("span", { text: v.dataset.title }));
    }));
  }
}

function show(name) {
  state.view = name;
  store("lpr_view", name);
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== "view-" + name; });
  document.querySelectorAll("[data-view]").forEach(b =>
    b.setAttribute("aria-current", b.dataset.view === name ? "page" : "false"));
  ({ live: refreshLive, search: () => runSearch(false), watchlist: refreshWatchlist,
     cameras: refreshCameras, admin: refreshAdmin })[name]?.();
  window.scrollTo(0, 0);
}

async function checkHealth() {
  try {
    const h = await api("/health");
    const b = $("#banner");
    if (h.model_error) {
      b.textContent = "⚠ Recognition models are not loaded: " + h.model_error;
      b.className = "banner critical";
      b.hidden = false;
    } else b.hidden = true;
  } catch (_) {
    const b = $("#banner");
    b.textContent = "⚠ Server unreachable - retrying…";
    b.className = "banner";
    b.hidden = false;
  }
}

/* --------------------------------------------------------------- scan -- */
function alertBoxes(alerts) {
  return (alerts || []).map(a => {
    if (a.type === "watchlist") {
      return el("div", { class: "alert-box" + (a.match === "fuzzy" ? " fuzzy" : "") },
        a.match === "fuzzy" ? `⚠ Possible watchlist match: ${a.watch_plate}` : `🚨 WATCHLIST: ${a.watch_plate}`,
        el("span", { class: "sub", text: (a.reason || "no reason given") +
          (a.match === "fuzzy" ? ` · differs by ${a.distance} character` : "") }));
    }
    if (a.type === "mismatch") {
      return el("div", { class: "alert-box" }, "🚨 Vehicle does not match its registration",
        el("span", { class: "sub", text: a.summary || "" }));
    }
    return el("div", { class: "alert-box", text: a.type });
  });
}

function verificationChip(v) {
  if (!v) return null;
  const cls = { match: "good", mismatch: "crit", not_registered: "warn" }[v.status] || "";
  const label = { match: "Registration match", mismatch: "Registration MISMATCH",
                  not_registered: "Not registered", insufficient_evidence: "Unverified" }[v.status] || v.status;
  return el("span", { class: "chip " + cls, title: v.summary || "", text: label });
}

function formatChip(valid) {
  if (valid === null || valid === undefined) return null;
  return valid ? el("span", { class: "chip good", text: "Valid format" })
               : el("span", { class: "chip warn", text: "Unknown format" });
}

async function scanFile(file) {
  if (!file) return;
  const status = $("#scan-status");
  const out = $("#scan-results");
  fill(out);
  fill(status, el("span", { class: "spinner" }), "Recognising…");
  const fd = new FormData();
  fd.append("image", await downscale(file), "photo.jpg");
  const save = $("#scan-save").checked;
  const cam = $("#scan-camera-id").value.trim() || "mobile";
  const t0 = performance.now();
  try {
    const r = await api(`/recognize?images=true&save=${save}&camera_id=${encodeURIComponent(cam)}`,
                        { method: "POST", body: fd });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = r.count ? `${r.count} plate(s) found in ${secs} s` +
                         (save ? " · saved to the event log" : "") : `No plate found (${secs} s)`;
    if (r.annotated_image) out.append(el("img", { class: "result-image", src: r.annotated_image, alt: "Detected plates" }));
    for (const p of r.plates) {
      const alerts = p.alerts || [];
      if (alerts.length) {
        if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
        beep();
      }
      out.append(el("div", { class: "card plate-card" },
        ...alertBoxes(alerts),
        el("div", {}, el("span", { class: "plate", text: p.plate || "?" })),
        el("div", { class: "meta" },
          el("span", { class: "chip", text: `OCR ${pct(p.ocr_confidence)}` }),
          el("span", { class: "chip", text: `Detection ${pct(p.detection_confidence)}` }),
          formatChip(p.format_valid),
          p.corrections ? el("span", { class: "chip", text: `${p.corrections} char. corrected` }) : null,
          verificationChip(p.verification)),
        p.verification ? el("p", { class: "muted small", text: p.verification.summary }) : null,
        el("div", { class: "crops" },
          p.crop_image ? el("figure", {}, el("img", { src: p.crop_image, alt: "Plate crop" }),
                                         el("figcaption", { text: "Detected plate" })) : null,
          p.enhanced_image ? el("figure", {}, el("img", { src: p.enhanced_image, alt: "Enhanced plate" }),
                                             el("figcaption", { text: "Enhanced for OCR" })) : null),
        el("p", { class: "muted small", text: `Raw OCR text: ${p.raw_text || "-"}` })));
    }
  } catch (e) {
    status.textContent = "";
    out.append(el("div", { class: "alert-box", text: "Recognition failed: " + e.message }));
  }
}

/* Phone photos are 12+ MP. 1920 px on the long side is plenty for plates
 * and makes uploads over mobile data fast. */
function downscale(file, maxSide = 1920) {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      if (scale === 1 && file.size < 3e6) { URL.revokeObjectURL(url); return resolve(file); }
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      c.toBlob(b => resolve(b || file), "image/jpeg", 0.9);
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
    img.src = url;
  });
}

for (const id of ["#scan-camera", "#scan-gallery"]) {
  $(id).addEventListener("change", (ev) => { scanFile(ev.target.files[0]); ev.target.value = ""; });
}
$("#scan-camera-id").value = store("lpr_scan_cam") || "mobile";
$("#scan-camera-id").addEventListener("change", (e) => store("lpr_scan_cam", e.target.value));

let audioCtx = null;
function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = 880; o.type = "square"; g.gain.value = 0.08;
    o.connect(g); g.connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + 0.35);
  } catch (_) { /* audio not available */ }
}

/* ---------------------------------------------------------- dashboard -- */
function eventRow(e, isNew = false) {
  const hasAlert = (e.alerts || []).length > 0;
  const alertLabel = hasAlert ? e.alerts.map(a => a.type === "watchlist"
      ? (a.match === "fuzzy" ? "possible watchlist" : "watchlist") : a.type).join(", ") : null;
  const thumb = e.crop_path
    ? el("img", { class: "thumb", src: imgUrl(e.id, "crop"), alt: "", loading: "lazy" })
    : el("div", { class: "thumb none", text: "no image" });
  return el("li", { class: (hasAlert ? "alert" : "") + (isNew ? " new" : ""), tabindex: "0",
                    onclick: () => openEvent(e.id),
                    onkeydown: (ev) => { if (ev.key === "Enter") openEvent(e.id); } },
    thumb,
    el("div", { class: "info" },
      el("span", { class: "plate sm", text: e.plate }),
      el("div", { class: "sub" }, `${e.camera_id} · ${pct(e.confidence)}`,
        e.n_reads > 1 ? ` · ${e.n_reads} reads` : "")),
    el("div", { class: "when" }, fmtTime(e.ts),
      hasAlert ? el("div", {}, el("span", { class: "chip crit", text: alertLabel })) : null,
      e.verification_status === "match" ? el("div", {}, el("span", { class: "chip good", text: "verified" })) : null));
}

function kpi(label, value, sub) {
  return el("div", { class: "kpi" }, el("div", { class: "label", text: label }),
    el("div", { class: "value", text: value }), sub ? el("div", { class: "sub", text: sub }) : null);
}

function drawHourly(hourly) {
  const chart = $("#hourly-chart");
  const tip = $("#chart-tip");
  // 24 slots ending with the current hour (server buckets are UTC "YYYY-MM-DDTHH").
  const counts = new Map(hourly.map(h => [h.hour, h.count]));
  const now = new Date();
  const slots = [];
  for (let i = 23; i >= 0; i--) {
    const d = new Date(now.getTime() - i * 3600e3);
    const key = d.toISOString().slice(0, 13);
    slots.push({ key, local: d, count: counts.get(key) || 0 });
  }
  const max = Math.max(...slots.map(s => s.count));
  chart.setAttribute("aria-label", `Plate reads per hour over the last 24 hours, peak ${max}`);
  if (!max) { fill(chart, el("div", { class: "chart-empty", text: "No reads in the last 24 hours" })); return; }
  fill(chart, 
    el("span", { class: "max-label", text: `${max}` }),
    ...slots.map(s => {
      const label = `${s.local.toLocaleTimeString([], { hour: "2-digit" })} · ${s.count} read${s.count === 1 ? "" : "s"}`;
      const hit = el("div", { class: "bar-hit", tabindex: "0", "aria-label": label },
        el("div", { class: "bar", style: `height:${(s.count / max) * 100}%` }));
      const showTip = () => {
        const r = hit.getBoundingClientRect(), p = chart.parentElement.getBoundingClientRect();
        tip.textContent = label;
        tip.style.left = `${r.left - p.left + r.width / 2}px`;
        tip.style.top = `${chart.offsetTop + 10}px`;
        tip.hidden = false;
      };
      hit.addEventListener("mouseenter", showTip);
      hit.addEventListener("focus", showTip);
      hit.addEventListener("click", showTip);
      hit.addEventListener("mouseleave", () => { tip.hidden = true; });
      hit.addEventListener("blur", () => { tip.hidden = true; });
      return hit;
    }));
  let axis = chart.nextElementSibling;
  if (!axis || !axis.classList.contains("chart-axis")) {
    axis = el("div", { class: "chart-axis" });
    chart.after(axis);
  }
  fill(axis, el("span", { text: slots[0].local.toLocaleTimeString([], { hour: "2-digit" }) }),
                       el("span", { text: "now" }));
}

async function refreshLive() {
  try {
    const [s, events] = await Promise.all([api("/stats"), api("/events?limit=30")]);
    fill($("#kpis"), 
      kpi("Plates read today", s.events.toLocaleString(), `${s.events_all_time.toLocaleString()} all time`),
      kpi("Unique vehicles", s.unique_plates.toLocaleString(), "today"),
      kpi("Alerts today", s.alerts.toLocaleString(), `${s.watchlist_size} on watchlist`),
      kpi("Cameras online", `${s.cameras_online}/${s.cameras}`, s.mismatches ? `${s.mismatches} mismatches today` : " "));
    drawHourly(s.hourly);
    const newest = events.length ? events[0].id : 0;
    const fresh = state.lastEventId ? events.filter(e => e.id > state.lastEventId) : [];
    const freshIds = new Set(fresh.map(e => e.id));
    const list = $("#live-events");
    fill(list, ...(events.length ? events.map(e => eventRow(e, freshIds.has(e.id)))
                             : [el("li", { class: "empty", text: "No plate reads yet" })]));
    const freshAlerts = fresh.filter(e => (e.alerts || []).length);
    if (freshAlerts.length) {
      const e = freshAlerts[0];
      toast(`🚨 Alert: ${e.plate} at ${e.camera_id}`, true);
      if ($("#live-sound").checked) beep();
      if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
    }
    state.lastEventId = Math.max(state.lastEventId, newest);
  } catch (e) { /* shown by the health banner */ }
}

/* ------------------------------------------------------- event detail -- */
async function openEvent(id) {
  const d = $("#event-dialog");
  const body = $("#event-detail");
  fill(body, el("p", {}, el("span", { class: "spinner" }), "Loading…"));
  d.showModal();
  try {
    const e = await api(`/events/${id}`);
    const v = (e.details || {}).verification;
    fill(body, 
      el("div", {}, el("span", { class: "plate", text: e.plate })),
      ...alertBoxes(e.alerts),
      e.frame_path ? el("img", { class: "result-image", src: imgUrl(e.id, "frame"), alt: "Full frame" }) : null,
      e.crop_path ? el("div", { class: "crops" }, el("figure", {},
        el("img", { src: imgUrl(e.id, "crop"), alt: "Plate" }), el("figcaption", { text: "Plate crop" }))) : null,
      el("dl", { class: "kv" },
        el("dt", { text: "Event" }), el("dd", { text: `#${e.id}` }),
        el("dt", { text: "Time" }), el("dd", { text: new Date(e.ts).toLocaleString() }),
        el("dt", { text: "Camera" }), el("dd", { text: e.camera_id }),
        el("dt", { text: "Confidence" }), el("dd", { text: pct(e.confidence) }),
        el("dt", { text: "Reads" }), el("dd", { text: String(e.n_reads) }),
        el("dt", { text: "Format" }), el("dd", { text: e.format_valid === null ? "-" : e.format_valid ? "valid" : "unknown" }),
        el("dt", { text: "Raw OCR" }), el("dd", { text: e.raw_text || "-" }),
        (e.details || {}).reads ? el("dt", { text: "Frame reads" }) : null,
        (e.details || {}).reads ? el("dd", { text: e.details.reads.join(", ") }) : null,
        v ? el("dt", { text: "Verification" }) : null,
        v ? el("dd", { text: v.summary }) : null),
      canDo("operator") && !(e.alerts || []).some(a => a.type === "watchlist")
        ? el("button", { class: "btn danger", type: "button", onclick: async () => {
            const reason = prompt(`Add ${e.plate} to the watchlist. Reason:`, "");
            if (reason === null) return;
            try { await api("/watchlist", { method: "POST", json: { plate: e.plate, reason } });
                  toast(`${e.plate} added to watchlist`); }
            catch (err) { toast(err.message, true); }
          } }, "🚨 Add to watchlist") : null);
  } catch (e) { fill(body, el("p", { class: "error", text: e.message })); }
}

/* ------------------------------------------------------------- search -- */
function searchParams() {
  const f = new FormData($("#search-form"));
  const p = new URLSearchParams();
  for (const k of ["plate", "camera_id"]) if (f.get(k)) p.set(k, f.get(k).trim());
  for (const k of ["since", "until"]) if (f.get(k)) p.set(k, new Date(f.get(k)).toISOString());
  if (f.get("alerts_only")) p.set("alerts_only", "true");
  return p;
}

async function runSearch(append) {
  const p = searchParams();
  if (!append) state.searchOffset = 0;
  p.set("limit", "50");
  p.set("offset", String(state.searchOffset));
  const csv = new URLSearchParams(p);
  csv.delete("limit"); csv.delete("offset");
  if (state.token) csv.set("token", state.token);
  $("#export-csv").href = "/events.csv?" + csv.toString();
  try {
    const rows = await api("/events?" + p.toString());
    const list = $("#search-results");
    if (!append) fill(list);
    list.append(...rows.map(e => eventRow(e)));
    if (!append && !rows.length) list.append(el("li", { class: "empty", text: "No matching events" }));
    state.searchOffset += rows.length;
    $("#search-count").textContent = `${state.searchOffset} result(s)${rows.length === 50 ? "+" : ""}`;
    $("#search-more").hidden = rows.length < 50;
  } catch (e) { toast(e.message, true); }
}
$("#search-form").addEventListener("submit", (ev) => { ev.preventDefault(); runSearch(false); });
$("#search-more").addEventListener("click", () => runSearch(true));

/* ---------------------------------------------------------- watchlist -- */
async function refreshWatchlist() {
  try {
    const items = await api("/watchlist");
    fill($("#watch-list"), ...(items.length ? items.map(w => el("li", {},
      el("span", { class: "plate sm", text: w.plate }),
      el("span", { class: "grow" }, w.reason || el("span", { class: "muted", text: "no reason" }),
        el("div", { class: "muted small", text: "added " + fmtTime(w.added_ts) })),
      canDo("operator") ? el("button", { class: "btn small danger", type: "button", onclick: async () => {
        if (!confirm(`Remove ${w.plate} from the watchlist?`)) return;
        try { await api(`/watchlist/${encodeURIComponent(w.plate)}`, { method: "DELETE" }); refreshWatchlist(); }
        catch (e) { toast(e.message, true); }
      } }, "Remove") : null))
      : [el("li", { class: "empty", text: "The watchlist is empty" })]));
  } catch (e) { toast(e.message, true); }
}
$("#watch-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  try {
    const r = await api("/watchlist", { method: "POST", json: { plate: f.get("plate"), reason: f.get("reason") } });
    ev.target.reset();
    toast(`${r.plate} added`);
    refreshWatchlist();
  } catch (e) { toast(e.message, true); }
});

/* ------------------------------------------------------------ cameras -- */
async function refreshCameras() {
  try {
    const cams = await api("/cameras");
    fill($("#camera-cards"), ...(cams.length ? cams.map(c => {
      const state_ = c.error && !c.running ? ["crit", "Error"]
        : c.finished ? ["", "Finished"] : c.connected ? ["good", "Online"] : ["warn", "Connecting"];
      return el("div", { class: "card" },
        el("div", { class: "card-head" }, el("h3", { text: c.camera_id }),
          el("span", { class: "chip " + state_[0], text: state_[1] })),
        el("div", { class: "cam-src", text: c.source || "" }),
        el("div", { class: "cam-stats" },
          el("div", {}, el("b", { text: String(c.fps) }), "fps"),
          el("div", {}, el("b", { text: String(c.events) }), "events"),
          el("div", {}, el("b", { text: String(c.active_tracks) }), "tracking")),
        c.error ? el("p", { class: "error small", text: c.error }) : null,
        canDo("operator") ? el("button", { class: "btn small danger", type: "button", onclick: async () => {
          if (!confirm(`Stop camera ${c.camera_id}?`)) return;
          try { await api(`/cameras/${encodeURIComponent(c.camera_id)}`, { method: "DELETE" }); refreshCameras(); }
          catch (e) { toast(e.message, true); }
        } }, "Stop") : null);
    }) : [el("p", { class: "muted", text: "No cameras running. Add one below or in configs/system.yaml." })]));
  } catch (e) { /* health banner */ }
}
$("#camera-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  try {
    await api("/cameras", { method: "POST",
      json: { id: f.get("id"), source: f.get("source"), stride: Number(f.get("stride") || 1) } });
    ev.target.reset();
    toast("Camera started");
    refreshCameras();
  } catch (e) { toast(e.message, true); }
});

/* -------------------------------------------------------------- admin -- */
async function refreshAdmin() {
  try {
    const [users, audit] = await Promise.all([api("/users"), api("/audit?limit=100")]);
    fill($("#user-list"), ...users.map(u => el("li", {},
      el("strong", { class: "grow", text: u.username }),
      el("span", { class: "role", text: u.role }),
      u.username !== state.user ? el("button", { class: "btn small danger", type: "button", onclick: async () => {
        if (!confirm(`Delete user ${u.username}?`)) return;
        try { await api(`/users/${encodeURIComponent(u.username)}`, { method: "DELETE" }); refreshAdmin(); }
        catch (e) { toast(e.message, true); }
      } }, "Delete") : el("span", { class: "muted small", text: "(you)" }))));
    fill($("#audit-list"), ...(audit.length ? audit.map(a => el("li", {},
      el("span", { class: "ts", text: fmtTime(a.ts) }),
      el("strong", { text: a.username || "-" }), ` ${a.action} `,
      el("span", { class: "muted", text: a.detail || "" })))
      : [el("li", { class: "empty", text: "No entries" })]));
  } catch (e) { toast(e.message, true); }
}
$("#user-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  try {
    await api("/users", { method: "POST",
      json: { username: f.get("username"), password: f.get("password"), role: f.get("role") } });
    ev.target.reset();
    toast("User added");
    refreshAdmin();
  } catch (e) { toast(e.message, true); }
});

/* ----------------------------------------------------------------- go -- */
if ("serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
window.addEventListener("hashchange", () => {
  const name = location.hash.slice(1);
  if (state.role && name !== state.view && document.getElementById("view-" + name)) show(name);
});
boot();
