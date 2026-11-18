// 图文幻觉词高亮器 · 前端（原生 JS，无依赖）
"use strict";

const $ = (s) => document.querySelector(s);
const TYPE_ZH = { object: "物体", attribute: "属性", relationship: "关系", scene: "场景" };
const state = { meta: null, list: [], idx: 0, ex: null, live: { file: null, imageId: null } };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const fmt = (x, d = 2) => (x >= 0 ? "+" : "") + x.toFixed(d);
const pct = (q) => Math.round(q * 100) + "%";
const typeZh = (t) => (t || "").split("|").filter(Boolean).map((x) => TYPE_ZH[x] || x).join("/");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ---------- 渲染：逐词上色 ---------- */

function renderAnnotated(prefix, data) {
  const withLabels = data.with_labels;
  const words = [...data.words].sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  words.forEach((w, i) => {
    html += esc(data.text.slice(pos, w.start));
    const cls = ["w", w.content ? "" : "stop", w.flag ? "flag" : "", withLabels && w.label ? "gold" : ""].join(" ");
    const bg = w.content && w.color && w.q >= 0.6 ? ` style="background:${w.color}"` : "";  // 分位 60% 以下不上色，避免满屏底色
    html += `<span class="${cls}" data-i="${i}"${bg}>${esc(data.text.slice(w.start, w.end))}</span>`;
    pos = w.end;
  });
  html += esc(data.text.slice(pos));
  const box = $(`#${prefix}-text${prefix === "live" ? "-out" : ""}`);
  box.innerHTML = html;
  box.onclick = (e) => {
    const el = e.target.closest(".w");
    if (el) selectWord(prefix, data, words, +el.dataset.i);
  };
  box.onmousemove = (e) => {
    const el = e.target.closest(".w");
    if (!el) return hideTip();
    const w = words[+el.dataset.i];
    showTip(e, w.content
      ? `<b>${esc(w.word)}</b><br>log p（原图）${w.logp_real.toFixed(2)}<br>log p（无关图平均）${w.logp_other.toFixed(2)}<br>VIG ${fmt(w.vig)} · 分位 ${pct(w.q)}` +
        (withLabels ? `<br>标注：${w.label ? "幻觉（" + typeZh(w.type) + "）" : "真实"}` : "")
      : `<b>${esc(w.word)}</b><br>停用词，不打分`);
  };
  box.onmouseleave = hideTip;

  // 摘要
  const s = data.summary;
  let chips = `<span class="chip">实词 ${s.n_content}</span><span class="chip red">判为可疑 ${s.n_flagged}</span>`;
  if (withLabels) {
    chips += `<span class="chip">标注幻觉 ${s.n_gold}</span><span class="chip green">命中 ${s.n_hit}</span>` +
             `<span class="chip red">误报 ${s.n_false}</span>`;
  }
  if (data.seconds != null) chips += `<span class="chip">用时 ${data.seconds} 秒（${esc(data.device || "")}）</span>`;
  $(`#${prefix}-chips`).innerHTML = chips;

  $(`#${prefix}-legend`).innerHTML =
    `<span>不可疑<span class="bar"></span>可疑（颜色 = VIG 在验证集中的分位数）</span>` +
    `<span><span class="sw" style="outline:2px solid var(--red);font-weight:700">词</span> 超过阈值，判为可疑</span>` +
    (withLabels ? `<span><span class="sw" style="text-decoration:underline;text-decoration-color:var(--blue);text-decoration-thickness:3px;text-underline-offset:4px">词</span> 数据集标注的幻觉</span>` : "") +
    `<span style="color:var(--ink-3)">灰字 = 停用词，不打分</span>`;

  // 最可疑的词
  const top = words.map((w, i) => ({ ...w, i })).filter((w) => w.content).sort((a, b) => b.vig - a.vig).slice(0, 8);
  $(`#${prefix}-top`).innerHTML =
    `<tr><th>词</th><th>log p 原图</th><th>无关图平均</th><th>VIG</th><th>分位</th><th>判断</th>${withLabels ? "<th>标注</th>" : ""}</tr>` +
    top.map((w) => `<tr data-i="${w.i}"><td><b>${esc(w.word)}</b></td><td class="num">${w.logp_real.toFixed(2)}</td>` +
      `<td class="num">${w.logp_other.toFixed(2)}</td><td class="num">${fmt(w.vig)}</td><td class="num">${pct(w.q)}</td>` +
      `<td>${w.flag ? '<span style="color:var(--red)">可疑</span>' : "—"}</td>` +
      (withLabels ? `<td>${w.label ? "幻觉（" + typeZh(w.type) + "）" : "真实"}</td>` : "") + `</tr>`).join("");
  $(`#${prefix}-top`).onclick = (e) => {
    const tr = e.target.closest("tr[data-i]");
    if (tr) selectWord(prefix, data, words, +tr.dataset.i);
  };

  // 无关图
  $(`#${prefix}-others`).innerHTML = (data.other_image_ids || []).map((id, k) =>
    `<figure><img src="/api/image/${esc(id)}" alt="无关图 ${k + 1}" loading="lazy"><figcaption>无关图 ${k + 1}</figcaption></figure>`).join("");

  // 默认选中最可疑的词
  if (top.length) selectWord(prefix, data, words, top[0].i);
}

function selectWord(prefix, data, words, i) {
  const w = words[i];
  document.querySelectorAll(`#${prefix}-text${prefix === "live" ? "-out" : ""} .w.sel`).forEach((el) => el.classList.remove("sel"));
  const el = document.querySelector(`#${prefix}-text${prefix === "live" ? "-out" : ""} .w[data-i="${i}"]`);
  if (el) el.classList.add("sel");
  const out = $(`#${prefix}-detail`);
  if (!w.content) {
    out.innerHTML = `<span class="word">${esc(w.word)}</span><p class="explain">停用词（冠词、介词等）不打分。</p>`;
    return;
  }
  const thr = state.meta.threshold;
  const bar = (lp) => Math.max(2, Math.min(100, ((lp + 15) / 15) * 100)); // 横轴 log p ∈ [−15, 0]
  const verdict = w.flag
    ? `<span class="verdict" style="color:var(--red)">可疑（VIG ≥ ${thr.toFixed(2)}）</span>`
    : `<span class="verdict" style="color:var(--green)">不可疑</span>`;
  let why;
  if (w.vig >= thr) why = `它在原图下反而比在无关图下更不可能出现（差 ${w.vig.toFixed(2)}），超过阈值 → 原图不支持它，判为可疑。`;
  else if (w.vig > 0) why = `原图对它的支持略弱于无关图（差 ${w.vig.toFixed(2)}），但没到阈值，不报。`;
  else why = `它在原图下比在无关图下更可能出现（差 ${w.vig.toFixed(2)}），说明原图支持这个词。` +
             (w.logp_real < -6 ? "注意它本身概率很低（罕见词），只看概率会误报，VIG 把它排除了。" : "");
  const gold = data.with_labels
    ? `<p class="explain">数据集标注：<b>${w.label ? "幻觉（" + typeZh(w.type) + "）" : "真实"}</b>` +
      (w.label === (w.flag ? 1 : 0) ? "，判断正确。" : (w.label ? "，这里漏检了。" : "，这里是误报。")) + `</p>`
    : "";
  out.innerHTML =
    `<span class="word">${esc(w.word)}</span>${verdict}
     <div class="bars">
       <div class="brow"><span>原图</span><div class="btrack"><div class="bfill" style="left:0;right:auto;width:${bar(w.logp_real)}%;background:var(--ink-2)"></div></div><span class="bnum">${w.logp_real.toFixed(2)}</span></div>
       <div class="brow"><span>无关图平均</span><div class="btrack"><div class="bfill" style="left:0;right:auto;width:${bar(w.logp_other)}%;background:var(--blue)"></div></div><span class="bnum">${w.logp_other.toFixed(2)}</span></div>
       <div class="brow"><span>VIG</span><span>${fmt(w.vig)}　·　验证集分位 ${pct(w.q)}</span><span></span></div>
     </div>
     <p class="small">条越长 = 模型越认为这里会出现这个词（log p，横轴 −15 到 0）。</p>
     <p class="explain">${why}</p>${gold}`;
}

/* ---------- 悬停提示 ---------- */
function showTip(e, html) {
  const t = $("#tip");
  t.innerHTML = html;
  t.hidden = false;
  t.style.left = Math.min(e.clientX + 14, window.innerWidth - 280) + "px";
  t.style.top = e.clientY + 16 + "px";
}
function hideTip() { $("#tip").hidden = true; }

/* ---------- 示例浏览 ---------- */

async function loadList() {
  state.list = await api(`/api/examples?only=${$("#filter").value}`);
  $("#example-select").innerHTML = state.list.map((x, i) =>
    `<option value="${i}">${esc(x.id)} · ${esc(x.tag)}${x.hit ? " · 抓到了" : ""}</option>`).join("");
}

async function showExample(i) {
  if (!state.list.length) return;
  state.idx = (i + state.list.length) % state.list.length;
  $("#example-select").value = state.idx;
  const item = state.list[state.idx];
  const data = await api(`/api/example/${encodeURIComponent(item.key)}`);
  state.ex = data;
  $("#ex-image").src = `/api/image/${data.image_id}`;
  $("#ex-caption").textContent = `${data.id} · Visual Genome 图 ${data.image_id} · 来源：${data.source}`;
  renderAnnotated("ex", data);
}

/* ---------- 在线检测 ---------- */

function setLiveImage({ file = null, imageId = null }) {
  state.live = { file, imageId };
  const img = $("#live-preview");
  img.src = file ? URL.createObjectURL(file) : `/api/image/${imageId}`;
  img.hidden = false;
  $("#drop-hint").innerHTML = `换一张：拖进来，或 <label class="link">选择文件<input type="file" id="file" accept="image/*" hidden></label>`;
  $("#file").onchange = onFile;
  $("#live-source").textContent = file ? `上传：${file.name}` : `示例图：Visual Genome ${imageId}`;
}
function onFile(e) { if (e.target.files[0]) setLiveImage({ file: e.target.files[0] }); }

function formWith(extra) {
  const f = new FormData();
  if (state.live.file) f.append("file", state.live.file);
  else if (state.live.imageId) f.append("image_id", state.live.imageId);
  Object.entries(extra || {}).forEach(([k, v]) => f.append(k, v));
  return f;
}

async function busy(btns, label, fn) {
  const st = $("#live-status");
  btns.forEach((b) => (b.disabled = true));
  st.classList.remove("err");
  const t0 = Date.now();
  const tick = () => (st.innerHTML = `<span class="spin"></span>${label}… ${((Date.now() - t0) / 1000).toFixed(0)} 秒` +
    (state.meta.model_loaded ? "" : "（第一次要加载模型，约半分钟）"));
  tick();
  const timer = setInterval(tick, 500);
  try { await fn(); st.textContent = ""; }
  catch (e) { st.textContent = "出错：" + e.message; st.classList.add("err"); }
  finally { clearInterval(timer); btns.forEach((b) => (b.disabled = false)); state.meta.model_loaded = true; }
}

async function doCaption() {
  if (!state.live.file && !state.live.imageId) return showErr("先放一张图片");
  await busy([$("#btn-caption"), $("#btn-detect")], "模型正在写描述", async () => {
    const r = await api("/api/caption", { method: "POST", body: formWith() });
    $("#live-text").value = r.caption;
  });
  $("#live-status").textContent = "描述由模型生成——它自己写的也可能有幻觉，点“检测”看看。";
}

async function doDetect() {
  const text = $("#live-text").value.trim();
  if (!state.live.file && !state.live.imageId) return showErr("先放一张图片");
  if (!text) return showErr("先写一段英文描述");
  await busy([$("#btn-caption"), $("#btn-detect")], "正在检测（原图 + 4 张无关图，共 5 次前向）", async () => {
    const data = await api("/api/detect", { method: "POST", body: formWith({ text }) });
    $("#live-result").hidden = false;
    renderAnnotated("live", data);
    $("#live-result").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}
async function loadPreset(name) {
  const d = await api(`/api/preset/${encodeURIComponent(name)}`);
  $("#preset").value = name;
  setLiveImage({ imageId: d.image_id });
  $("#live-text").value = d.text;
  $("#live-result").hidden = false;
  renderAnnotated("live", d);
  $("#live-status").textContent = `演示预设「${d.title}」：之前在 ${d.device} 上实际跑出的结果（用时 ${d.seconds} 秒），这里直接读出。`;
  switchTab("live");
}

function showErr(m) { const st = $("#live-status"); st.textContent = m; st.classList.add("err"); }

/* ---------- 初始化 ---------- */

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
}

async function init() {
  state.meta = await api("/api/meta");
  $("#thr").textContent = state.meta.threshold.toFixed(2);
  $("#k-num").textContent = state.meta.k;
  $("#meta").textContent = `示例 ${state.meta.n_examples} 条（其中 ${state.meta.n_hit} 条抓到了幻觉）· 模型 ${state.meta.model} · ` +
    `阈值在验证集上冻结：VIG ≥ ${state.meta.threshold.toFixed(2)}`;

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));
  $("#filter").onchange = async () => { await loadList(); showExample(0); };
  $("#example-select").onchange = (e) => showExample(+e.target.value);
  $("#prev").onclick = () => showExample(state.idx - 1);
  $("#next").onclick = () => showExample(state.idx + 1);
  $("#rand").onclick = () => showExample(Math.floor(Math.random() * state.list.length));
  $("#to-live").onclick = () => {
    if (!state.ex) return;
    setLiveImage({ imageId: state.ex.image_id });
    $("#live-text").value = state.ex.text;
    $("#live-status").textContent = "已带过来。改动文字（比如加一个图里没有的东西）后点“检测”。";
    switchTab("live");
  };

  const drop = $("#drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setLiveImage({ file: f });
  };
  $("#file").onchange = onFile;
  $("#btn-caption").onclick = doCaption;
  $("#btn-detect").onclick = doDetect;

  const presets = await api("/api/presets");
  $("#preset").innerHTML = `<option value="">— 选一个预设 —</option>` +
    presets.map((p) => `<option value="${esc(p.name)}">${esc(p.title)}</option>`).join("");
  $("#preset").onchange = (e) => e.target.value && loadPreset(e.target.value);
  $("#preset-wrap").hidden = !presets.length;

  // 链接参数：?ex=<示例 key> 打开某条示例；?preset=<名字> 打开演示预设
  const qs = new URLSearchParams(location.search);
  if (qs.get("ex")) $("#filter").value = "all";
  await loadList();
  const want = qs.get("ex") || state.meta.default;
  const d = state.list.findIndex((x) => x.key === want);
  await showExample(d >= 0 ? d : 0);
  if (qs.get("preset")) await loadPreset(qs.get("preset"));
}

init().catch((e) => ($("#meta").textContent = "载入失败：" + e.message));
