"use strict";

// Pure functions also exercised by Node without starting a browser.
const ComparisonMath = Object.freeze({
  clamp(value, low, high) { return Math.max(low, Math.min(high, value)); },
  frameAt(time, fps, count) { return Math.max(0, Math.min(count - 1, Math.round(time * fps))); },
});
if (typeof module !== "undefined") module.exports = ComparisonMath;

if (typeof document !== "undefined") (() => {
  const data = JSON.parse(document.getElementById("data").textContent);
  const $ = id => document.getElementById(id);
  const canvas = $("picture"), ctx = canvas.getContext("2d"), video = $("video");
  let width = data.sequence.width, height = data.sequence.height, fps = data.sequence.fps;
  let count = data.frame_count;
  canvas.width = width; canvas.height = height;
  $("time").max = count - 1;
  $("language").value = data.language;
  let language = data.language, clip = data.cases[0], method = "automatic", frame = 0;
  let picture = null, revision = 0, playing = false, status = "loading", callback = null;
  let pending = null;
  let decoding = false, wantedFrame = null;
  const cache = new Map(); // At most six decoded PNGs from the current clip.
  const t = key => data.messages[language][key];
  const number = (value, digits = 0) => new Intl.NumberFormat(language, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }).format(value);
  const format = (key, parameters) => t(key).replace(/\{(\w+)\}/g, (_, name) => parameters[name]);

  function setStatus(value) {
    status = value;
    $("status").textContent = t(value);
    $("play").textContent = t(playing ? "pause" : "play");
    $("play").setAttribute("aria-pressed", String(playing));
  }
  function updatePosition() {
    $("time").value = frame;
    const position = format("position", {time: number(frame / fps, 3), frame: number(frame + 1), count: number(count)});
    $("position").textContent = position;
    $("time").setAttribute("aria-valuetext", position);
    $("previous").disabled = frame === 0;
    $("next").disabled = frame === count - 1;
    const traces = clip.traces_by_method?.[method] || clip.traces;
    $("amount").textContent = format("amount", {value: number(traces[frame].amount, 3)});
  }
  function render(source = picture) {
    if (!source) return;
    const x = width * Number($("split").value) / 1000;
    ctx.imageSmoothingEnabled = false;
    // One decoded image supplies BOTH sides. There are never two playback clocks.
    ctx.drawImage(source, 0, 0, width, height, 0, 0, width, height);
    ctx.save(); ctx.beginPath(); ctx.rect(x, 0, width - x, height); ctx.clip();
    ctx.drawImage(source, data.methods.indexOf(method) * width, 0, width, height, 0, 0, width, height);
    ctx.restore();
  }
  function renderCurrent() { render(playing && video.readyState >= 2 ? video : picture); }
  function metrics() {
    $("right-label").textContent = $("metric-method").textContent = t(method);
    for (const [suffix, key] of [["peak", "peak_y"], ["delta", "max_pixel_delta"], ["dark", "dark_luminance_retention"]]) {
      for (const [prefix, id] of [["source", "original"], ["result", method]]) {
        const value = clip.metrics[id][key];
        $(prefix + "-" + suffix).textContent = value === null ? t("not_applicable") : number(value, 5);
      }
    }
    $("scope").textContent = t(data.scope_key || "scope");
    $("source").hidden = !clip.source;
    if (clip.source) {
      $("source-link").textContent = t("source");
      $("source-link").href = clip.source.page_url;
      $("source-range").textContent = format("source_range", {start: number(clip.source.start_seconds, 3),
        end: number(clip.source.start_seconds + count / fps, 3), fps: number(fps)});
    }
  }
  function translate() {
    document.documentElement.lang = language; document.title = t("title");
    document.querySelectorAll("[data-text]").forEach(el => { el.textContent = t(el.dataset.text); });
    $("case").replaceChildren(...data.cases.map(item => new Option(t("case_" + item.id), item.id)));
    $("case").value = clip.id;
    $("method").replaceChildren(...data.methods.slice(1).map(id => new Option(t(id), id)));
    $("method").value = method;
    canvas.setAttribute("aria-label", t("image"));
    $("split").setAttribute("aria-valuetext", format("split_value", {value: number(Number($("split").value) / 10)}));
    metrics(); updatePosition(); setStatus(status);
  }
  function cancelCallback() {
    if (callback === null) return;
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(callback);
    else cancelAnimationFrame(callback);
    callback = null;
  }
  function stop() {
    revision++; playing = false; video.pause(); cancelCallback();
    wantedFrame = null;
    if (pending) { pending.abort(); pending = null; }
  }
  function waitMedia(event, signal) {
    return new Promise((resolve, reject) => {
      const finish = error => {
        clearTimeout(timeout);
        video.removeEventListener(event, ready); video.removeEventListener("error", failed);
        signal.removeEventListener("abort", aborted);
        if (error) reject(error); else resolve();
      };
      const ready = () => finish(), failed = () => finish(new Error("media"));
      const aborted = () => finish(new Error("cancelled"));
      const timeout = setTimeout(failed, 10000);
      video.addEventListener(event, ready); video.addEventListener("error", failed);
      signal.addEventListener("abort", aborted, {once: true});
      if (signal.aborted) aborted();
    });
  }
  async function showFrame(index) {
    stop(); frame = ComparisonMath.clamp(index, 0, count - 1);
    wantedFrame = {token: revision, currentClip: clip, ref: clip.frame_refs[frame]};
    updatePosition(); setStatus("loading");
    if (decoding) return;
    decoding = true;
    try {
      // Keep one decode in flight; rapid scrubbing retains only the newest request.
      while (wantedFrame) {
        const {token, currentClip, ref} = wantedFrame;
        wantedFrame = null;
        try {
          let image = cache.get(ref);
          if (!image) {
            const record = currentClip.png_pool[ref];
            image = new Image(); image.src = record.data ? "data:image/png;base64," + record.data : record.file;
            await image.decode();
            if (token !== revision || currentClip !== clip) continue;
            if (cache.size >= 6) cache.delete(cache.keys().next().value);
            cache.set(ref, image);
          }
          if (token !== revision) continue;
          picture = image; render(); setStatus("paused_png");
        } catch (_) { if (token === revision) setStatus("image_error"); }
      }
    } finally { decoding = false; }
  }
  function schedule(token) {
    const paint = (_, metadata) => {
      callback = null;
      if (!playing || token !== revision) return;
      frame = ComparisonMath.frameAt(metadata ? metadata.mediaTime : video.currentTime, fps, count);
      render(video); updatePosition(); schedule(token);
    };
    callback = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(paint) : requestAnimationFrame(paint);
  }
  async function play() {
    if (playing) { await showFrame(frame); return; }
    stop(); const token = revision;
    pending = new AbortController(); const signal = pending.signal;
    if (frame === count - 1) frame = 0;
    playing = true; setStatus("loading");
    try {
      // Seeking and play are asynchronous. Revision guards prevent a late play
      // from resuming after scrubbing, switching cases, or leaving the page.
      if (video.readyState < 1) await waitMedia("loadedmetadata", signal);
      if (token !== revision) return;
      video.currentTime = frame / fps;
      if (video.seeking) await waitMedia("seeked", signal);
      if (token !== revision) return;
      setStatus("playing_video"); schedule(token);
      await video.play();
      if (token !== revision && !playing) video.pause();
    } catch (_) {
      if (token === revision) { stop(); setStatus("video_error"); }
    }
  }
  function selectCase() {
    stop(); clip = data.cases.find(item => item.id === $("case").value) || data.cases[0];
    const sequence = clip.sequence || data.sequence;
    width = sequence.width; height = sequence.height; fps = sequence.fps;
    count = clip.frame_count || data.frame_count;
    canvas.width = width; canvas.height = height; $("time").max = count - 1;
    picture = null; cache.clear(); ctx.clearRect(0, 0, width, height);
    video.src = clip.video_data ? "data:video/mp4;base64," + clip.video_data : clip.video_file;
    video.load(); metrics(); showFrame(clip.keyframe);
  }
  function splitChanged() {
    $("divider").style.left = Number($("split").value) / 10 + "%";
    $("split").setAttribute("aria-valuetext", format("split_value", {value: number(Number($("split").value) / 10)}));
    renderCurrent();
  }
  let dragging = false;
  function drag(event) {
    const rect = $("stage").getBoundingClientRect();
    $("split").value = Math.round(ComparisonMath.clamp((event.clientX - rect.left) / rect.width, 0, 1) * 1000);
    splitChanged();
  }
  $("stage").addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    dragging = true; $("stage").setPointerCapture(event.pointerId); $("split").focus({preventScroll: true}); drag(event);
  });
  $("stage").addEventListener("pointermove", event => { if (dragging) drag(event); });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) $("stage").addEventListener(name, () => { dragging = false; });
  $("split").addEventListener("input", splitChanged);
  $("time").addEventListener("input", () => showFrame(Number($("time").value)));
  $("previous").addEventListener("click", () => showFrame(frame - 1));
  $("next").addEventListener("click", () => showFrame(frame + 1));
  $("keyframe").addEventListener("click", () => showFrame(clip.keyframe));
  $("play").addEventListener("click", play);
  $("case").addEventListener("change", selectCase);
  $("method").addEventListener("change", () => { method = $("method").value; metrics(); updatePosition(); renderCurrent(); });
  $("language").addEventListener("change", () => { language = $("language").value; translate(); });
  video.addEventListener("ended", () => showFrame(count - 1));
  video.addEventListener("error", () => { if (playing) { stop(); setStatus("video_error"); } });
  document.addEventListener("visibilitychange", () => { if (document.hidden) showFrame(frame); });
  window.addEventListener("pagehide", stop);
  translate(); selectCase();
})();
