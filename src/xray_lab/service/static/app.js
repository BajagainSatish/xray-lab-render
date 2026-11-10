(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 25 * 1024 * 1024;
  const ALLOWED = ["image/png", "image/jpeg", "image/jpg", "application/dicom"];

  const PLAIN = Object.freeze({
    Atelectasis: "Part of a lung collapses or does not inflate fully.",
    Consolidation: "Air spaces fill with fluid or cells, so an area looks solid.",
    Infiltration: "A broad term for material in the lung where air should be.",
    Pneumothorax: "Air leaks around a lung, which can partly collapse it.",
    Edema: "Extra fluid in the lungs, often because of the heart.",
    Emphysema: "Damaged lung tissue makes it harder to push air out.",
    Fibrosis: "Scarring makes lung tissue stiff.",
    Effusion: "Fluid collects in the space around a lung.",
    Pneumonia: "Infection inflames the lungs and can fill air spaces.",
    Pleural_Thickening: "The lining around the lung becomes thicker.",
    Cardiomegaly: "The heart looks larger than expected on the image.",
    Nodule: "A small, rounded spot in the lung.",
    Mass: "A larger rounded abnormality; it may or may not be harmful.",
    Hernia: "Part of the abdomen pushes up through the diaphragm.",
    "Lung Lesion": "A general term for an abnormal area in the lung.",
    Fracture: "A broken bone, such as a rib.",
    "Lung Opacity": "A whiter area where something is denser than air.",
    "Enlarged Cardiomediastinum": "The central chest area looks wider than expected.",
    "No Finding": "The original report mentioned no finding; not a model output.",
  });

  let guideOpener = null;

  const state = {
    items: [],
    model: null,
    selectedId: null,
    currentReference: null,
    lastResult: null,
    lastFile: null,
    busy: false,
    showAll: false,
    view: "",
    findingStatus: "",
    previewUrl: null,
    explaining: false,
    explainToken: 0,
    validationLoaded: false,
    externalItems: [],
    externalSelectedId: null,
    externalSplit: "test",
    externalLabel: "all",
    externalLimit: 48,
    externalLoaded: false,
    externalToken: 0,
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function setStatus(text, kind = "") {
    const node = $("status");
    node.textContent = text;
    node.className = `status ${kind}`.trim();
  }

  function showError(message) {
    const node = $("error");
    if (!message) {
      node.hidden = true;
      node.textContent = "";
      return;
    }
    node.hidden = false;
    node.textContent = message;
  }

  async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
    return response.json();
  }

  function buildLabelFilter() {
    const counts = new Map();
    for (const item of state.items) {
      for (const label of item.finding_labels.split("|")) {
        counts.set(label, (counts.get(label) || 0) + 1);
      }
    }
    const labels = [...counts.keys()].sort((a, b) => {
      if (a === "No Finding") return -1;
      if (b === "No Finding") return 1;
      return a.localeCompare(b);
    });
    const select = $("label-filter");
    for (const label of labels) {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = `${label} (${counts.get(label)})`;
      select.append(option);
    }
  }


  function updateFilterCounts() {
    const total = state.items.length;
    const normal = state.items.filter((item) => referenceLabels(item).length === 0).length;
    const findings = total - normal;
    const views = { PA: 0, AP: 0 };
    for (const item of state.items) {
      if (item.view_position in views) views[item.view_position] += 1;
    }
    for (const button of document.querySelectorAll(".segmented [data-status]")) {
      const key = button.dataset.status || "";
      if (key === "") button.textContent = `All studies (${total})`;
      else if (key === "findings") button.textContent = `With finding (${findings})`;
      else if (key === "normal") button.textContent = `No Finding (${normal})`;
    }
    for (const button of document.querySelectorAll(".segmented [data-view]")) {
      const key = button.dataset.view || "";
      if (key === "") button.textContent = `All views (${total})`;
      else if (key in views) button.textContent = `${key} (${views[key]})`;
    }
  }

  function renderGallery() {
    const query = $("search").value.trim().toLowerCase();
    const label = $("label-filter").value;
    const filtered = state.items.filter((item) => {
      if (state.view && item.view_position !== state.view) return false;
      if (query && !item.sample_id.toLowerCase().includes(query)) return false;
      if (label && !item.finding_labels.split("|").includes(label)) return false;
      const refs = referenceLabels(item);
      if (state.findingStatus === "findings" && refs.length === 0) return false;
      if (state.findingStatus === "normal" && refs.length > 0) return false;
      return true;
    });

    $("gallery-count").textContent =
      `${filtered.length} of ${state.items.length} studies · click one to score it`;

    const fragment = document.createDocumentFragment();
    for (const item of filtered) {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "tile";
      if (item.sample_id === state.selectedId) tile.classList.add("active");
      tile.setAttribute("aria-pressed", item.sample_id === state.selectedId ? "true" : "false");
      tile.setAttribute("aria-label",
        `${item.sample_id}, ${item.finding_labels}, ${item.view_position} view`);

      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = item.thumbnail_url || item.url;
      image.alt = "";

      const meta = el("span", "tile-meta");
      meta.append(
        el("span", "tile-label", item.finding_labels.replaceAll("|", " · ")),
        el("span", "tile-sub", `${item.view_position} · ${item.sample_id.replace(".png", "")}`),
      );
      tile.append(image, meta);
      tile.addEventListener("click", () => selectDemo(item));
      fragment.append(tile);
    }
    $("gallery").replaceChildren(fragment);
    $("gallery-empty").hidden = filtered.length !== 0;
  }

  function setViewerLoading(text) {
    const node = $("viewer-loading");
    node.textContent = text;
    node.hidden = false;
  }

  function setViewerBackground(src) {
    const background = $("viewer-bg");
    if (src) {
      background.src = src;
      background.hidden = false;
    } else {
      background.removeAttribute("src");
      background.hidden = true;
    }
  }

  function setPreview(file) {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    $("preview").src = state.previewUrl;
    $("preview").hidden = false;
    $("viewer-loading").hidden = true;
    setViewerBackground(state.previewUrl);
    $("upload-hint").textContent = `${file.name || "upload"} · ${formatBytes(file.size)}`;
  }

  function prepareDicomPreview(file) {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
    $("preview").hidden = true;
    setViewerBackground(null);
    setViewerLoading("Decoding DICOM…");
    $("upload-hint").textContent = `${file.name} · ${formatBytes(file.size)} · DICOM`;
  }

  function revokePreview() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
    $("preview").removeAttribute("src");
    $("preview").hidden = true;
    setViewerBackground(null);
  }

  function resetAnalysis(title, subtitle) {
    $("analysis-title").textContent = title;
    $("analysis-sub").textContent = subtitle;
    $("viewer-top").hidden = true;
    $("viewer-badge").hidden = true;
    $("reference").hidden = true;
    $("reference").replaceChildren();
    $("meta-chips").replaceChildren();
    $("meta").replaceChildren();
    $("details").hidden = true;
    $("toggle-details").textContent = "Details";
    $("scores").replaceChildren();
    $("explain-label").replaceChildren();
    $("explain-image").hidden = true;
    $("explain-image").removeAttribute("src");
    $("explain-caption").hidden = true;
    setExplainStatus("");
    showError(null);
  }

  function revealAnalysis(title, subtitle) {
    $("analysis").hidden = false;
    revokePreview();
    resetAnalysis(title, subtitle);
    setStatus("Loading…", "busy");
    $("viewer-loading").hidden = false;
    $("viewer-loading").textContent = "Preparing image…";
    $("analysis").scrollIntoView?.({ behavior: "smooth", block: "start" });
  }

  async function selectDemo(item) {
    if (state.busy) return;
    state.selectedId = item.sample_id;
    renderGallery();
    revealAnalysis(item.sample_id, `${item.finding_labels} · ${item.view_position} view`);
    try {
      const response = await fetch(item.url);
      if (!response.ok) throw new Error(`Could not load image (HTTP ${response.status}).`);
      const blob = await response.blob();
      const file = new File([blob], item.sample_id, { type: blob.type || "image/png" });
      setPreview(file);
      await predict(file, item);
    } catch (error) {
      setStatus("Error", "error");
      showError(error.message);
      $("viewer-loading").hidden = true;
    }
  }

  function rejectUpload(name, message) {
    revealAnalysis(name || "Upload", "Not scored");
    $("viewer-loading").hidden = true;
    setStatus("Error", "error");
    showError(message);
  }

  async function onFile(file) {
    if (!file) return;
    const isDicom = /\.dcm$/i.test(file.name || "") || file.type === "application/dicom";
    const nameOk = /\.(png|jpe?g|dcm)$/i.test(file.name || "");
    if (!ALLOWED.includes(file.type) && !nameOk) {
      rejectUpload(file.name, "Please choose a PNG, JPEG, or DICOM image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      rejectUpload(file.name, `That file is ${formatBytes(file.size)}; the limit is 25 MB.`);
      return;
    }
    state.selectedId = null;
    renderGallery();
    revealAnalysis(file.name, isDicom ? "DICOM upload" : "Uploaded image");
    if (isDicom) prepareDicomPreview(file);
    else setPreview(file);
    await predict(file, null);
  }

  async function predict(file, item) {
    if (state.busy) return;
    state.busy = true;
    state.lastFile = file;
    showError(null);
    setStatus("Scoring…", "busy");
    const form = new FormData();
    form.append("file", file, file.name || "upload.png");
    try {
      const response = await fetch("/v1/predict", { method: "POST", body: form });
      let body = null;
      try { body = await response.json(); } catch { /* non-JSON error */ }
      if (!response.ok) {
        throw new Error(body?.detail?.message || `Request failed (HTTP ${response.status}).`);
      }
      renderResult(body, item);
      setStatus(`Scored in ${Math.round(body.timing_ms.total)} ms`);
    } catch (error) {
      setStatus("Error", "error");
      showError(error.message);
      $("viewer-loading").hidden = true;
    } finally {
      state.busy = false;
    }
  }

  function referenceLabels(item) {
    if (!item) return [];
    return item.finding_labels.split("|").filter((label) => label !== "No Finding");
  }

  function renderResult(data, item) {
    state.currentReference = item;
    state.lastResult = data;
    const predictions = [...data.predictions].sort(
      (a, b) => b.sigmoid_score - a.sigmoid_score);
    const top = predictions[0];

    $("analysis-title").textContent = data.image.filename || "Uploaded image";
    $("analysis-sub").textContent =
      `${data.image.format} · ${data.image.mode} · ${data.image.shape_hw.join("×")} · ${formatBytes(data.image.bytes)}`;
    $("viewer-badge").textContent =
      `${data.image.format} · ${data.image.shape_hw.join("×")}`;
    $("viewer-badge").hidden = false;

    if (data.image.preview_png_base64) {
      const source = `data:image/png;base64,${data.image.preview_png_base64}`;
      $("preview").src = source;
      $("preview").hidden = false;
      setViewerBackground(source);
    }
    $("viewer-loading").hidden = true;

    $("top-label").textContent = top.label;
    $("top-score").textContent = top.sigmoid_score.toFixed(3);
    $("viewer-top").hidden = false;

    renderReference(data, item);
    renderChips(data);
    renderScores(predictions);
    populateExplainSelect(predictions);
    renderMeta(data);
  }

  function renderReference(data, item) {
    const node = $("reference");
    node.replaceChildren();
    node.hidden = false;

    if (!item) {
      node.append(
        el("strong", "", "Uploaded image. "),
        el("span", "", "No NIH reference label is attached; the model returns scores only."),
      );
      return;
    }

    const refs = referenceLabels(item);
    if (refs.length === 0) {
      node.append(
        el("strong", "", "NIH reference: No Finding. "),
        el("span", "", "No Finding is not a model output, so there is no direct match to check. "),
        el("span", "", "Labels are report-mined and noisy."),
      );
      return;
    }

    const top = data.predictions.reduce(
      (best, row) => (row.sigmoid_score > best.sigmoid_score ? row : best));
    const matches = refs.includes(top.label);
    node.append(el("strong", "", `NIH reference: ${refs.join(" · ")}. `));
    for (const label of refs) {
      const row = data.predictions.find((prediction) => prediction.label === label);
      if (!row) continue;
      node.append(el("div", "", `${label} — ${PLAIN[label] || "no description"} ` +
        `(rank ${row.rank}/18, score ${row.sigmoid_score.toFixed(3)})`));
    }
    node.append(el("div", "", matches
      ? `Top output (${top.label}) matches one of the reference labels.`
      : `Top output (${top.label}) differs from the reference labels.`));
    node.append(el("div", "micro",
      "The NIH label comes from the original report and can be incomplete or wrong. It is not expert ground truth."));
  }

  function renderChips(data) {
    const container = $("meta-chips");
    const chips = [
      `${data.image.format} · ${data.image.shape_hw.join("×")}`,
      `${Math.round(data.timing_ms.total)} ms total`,
      data.model.id,
    ];
    if (data.image.dicom) {
      const dicom = data.image.dicom;
      chips.push(`${dicom.photometric_interpretation} · ${dicom.modality || "modality n/a"}`);
      chips.push(dicom.window_center !== null
        ? `window ${dicom.window_center}/${dicom.window_width}`
        : `window ${Math.round(dicom.window_low)}–${Math.round(dicom.window_high)}`);
    }
    container.replaceChildren(...chips.map((text) => el("span", "", text)));
  }

  function renderScores(sortedPredictions) {
    const refs = referenceLabels(state.currentReference);
    let visible;
    if (state.showAll) {
      visible = sortedPredictions;
    } else {
      visible = sortedPredictions.slice(0, 8);
      for (const label of refs) {
        if (!visible.some((row) => row.label === label)) {
          const row = sortedPredictions.find((prediction) => prediction.label === label);
          if (row) visible.push(row);
        }
      }
    }

    const fragment = document.createDocumentFragment();
    for (const row of visible) {
      const item = el("div", "score-row");
      if (refs.includes(row.label)) item.classList.add("ref");

      const top = el("div", "score-top");
      const name = el("span", "score-name");
      name.append(el("span", "rank", String(row.rank)), el("span", "", row.label));
      top.append(name, el("span", "score-value", row.sigmoid_score.toFixed(3)));

      const track = el("div", "track");
      const fill = el("span", "fill");
      fill.style.width = `${Math.max(1, Math.min(100, row.sigmoid_score * 100))}%`;
      track.append(fill);
      item.title = `${PLAIN[row.label] || row.label}\nlogit ${row.logit.toFixed(3)} · XRV-rescaled ${row.xrv_rescaled_score.toFixed(3)}`;
      item.append(top, track);
      fragment.append(item);
    }
    $("scores").replaceChildren(fragment);
    $("toggle-scores").textContent = state.showAll ? "Show top 8" : "Show all 18";
  }

  function renderMeta(data) {
    const rows = [
      ["File", data.image.filename || "upload"],
      ["SHA256", data.image.sha256],
      ["Format", `${data.image.format} · ${data.image.mode} · ${data.image.shape_hw.join("×")}`],
      ["Bytes", formatBytes(data.image.bytes)],
      ["Model", data.model.id],
      ["Weights SHA256", data.model.sha256 || "not recorded"],
      ["Decode", `${data.timing_ms.decode} ms`],
      ["Preprocess", `${data.timing_ms.preprocess} ms`],
      ["Inference", `${data.timing_ms.inference} ms`],
      ["Total", `${data.timing_ms.total} ms`],
    ];
    const dicom = data.image.dicom;
    if (dicom) {
      rows.splice(4, 0,
        ["Modality", dicom.modality || "not recorded"],
        ["View", dicom.view_position || "not recorded"],
        ["Photometric", dicom.photometric_interpretation],
        ["Transfer syntax", dicom.transfer_syntax_name],
        ["Window", dicom.window_center !== null
          ? `center ${dicom.window_center}, width ${dicom.window_width} (tag)`
          : `${Math.round(dicom.window_low)}–${Math.round(dicom.window_high)} (${dicom.window_source})`],
        ["De-identification", "not performed; identifiers are not returned"]);
    }
    const list = $("meta");
    list.replaceChildren();
    for (const [term, value] of rows) {
      list.append(el("dt", "", term), el("dd", "", value));
    }
  }

  function populateExplainSelect(predictions) {
    const select = $("explain-label");
    const options = predictions.map((row) => {
      const option = document.createElement("option");
      option.value = row.label;
      option.textContent = `${row.label} (${row.sigmoid_score.toFixed(3)})`;
      return option;
    });
    select.replaceChildren(...options);
    updateExplainHelp();
  }

  function updateExplainHelp() {
    const label = $("explain-label").value;
    $("explain-help").textContent = label && PLAIN[label]
      ? `${label}: ${PLAIN[label]}`
      : "Choose a finding to see a plain-language explanation.";
  }

  function openGuide() {
    const dialog = $("guide");
    guideOpener = document.activeElement;
    if (typeof dialog.showModal === "function") {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
    document.body.classList.add("guide-open");
    $("guide-close").focus?.();
  }

  function closeGuide() {
    const dialog = $("guide");
    if (typeof dialog.close === "function") dialog.close();
    else dialog.removeAttribute("open");
    document.body.classList.remove("guide-open");
    if (guideOpener && typeof guideOpener.focus === "function") guideOpener.focus();
  }

  function setExplainStatus(text, isError = false) {
    const node = $("explain-status");
    node.textContent = text;
    node.className = `micro${isError ? " error" : ""}`;
  }

  function resetExplanation() {
    state.explainToken += 1;
    $("explain-image").hidden = true;
    $("explain-image").removeAttribute("src");
    $("explain-caption").hidden = true;
    setExplainStatus("");
  }

  async function explainFinding() {
    if (!state.lastFile || state.explaining) return;
    const label = $("explain-label").value;
    if (!label) return;
    const token = state.explainToken;
    const button = $("explain-btn");
    state.explaining = true;
    button.disabled = true;
    button.textContent = "Computing…";
    setExplainStatus("Computing Grad-CAM for the selected output…");

    const form = new FormData();
    form.append("file", state.lastFile, state.lastFile.name || "upload.png");
    form.append("label", label);
    try {
      const response = await fetch("/v1/explain", { method: "POST", body: form });
      let body = null;
      try { body = await response.json(); } catch { /* non-JSON error */ }
      if (token !== state.explainToken) return;
      if (!response.ok) {
        throw new Error(body?.detail?.message || `Request failed (HTTP ${response.status}).`);
      }
      $("explain-image").src = `data:image/png;base64,${body.overlay_png_base64}`;
      $("explain-image").hidden = false;
      const target = body.target;
      $("explain-caption").textContent =
        `${target.label}: score ${target.sigmoid_score.toFixed(3)}, rank ${target.rank}/18. ` +
        `Class-targeting check ${body.class_targeting.mean_abs_difference.toFixed(3)}; ` +
        `total ${Math.round(body.timing_ms.total)} ms. ` +
        "Coarse saliency, not a lesion mask.";
      $("explain-caption").hidden = false;
      setExplainStatus("");
    } catch (error) {
      if (token === state.explainToken) setExplainStatus(error.message, true);
    } finally {
      state.explaining = false;
      button.disabled = false;
      button.textContent = "Explain";
    }
  }

  function clearSelection() {
    state.selectedId = null;
    state.currentReference = null;
    state.lastResult = null;
    state.lastFile = null;
    state.showAll = false;
    $("analysis").hidden = true;
    revokePreview();
    $("upload-hint").textContent = "No file selected";
    showError(null);
    renderGallery();
    $("library").scrollIntoView?.({ behavior: "smooth", block: "start" });
  }


  function percent(value) {
    if (value === null || value === undefined) return "—";
    return `${(value * 100).toFixed(1)}%`;
  }

  function metricCard(label, value, note) {
    const card = el("div", "metric");
    card.append(el("span", "label", label), el("span", "value", value),
                el("span", "note", note));
    return card;
  }

  function validationFigure(data, name, caption) {
    const meta = data.figures?.[name];
    if (!meta?.url) return null;
    const figure = el("figure", "figure-card");
    const image = document.createElement("img");
    image.src = meta.url;
    image.loading = "lazy";
    image.alt = caption;
    figure.append(image, el("figcaption", "", caption));
    return figure;
  }

  function renderValidationMissing() {
    const container = $("validation-content");
    container.replaceChildren();
    $("validation-pill").textContent = "Not run";
    const box = el("div", "validation-empty");
    box.append(el("p", "", "No external validation run was found on this machine."),
               el("p", "", "Run: "),
               el("code", "", "python scripts/run_external_validation.py"));
    container.append(box);
  }


  function buildExternalPicker() {
    const card = el("div", "validation-card validation-picker");
    const head = el("div", "picker-head");
    const copy = el("div");
    copy.append(
      el("h3", "", "Pick an external image"),
      el("p", "", "Choose a SIIM-ACR study and score it now. The reference label and mask come from the dataset, not the model."),
    );
    const count = el("span", "validation-pill", "Loading…");
    count.id = "external-count";
    head.append(copy, count);

    const controls = el("div", "picker-controls");
    const splitGroup = el("div", "segmented");
    splitGroup.id = "external-split";
    for (const [value, label] of [["test", "Test"], ["validation", "Validation"], ["all", "All"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.split = value;
      button.textContent = label;
      button.className = value === state.externalSplit ? "active" : "";
      button.addEventListener("click", () => {
        state.externalSplit = value;
        state.externalLimit = 48;
        for (const node of splitGroup.children) {
          node.classList.toggle("active", node.dataset.split === value);
        }
        renderExternalGallery();
      });
      splitGroup.append(button);
    }

    const labelGroup = el("div", "segmented");
    labelGroup.id = "external-label";
    for (const [value, label] of [["all", "All labels"], ["1", "Pneumothorax"], ["0", "No pneumothorax"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.label = value;
      button.textContent = label;
      button.className = value === state.externalLabel ? "active" : "";
      button.addEventListener("click", () => {
        state.externalLabel = value;
        state.externalLimit = 48;
        for (const node of labelGroup.children) {
          node.classList.toggle("active", node.dataset.label === value);
        }
        renderExternalGallery();
      });
      labelGroup.append(button);
    }

    const search = document.createElement("input");
    search.id = "external-search";
    search.type = "search";
    search.placeholder = "Search image ID";
    search.setAttribute("aria-label", "Search external image ID");
    search.addEventListener("input", () => {
      state.externalLimit = 48;
      renderExternalGallery();
    });
    controls.append(splitGroup, labelGroup, search);

    const layout = el("div", "external-layout");
    const wrap = el("div", "external-gallery-wrap");
    const gallery = el("div", "external-gallery");
    gallery.id = "external-gallery";
    const more = document.createElement("button");
    more.type = "button";
    more.id = "external-more";
    more.className = "ghost external-more";
    more.textContent = "Show more";
    more.hidden = true;
    more.addEventListener("click", () => {
      state.externalLimit += 48;
      renderExternalGallery();
    });
    const all = document.createElement("button");
    all.type = "button";
    all.id = "external-all";
    all.className = "ghost external-more";
    all.textContent = "Show all";
    all.hidden = true;
    all.addEventListener("click", () => {
      state.externalLimit = state.externalItems.length || 0;
      renderExternalGallery();
    });
    const pager = el("div", "external-pager");
    pager.append(more, all);
    wrap.append(gallery, pager);

    const viewer = el("div", "external-viewer");
    viewer.id = "external-viewer";
    viewer.append(el("p", "empty", "Select an image to score it."));
    layout.append(wrap, viewer);

    card.append(head, controls, layout, el("p", "validation-note",
      "This is a demonstration picker. Scoring one image does not update the aggregate metrics above."));
    return card;
  }

  async function loadExternalGallery() {
    if (state.externalLoaded) {
      renderExternalGallery();
      return;
    }
    state.externalLoaded = true;
    try {
      const data = await fetchJSON("/v1/validation/pneumothorax/images");
      state.externalItems = data.items || [];
      $("external-count").textContent = `${data.count} images · ${data.positive_count} positive`;
      renderExternalGallery();
    } catch (error) {
      state.externalLoaded = false;
      const count = $("external-count");
      if (count) count.textContent = "Not built";
      const gallery = $("external-gallery");
      if (gallery) {
        gallery.replaceChildren();
        const empty = el("div", "validation-empty");
        empty.append(
          el("p", "", "The external image gallery is not available."),
          el("p", "", "Run: "),
          el("code", "", "python scripts/build_external_gallery.py"),
        );
        gallery.append(empty);
      }
    }
  }

  function renderExternalGallery() {
    const gallery = $("external-gallery");
    if (!gallery) return;
    const query = ($("external-search")?.value || "").trim().toLowerCase();
    const filtered = state.externalItems.filter((item) => {
      if (state.externalSplit !== "all" && item.split !== state.externalSplit) return false;
      if (state.externalLabel !== "all" && String(item.label) !== state.externalLabel) return false;
      if (query && !`${item.id} ${item.image_id}`.toLowerCase().includes(query)) return false;
      return true;
    });
    const visible = filtered.slice(0, state.externalLimit);
    const count = $("external-count");
    if (count && state.externalItems.length) {
      count.textContent = `${filtered.length} matching · ${state.externalItems.length} total`;
    }
    const fragment = document.createDocumentFragment();
    for (const item of visible) {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "external-tile";
      if (item.id === state.externalSelectedId) tile.classList.add("active");
      tile.setAttribute("aria-pressed", item.id === state.externalSelectedId ? "true" : "false");
      tile.setAttribute("aria-label",
        `${item.id}, ${item.label_text}, ${item.split} split`);
      tile.title = `${item.label_text} · ${item.split} · ${item.image_id}`;
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = item.thumbnail_url;
      image.alt = "";
      tile.append(image);
      if (item.label === 1) {
        tile.append(el("span", "external-tile-badge", "PTX"));
      }
      tile.append(el("span", "external-tile-meta",
        `${item.split === "test" ? "Test" : "Val"} · ${item.id}`));
      tile.addEventListener("click", () => selectExternalImage(item));
      fragment.append(tile);
    }
    if (!visible.length) {
      fragment.append(el("p", "empty", "No external images match these filters."));
    }
    gallery.replaceChildren(fragment);
    const more = $("external-more");
    if (more) {
      more.hidden = filtered.length <= state.externalLimit;
      more.textContent = `Show ${Math.min(48, filtered.length - state.externalLimit)} more`;
    }
    const all = $("external-all");
    if (all) all.hidden = filtered.length <= state.externalLimit;
  }

  function selectExternalImage(item) {
    state.externalSelectedId = item.id;
    renderExternalGallery();
    const viewer = $("external-viewer");
    if (!viewer) return;

    viewer.replaceChildren();
    const frame = el("div", "external-frame");
    const image = document.createElement("img");
    image.id = "external-image";
    image.src = item.url;
    image.alt = `${item.label_text} external chest X-ray`;
    const overlay = document.createElement("img");
    overlay.id = "external-overlay";
    overlay.src = item.overlay_url || "";
    overlay.alt = "Reference pneumothorax mask overlay";
    overlay.hidden = true;
    frame.append(image, overlay);

    const toolbar = el("div", "external-toolbar");
    toolbar.append(el("span", "external-split-chip",
      `${item.split === "test" ? "Test" : "Validation"} split`));
    const maskButton = document.createElement("button");
    maskButton.type = "button";
    maskButton.className = "ghost";
    maskButton.textContent = "Show reference mask";
    maskButton.hidden = !item.overlay_url;
    maskButton.addEventListener("click", () => {
      const show = overlay.hidden;
      overlay.hidden = !show;
      maskButton.textContent = show ? "Hide reference mask" : "Show reference mask";
    });
    const explainButton = document.createElement("button");
    explainButton.type = "button";
    explainButton.className = "ghost";
    explainButton.textContent = "Explain";
    explainButton.addEventListener("click", () => explainExternalImage(item, explainButton));
    toolbar.append(maskButton, explainButton);

    const result = el("div", "external-result");
    result.id = "external-result";
    result.append(el("p", "empty", "Scoring this image…"));
    const explainWrap = el("div", "external-explain");
    explainWrap.id = "external-explain";
    explainWrap.hidden = true;
    const explainImage = document.createElement("img");
    explainImage.id = "external-explain-image";
    explainImage.alt = "Grad-CAM overlay for the selected external image";
    const explainCaption = el("p", "validation-note", "");
    explainCaption.id = "external-explain-caption";
    explainWrap.append(explainImage, explainCaption);
    viewer.append(frame, toolbar, result, explainWrap, el("p", "validation-note",
      "The red overlay is the dataset's reference mask, not a model explanation. The model returns one pneumothorax score for this image; Explain shows a coarse Grad-CAM."));
    if (window.matchMedia?.("(max-width: 900px)").matches) {
      viewer.scrollIntoView?.({ behavior: "smooth", block: "start" });
    }

    const token = ++state.externalToken;
    fetch("/v1/validation/pneumothorax/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: item.id }),
    }).then(async (response) => {
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(body?.detail?.message || `HTTP ${response.status}`);
      }
      if (token !== state.externalToken) return;
      renderExternalPrediction(body);
    }).catch((error) => {
      if (token !== state.externalToken) return;
      const node = $("external-result");
      if (node) node.replaceChildren(el("p", "validation-empty",
        `Could not score this image: ${error.message}`));
    });
  }


  async function explainExternalImage(item, button) {
    const wrap = $("external-explain");
    const image = $("external-explain-image");
    const caption = $("external-explain-caption");
    if (!wrap || !image || !caption) return;
    button.disabled = true;
    button.textContent = "Explaining…";
    try {
      const imageResponse = await fetch(item.url);
      if (!imageResponse.ok) {
        throw new Error(`image HTTP ${imageResponse.status}`);
      }
      const blob = await imageResponse.blob();
      const form = new FormData();
      form.append("file", blob, `${item.id}.png`);
      form.append("label", "Pneumothorax");
      const response = await fetch("/v1/explain", { method: "POST", body: form });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(body?.detail?.message || `HTTP ${response.status}`);
      }
      image.src = `data:image/png;base64,${body.overlay_png_base64}`;
      wrap.hidden = false;
      caption.textContent =
        `Grad-CAM for Pneumothorax · score ${body.target.sigmoid_score.toFixed(3)} · ` +
        `${Math.round(body.timing_ms.total)} ms. Coarse 7×7 map; not a lesion mask.`;
      wrap.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
    } catch (error) {
      image.removeAttribute("src");
      wrap.hidden = false;
      caption.textContent = `Could not explain this image: ${error.message}`;
    } finally {
      button.disabled = false;
      button.textContent = "Explain";
    }
  }

  function renderExternalPrediction(data) {
    const node = $("external-result");
    if (!node) return;
    const prediction = data.prediction;
    node.replaceChildren();
    const head = el("div", "external-result-head");
    head.append(el("div", "external-score", prediction.sigmoid_score.toFixed(3)));
    const details = el("div", "external-details");
    details.append(
      el("strong", "", data.reference.label_text),
      el("span", "", ` · threshold ${prediction.threshold.toFixed(3)}`),
      el("span", "", ` · model says ${prediction.positive ? "positive" : "negative"}`),
    );
    head.append(details);
    const badgeClass = data.match ? "agree" : "disagree";
    head.append(el("span", `external-badge ${badgeClass}`,
      data.match ? "Agrees with reference" : "Differs from reference"));
    node.append(head);

    const track = el("div", "external-track");
    const fill = el("span", "external-fill");
    fill.style.width = `${Math.max(0.5, Math.min(100, prediction.sigmoid_score * 100))}%`;
    const marker = el("span", "external-marker");
    marker.style.left = `${Math.min(100, prediction.threshold * 100)}%`;
    marker.title = `threshold ${prediction.threshold.toFixed(3)}`;
    track.append(fill, marker);
    node.append(track);

    const rows = el("dl", "external-stats");
    const values = [
      ["Reference", data.reference.label_text],
      ["Model score", prediction.sigmoid_score.toFixed(3)],
      ["Threshold", prediction.threshold.toFixed(3)],
      ["Prediction", prediction.positive ? "positive" : "negative"],
      ["Reference mask", `${percent(data.reference.mask_area_fraction)} of image`],
      ["Scored in", `${Math.round(data.timing_ms.total)} ms`],
      ["Threshold source", data.threshold_source],
    ];
    for (const [term, value] of values) {
      rows.append(el("dt", "", term), el("dd", "", value));
    }
    node.append(rows);
    node.append(el("p", "validation-note",
      data.image.split === "validation"
        ? "This image is from the validation split used to choose the threshold; treat its agreement as illustrative, not independent."
        : "This image is from the test split. The threshold was chosen on a separate validation split."));

    const scoreDetails = document.createElement("details");
    scoreDetails.className = "external-score-details";
    const summary = document.createElement("summary");
    summary.textContent = `All ${data.predictions.length} model outputs`;
    const list = el("div", "external-score-list");
    const sorted = [...data.predictions].sort(
      (a, b) => b.sigmoid_score - a.sigmoid_score);
    for (const row of sorted) {
      const item = el("div", "score-row");
      if (row.label === "Pneumothorax") item.classList.add("ref");
      const top = el("div", "score-top");
      const name = el("span", "score-name");
      name.append(el("span", "rank", String(row.rank)), el("span", "", row.label));
      top.append(name, el("span", "score-value", row.sigmoid_score.toFixed(3)));
      const track = el("div", "track");
      const fill = el("span", "fill");
      fill.style.width = `${Math.max(1, Math.min(100, row.sigmoid_score * 100))}%`;
      track.append(fill);
      item.append(top, track);
      list.append(item);
    }
    scoreDetails.append(summary, list, el("p", "validation-note",
      "Only Pneumothorax has a reference label in this dataset; the other outputs are shown for transparency and are not validated here."));
    node.append(scoreDetails);
  }

  function renderValidation(data) {
    const container = $("validation-content");
    container.replaceChildren();
    const test = data.test.at_selected_threshold;
    const ci = data.test.bootstrap.ci95;
    const validation = data.validation;
    const dataset = data.dataset;
    $("validation-pill").textContent =
      `SIIM-ACR · ${test.n} test images · 1 finding`;

    container.append(el("div", "validation-banner",
      "One finding out of 18. This is an external single-finding check, not a " +
      "diagnosis and not clinical validation."));

    const metrics = el("div", "metric-grid");
    metrics.append(
      metricCard("AUROC", test.auroc.toFixed(3),
        `95% CI ${ci.auroc[0].toFixed(3)}–${ci.auroc[1].toFixed(3)}`),
      metricCard("AUPRC", test.auprc.toFixed(3),
        `prevalence ${percent(test.positives / test.n)}`),
      metricCard("Sensitivity", percent(test.sensitivity),
        `threshold ${validation.threshold.toFixed(3)}`),
      metricCard("Specificity", percent(test.specificity),
        "same threshold"),
      metricCard("PPV", percent(test.ppv), `NPV ${percent(test.npv)}`),
      metricCard("Brier", data.calibration.brier_score.toFixed(3),
        "lower is better · uncalibrated"),
    );
    container.append(metrics);
    container.append(el("p", "validation-note",
      "AUROC: 0.5 is chance and 1.0 is perfect ranking. AUPRC depends on prevalence. " +
      "PPV/NPV depend on prevalence and are not transportable to another population."));

    container.append(buildExternalPicker());
    if (state.externalLoaded) renderExternalGallery();
    else if (!$("validation").hidden) loadExternalGallery();

    const split = el("div", "validation-split");

    const dataCard = el("div", "validation-card");
    dataCard.append(
      el("h3", "", "Dataset and split"),
      el("p", "", `${dataset.dataset_id} — SIIM-ACR Pneumothorax Segmentation Challenge (2019), 256 px educational subset.`),
      el("p", "", `${test.n} test images (${test.positives} positive). A separate validation split of ${validation.n} images (${validation.positives} positive) chose the threshold. The test split was used only for the reported evaluation; no test-based tuning.`),
      el("p", "", "The original challenge has ~12,000 DICOMs at higher resolution. This subset is downscaled and may erase small pneumothoraces (reported median lesion area 0.76% of the image)."),
    );
    const sourceLink = document.createElement("a");
    sourceLink.href = dataset.source.page;
    sourceLink.target = "_blank";
    sourceLink.rel = "noreferrer";
    sourceLink.textContent = "Subset source";
    const termsLink = document.createElement("a");
    termsLink.href = dataset.source.original_terms;
    termsLink.target = "_blank";
    termsLink.rel = "noreferrer";
    termsLink.textContent = "Original challenge terms";
    dataCard.append(el("p", "", "Links: "), sourceLink, el("span", "", " · "), termsLink);

    const thresholdCard = el("div", "validation-card");
    thresholdCard.append(
      el("h3", "", "Operating point"),
      el("p", "", `Youden J on validation = ${validation.youden_j.toFixed(3)}; threshold = ${validation.threshold.toFixed(3)}. It was applied once to the test split.`),
      el("p", "", `At 0.5, the model would call 0 of ${test.positives} positive images positive. The scores are uncalibrated rankings, not probabilities.`),
      el("p", "", `At the model's own XRV threshold (${data.test.at_xrv_operating_threshold.threshold.toFixed(3)}), sensitivity is ${percent(data.test.at_xrv_operating_threshold.sensitivity)} and specificity is ${percent(data.test.at_xrv_operating_threshold.specificity)}.`),
      el("p", "", `Confusion at the selected threshold: TP ${test.tp}, FP ${test.fp}, FN ${test.fn}, TN ${test.tn}.`),
      el("p", "", `PPV and NPV depend on prevalence. This test set is enriched at ${percent(test.positives / test.n)}; a real screening population will differ.`),
    );
    split.append(dataCard, thresholdCard);
    container.append(split);

    const figures = el("div", "figure-grid");
    const figureSpecs = [
      ["roc_test.png", "ROC on the external test split."],
      ["pr_test.png", "Precision-recall; the dashed line is prevalence."],
      ["score_distribution.png", "Score separation by reference label."],
      ["calibration.png", "Reliability in five equal-count bins; scores are not calibrated."],
      ["sensitivity_by_size.png", "Sensitivity by pneumothorax mask area at the selected threshold."],
      ["cam_examples.png", "Grad-CAM vs external masks — the four largest masks."],
    ];
    for (const [name, caption] of figureSpecs) {
      const figure = validationFigure(data, name, caption);
      if (figure) figures.append(figure);
    }
    container.append(figures);

    const sizeCard = el("div", "validation-card");
    sizeCard.append(el("h3", "", "Detection by lesion size"));
    const table = el("table", "size-table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const title of ["Mask area", "Positive images", "Detected", "Sensitivity"]) {
      headRow.append(el("th", "", title));
    }
    head.append(headRow);
    const body = document.createElement("tbody");
    for (const row of data.size_analysis || []) {
      const tr = document.createElement("tr");
      tr.append(el("td", "", row.bin), el("td", "", String(row.positives)),
                el("td", "", String(row.detected)), el("td", "", percent(row.sensitivity)));
      body.append(tr);
    }
    table.append(head, body);
    sizeCard.append(table, el("p", "validation-note",
      "Bin counts are small, especially in the largest bins; these differences are noisy. Downsampling to 256 px may still erase small lesions even though this split does not show a monotonic trend."));
    container.append(sizeCard);

    const camCard = el("div", "validation-card");
    camCard.append(
      el("h3", "", "Explanation audit"),
      el("p", "", `On ${data.cam.evaluated_positive_images} positive test images, the CAM peak fell inside the reference mask ${percent(data.cam.pointing_hit_rate)} of the time. Random pointing would hit ${percent(data.cam.random_pointing_baseline)} (the mean mask area). Mean CAM mass inside the mask: ${percent(data.cam.mean_cam_mass_inside)}.`),
      el("div", "cam-callout",
        "The 7×7 heatmap does not localize small pneumothoraces. Do not present it as a lesion detector; use it only as a coarse view of model sensitivity."),
    );
    container.append(camCard);

    const limits = el("div", "validation-card");
    limits.append(el("h3", "", "What this does and does not establish"));
    const list = document.createElement("ul");
    for (const text of [
      "It establishes a modest external ranking signal for one finding (AUROC 0.747), with a high-sensitivity / low-specificity operating point.",
      "It does not validate the other 17 outputs, nor a local patient population.",
      "It does not establish clinical utility, workflow benefit, or safety.",
      "The 256 px subset and missing original DICOM headers are real limitations.",
      "A local, de-identified pilot with radiologist reads is still required.",
    ]) {
      list.append(el("li", "", text));
    }
    limits.append(list);
    container.append(limits);
  }

  async function loadValidation() {
    if (state.validationLoaded) return;
    state.validationLoaded = true;
    try {
      const response = await fetch("/v1/validation/pneumothorax");
      if (response.status === 404) {
        state.validationLoaded = false;
        renderValidationMissing();
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderValidation(await response.json());
    } catch (error) {
      state.validationLoaded = false;
      $("validation-content").replaceChildren(
        el("div", "validation-empty", `Could not load validation results: ${error.message}`));
    }
  }

  function showView(view) {
    const workbench = view === "workbench";
    $("hero").hidden = !workbench;
    $("workbench-view").hidden = !workbench;
    $("validation").hidden = workbench;
    for (const button of document.querySelectorAll(".view-tabs .tab")) {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    }
    if (!workbench) loadValidation().then(loadExternalGallery);
    window.scrollTo?.({ top: 0, behavior: "smooth" });
  }

  function wireEvents() {
    const dropzone = $("dropzone");
    const input = $("file-input");

    dropzone.addEventListener("click", (event) => {
      if (event.target !== input) input.click();
    });
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    for (const type of ["dragenter", "dragover"]) {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    }
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) onFile(file);
    });
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      if (file) onFile(file);
      input.value = "";
    });

    $("search").addEventListener("input", renderGallery);
    $("label-filter").addEventListener("change", renderGallery);
    for (const button of document.querySelectorAll(".segmented [data-view]")) {
      button.addEventListener("click", () => {
        state.view = button.dataset.view || "";
        for (const sibling of document.querySelectorAll(".segmented [data-view]")) {
          const active = sibling === button;
          sibling.classList.toggle("active", active);
          sibling.setAttribute("aria-pressed", active ? "true" : "false");
        }
        renderGallery();
      });
    }
    for (const button of document.querySelectorAll(".segmented [data-status]")) {
      button.addEventListener("click", () => {
        state.findingStatus = button.dataset.status || "";
        for (const sibling of document.querySelectorAll(".segmented [data-status]")) {
          const active = sibling === button;
          sibling.classList.toggle("active", active);
          sibling.setAttribute("aria-pressed", active ? "true" : "false");
        }
        renderGallery();
      });
    }
    $("toggle-scores").addEventListener("click", () => {
      state.showAll = !state.showAll;
      if (state.lastResult) {
        renderScores([...state.lastResult.predictions].sort(
          (a, b) => b.sigmoid_score - a.sigmoid_score));
      }
    });
    $("toggle-details").addEventListener("click", () => {
      const hidden = $("details").hidden;
      $("details").hidden = !hidden;
      $("toggle-details").textContent = hidden ? "Hide details" : "Details";
    });
    $("clear-selection").addEventListener("click", clearSelection);
    $("explain-btn").addEventListener("click", explainFinding);
    $("explain-label").addEventListener("change", () => {
      resetExplanation();
      updateExplainHelp();
    });
    for (const button of document.querySelectorAll(".guide-open")) {
      button.addEventListener("click", openGuide);
    }
    $("guide-close").addEventListener("click", closeGuide);
    $("guide").addEventListener("close", () => {
      document.body.classList.remove("guide-open");
      if (guideOpener && typeof guideOpener.focus === "function") guideOpener.focus();
    });
    for (const button of document.querySelectorAll(".view-tabs .tab")) {
      button.addEventListener("click", () => {
        showView(button.dataset.view);
        history.replaceState?.(null, "", `#${button.dataset.view}`);
      });
    }
    $("guide").addEventListener("click", (event) => {
      const dialog = $("guide");
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const outside = event.clientX < rect.left || event.clientX > rect.right
        || event.clientY < rect.top || event.clientY > rect.bottom;
      if (outside) closeGuide();
    });
  }

  async function init() {
    wireEvents();
    try {
      const [model, items] = await Promise.all([
        fetchJSON("/v1/models"),
        fetchJSON("/v1/demo-images"),
      ]);
      state.model = model;
      state.items = items;
      $("model-chip").textContent = `${model.id} · ${model.output_count} outputs`;
      $("model-chip").title = model.training_description;
      buildLabelFilter();
      updateFilterCounts();
      renderGallery();
      loadValidation();
      if (location.hash === "#validation") showView("validation");
    } catch (error) {
      $("gallery-count").textContent = "Could not load the reference library.";
      $("model-chip").textContent = "API offline";
    }
  }

  init();
})();
