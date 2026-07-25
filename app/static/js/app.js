(() => {
  const $ = (sel, root = document) => root.querySelector(sel);

  const themeBtn = $("#theme-toggle");
  const form = $("#job-form");
  const drawingsInput = $("#drawings-input");
  const photosInput = $("#photos-input");
  const drawingsList = $("#drawings-list");
  const photosList = $("#photos-list");
  const startBtn = $("#start-btn");
  const progressWrap = $("#progress-wrap");
  const progressBar = $("#progress-bar");
  const progressPct = $("#progress-pct");
  const progressStep = $("#progress-step");
  const statusEl = $("#status");
  const downloadBtn = $("#download-btn");
  const viewerEl = $("#viewer");
  const layerToggles = $("#layer-toggles");
  const scaleHint = $("#scale-hint");

  const state = {
    drawings: [],
    photos: [],
    jobId: null,
    pollTimer: null,
    viewer: null,
  };

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("itd-theme", theme);
    if (themeBtn) themeBtn.textContent = theme === "dark" ? "Светлая" : "Тёмная";
    state.viewer?.draw?.();
  }

  const saved = localStorage.getItem("itd-theme");
  if (saved) applyTheme(saved);
  else if (window.matchMedia("(prefers-color-scheme: light)").matches) applyTheme("light");
  else applyTheme("dark");

  themeBtn?.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  function fmtSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function renderList(ul, files) {
    ul.innerHTML = "";
    files.forEach((f) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${f.name}</span><span class="meta">${fmtSize(f.size)}</span>`;
      ul.appendChild(li);
    });
  }

  function wireDrop(zone, input, key) {
    const open = () => input.click();
    zone.addEventListener("click", open);
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });
    ["dragenter", "dragover"].forEach((ev) => {
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.add("is-drag");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.remove("is-drag");
      });
    });
    zone.addEventListener("drop", (e) => {
      const files = [...(e.dataTransfer?.files || [])];
      if (!files.length) return;
      state[key] = files;
      input.files = e.dataTransfer.files;
      renderList(key === "drawings" ? drawingsList : photosList, files);
      updateStart();
    });
    input.addEventListener("change", () => {
      state[key] = [...input.files];
      renderList(key === "drawings" ? drawingsList : photosList, state[key]);
      updateStart();
    });
  }

  function updateStart() {
    startBtn.disabled = state.drawings.length === 0;
  }

  wireDrop($("#drawings-drop"), drawingsInput, "drawings");
  wireDrop($("#photos-drop"), photosInput, "photos");

  function setStatus(text, kind = "") {
    statusEl.textContent = text;
    statusEl.className = `status${kind ? ` ${kind}` : ""}`;
  }

  function setProgress(pct, step) {
    progressWrap.classList.add("is-on");
    progressBar.style.width = `${pct}%`;
    progressPct.textContent = `${pct}%`;
    progressStep.textContent = step || "";
  }

  function renderLayers(layerState) {
    layerToggles.innerHTML = "";
    const labels = {
      "00_FRAME": "00_Оформление",
      "00_STAMP_TEXT": "00_Текст",
      "01_WALLS": "01_Стены",
      "02_PARTITIONS": "02_Перегородки",
      "03_AXES": "03_Оси",
      "04_DIMS_GOST": "04_Размеры",
      "05_OPENINGS": "05_Окна_Двери",
    };
    Object.keys(layerState).sort().forEach((name) => {
      const id = `layer-${name}`;
      const label = document.createElement("label");
      const title = labels[name] || name;
      label.innerHTML = `<input type="checkbox" id="${id}" ${layerState[name] ? "checked" : ""}> ${title}`;
      label.querySelector("input").addEventListener("change", (e) => {
        state.viewer?.setLayer(name, e.target.checked);
      });
      layerToggles.appendChild(label);
    });
  }

  async function loadViewer(jobId) {
    const res = await fetch(`/api/jobs/${jobId}/scene`);
    if (!res.ok) throw new Error("Не удалось загрузить сцену");
    const scene = await res.json();
    if (!state.viewer) {
      state.viewer = new window.DxfWebGLViewer(viewerEl);
    }
    const layers = state.viewer.setScene(scene);
    renderLayers(layers);
  }

  async function pollJob(jobId) {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error("Не удалось получить статус");
        const data = await res.json();
        setProgress(data.progress || 0, data.step || "");
        if (data.status === "done") {
          clearInterval(state.pollTimer);
          if (data.needs_scale) {
            scaleHint?.classList.add("is-warn");
            const warn =
              data.scale_warning ||
              "Масштаб ненадёжен. Укажите контрольный отрезок в мм и запустите снова — без этого условные 12 000 мм не считать верными.";
            setStatus(`Готово. DXF готов. ${warn}`, "warn");
          } else {
            scaleHint?.classList.remove("is-warn");
            setStatus("Готово. DXF готов — откройте в AutoCAD или LibreCAD.", "ok");
          }
          downloadBtn.disabled = false;
          downloadBtn.removeAttribute("aria-disabled");
          downloadBtn.href = `/api/jobs/${jobId}/download`;
          downloadBtn.textContent = "Скачать DXF";
          try {
            await loadViewer(jobId);
          } catch (err) {
            viewerEl.textContent = err.message || "Ошибка просмотрщика";
          }
          startBtn.disabled = false;
        } else if (data.status === "error") {
          clearInterval(state.pollTimer);
          setStatus(data.error || "Ошибка обработки", "error");
          startBtn.disabled = false;
        }
      } catch (err) {
        clearInterval(state.pollTimer);
        setStatus(err.message || "Ошибка сети", "error");
        startBtn.disabled = false;
      }
    }, 500);
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.drawings.length) return;

    const fd = new FormData(form);
    state.drawings.forEach((f) => fd.append("drawings", f));
    state.photos.forEach((f) => fd.append("photos", f));

    startBtn.disabled = true;
    downloadBtn.disabled = true;
    downloadBtn.setAttribute("aria-disabled", "true");
    downloadBtn.removeAttribute("href");
    layerToggles.innerHTML = "";
    viewerEl.textContent = "Обработка…";
    setProgress(2, "Загрузка файлов…");
    setStatus("Задача принята, обрабатываем…");

    try {
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      state.jobId = data.job_id;
      scaleHint?.classList.remove("is-warn");
      pollJob(data.job_id);
    } catch (err) {
      setStatus(err.message || "Ошибка загрузки", "error");
      startBtn.disabled = false;
    }
  });
})();
