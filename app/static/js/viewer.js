/* Lightweight WebGL line viewer + 2D canvas text overlay (GOST dim/axis labels) */
(() => {
  const LAYER_COLORS = {
    "00_FRAME": [0.55, 0.58, 0.62],
    "00_STAMP_TEXT": [0.45, 0.5, 0.6],
    "01_WALLS": [0.85, 0.2, 0.18],
    "02_PARTITIONS": [0.85, 0.65, 0.15],
    "03_AXES": [0.1, 0.7, 0.78],
    "04_DIMS_GOST": [0.12, 0.65, 0.28],
    "05_OPENINGS": [0.25, 0.4, 0.95],
    "06_DETAIL": [0.75, 0.78, 0.82],
    "06_STAIRS": [0.85, 0.75, 0.55],
    "07_HATCH": [0.75, 0.25, 0.7],
  };

  const LAYER_LABELS = {
    "00_FRAME": "00_Оформление",
    "00_STAMP_TEXT": "00_Текст_Штампа",
    "01_WALLS": "01_Стены",
    "02_PARTITIONS": "02_Перегородки",
    "03_AXES": "03_Оси",
    "04_DIMS_GOST": "04_Размеры",
    "05_OPENINGS": "05_Окна_Двери",
    "06_DETAIL": "06_Детали",
    "06_STAIRS": "06_Лестницы",
    "07_HATCH": "07_Штриховка",
  };

  function hexThemeLine() {
    const dark = document.documentElement.getAttribute("data-theme") !== "light";
    return dark ? [0.12, 0.16, 0.22] : [0.92, 0.94, 0.96];
  }

  function rgbCss(rgb) {
    return `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`;
  }

  class DxfWebGLViewer {
    constructor(container) {
      this.container = container;
      this.container.style.position = "relative";
      this.container.innerHTML = "";

      this.canvas = document.createElement("canvas");
      this.canvas.className = "viewer-canvas";
      this.canvas.setAttribute("tabindex", "0");

      this.textCanvas = document.createElement("canvas");
      this.textCanvas.className = "viewer-text-canvas";

      this.container.appendChild(this.canvas);
      this.container.appendChild(this.textCanvas);

      this.gl = this.canvas.getContext("webgl", { antialias: true })
        || this.canvas.getContext("experimental-webgl");
      this.tctx = this.textCanvas.getContext("2d");
      if (!this.gl) {
        container.textContent = "WebGL недоступен в этом браузере.";
        return;
      }

      this.layersOn = {};
      this.scene = null;
      this.raw = null;
      this.filters = {
        med_t_min: 0.55,
        thinness_min: 40,
        thinness_max: 320,
        aspect_min: 1.2,
        ink_ratio_min: 0.15,
        min_length: 200,
      };
      this.filteredContours = [];
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this._dragging = false;
      this._last = null;
      this.underlayOpacity = 0.28;
      this._underlayImg = null;
      this._underlayUrl = null;
      this._initGL();
      this._bindInput();
      this._resize();
      this._onWindowResize = () => this._resize();
      window.addEventListener("resize", this._onWindowResize);
      if (typeof ResizeObserver !== "undefined") {
        this._ro = new ResizeObserver(() => this._resize());
        this._ro.observe(this.container);
      }
    }

    _initGL() {
      const gl = this.gl;
      const vs = this._shader(gl.VERTEX_SHADER, `
        attribute vec2 a_pos;
        uniform vec2 u_resolve;
        uniform vec2 u_center;
        uniform float u_zoom;
        uniform vec2 u_pan;
        void main() {
          vec2 p = (a_pos - u_center) * u_zoom + u_pan;
          vec2 clip = vec2(p.x / u_resolve.x, p.y / u_resolve.y);
          gl_Position = vec4(clip, 0.0, 1.0);
        }
      `);
      const fs = this._shader(gl.FRAGMENT_SHADER, `
        precision mediump float;
        uniform vec3 u_color;
        void main() { gl_FragColor = vec4(u_color, 1.0); }
      `);
      this.prog = gl.createProgram();
      gl.attachShader(this.prog, vs);
      gl.attachShader(this.prog, fs);
      gl.linkProgram(this.prog);
      this.aPos = gl.getAttribLocation(this.prog, "a_pos");
      this.uResolve = gl.getUniformLocation(this.prog, "u_resolve");
      this.uCenter = gl.getUniformLocation(this.prog, "u_center");
      this.uZoom = gl.getUniformLocation(this.prog, "u_zoom");
      this.uPan = gl.getUniformLocation(this.prog, "u_pan");
      this.uColor = gl.getUniformLocation(this.prog, "u_color");
      this.buf = gl.createBuffer();
    }

    _shader(type, src) {
      const gl = this.gl;
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }

    _bindInput() {
      this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        this.zoom = Math.min(80, Math.max(0.05, this.zoom * factor));
        this.draw();
      }, { passive: false });

      this.canvas.addEventListener("pointerdown", (e) => {
        this._dragging = true;
        this._last = [e.clientX, e.clientY];
        this.canvas.setPointerCapture(e.pointerId);
      });
      this.canvas.addEventListener("pointermove", (e) => {
        if (!this._dragging || !this._last) return;
        const dx = e.clientX - this._last[0];
        const dy = e.clientY - this._last[1];
        this._last = [e.clientX, e.clientY];
        const scale = (this._resolveY * 2) / this.canvas.height;
        this.panX += dx * scale;
        this.panY -= dy * scale;
        this.draw();
      });
      const end = () => { this._dragging = false; this._last = null; };
      this.canvas.addEventListener("pointerup", end);
      this.canvas.addEventListener("pointercancel", end);
    }

    _resize() {
      const rect = this.container.getBoundingClientRect();
      const cssW = Math.max(2, rect.width);
      const cssH = Math.max(2, rect.height || 320);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(2, Math.floor(cssW * dpr));
      const h = Math.max(2, Math.floor(cssH * dpr));
      let changed = false;
      for (const c of [this.canvas, this.textCanvas]) {
        if (c.width !== w || c.height !== h) {
          c.width = w;
          c.height = h;
          changed = true;
        }
        // Keep CSS fluid — never lock to a fixed px width
        c.style.width = "100%";
        c.style.height = "100%";
      }
      if (this.gl && changed) {
        this.gl.viewport(0, 0, w, h);
        this.draw();
      } else if (this.gl) {
        this.draw();
      }
    }

    /** World mm → canvas pixel */
    _worldToScreen(x, y) {
      const cx = this._center[0];
      const cy = this._center[1];
      const px = (x - cx) * this.zoom + this.panX;
      const py = (y - cy) * this.zoom + this.panY;
      const sx = (px / this._resolveX) * 0.5 + 0.5;
      const sy = 0.5 - (py / this._resolveY) * 0.5;
      return [sx * this.canvas.width, sy * this.canvas.height];
    }

    setScene(scene) {
      this.raw = null;
      this.scene = scene;
      (scene.layers || []).forEach((l) => {
        if (this.layersOn[l] === undefined) this.layersOn[l] = true;
      });
      const b = scene.bounds || { min: [0, 0], max: [100, 100] };
      const w = Math.max(1, b.max[0] - b.min[0]);
      const h = Math.max(1, b.max[1] - b.min[1]);
      this._center = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2];
      this._resolveY = h * 0.6;
      this._resolveX = (w * 0.6) * (this.canvas.width / Math.max(this.canvas.height, 1));
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this.draw();
      const labeled = {};
      Object.keys(this.layersOn).forEach((k) => {
        labeled[k] = this.layersOn[k];
      });
      labeled.__labels = LAYER_LABELS;
      return this.layersOn;
    }

    setRawGeometry(raw) {
      this.raw = raw || null;
      this.scene = null;
      if (raw?.filters_default) {
        this.filters = { ...this.filters, ...raw.filters_default };
      }
      const b = raw?.bounds || { min: [0, 0], max: [12000, 8000] };
      const w = Math.max(1, b.max[0] - b.min[0]);
      const h = Math.max(1, b.max[1] - b.min[1]);
      this._center = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2];
      this._resolveY = h * 0.6;
      this._resolveX = (w * 0.6) * (this.canvas.width / Math.max(this.canvas.height, 1));
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this.layersOn = {
        "01_WALLS": true,
        "03_AXES": true,
        "04_DIMS_GOST": true,
        "05_OPENINGS": true,
        "06_DETAIL": true,
      };
      this._loadUnderlay(raw?.scan_url || raw?.underlay?.url || null);
      this.applyFilters();
      return this.layersOn;
    }

    setUnderlayOpacity(v) {
      this.underlayOpacity = Math.max(0, Math.min(1, Number(v) || 0));
      this.draw();
    }

    _loadUnderlay(url) {
      if (!url) {
        this._underlayImg = null;
        this._underlayUrl = null;
        return;
      }
      if (url === this._underlayUrl && this._underlayImg) return;
      this._underlayUrl = url;
      const img = new Image();
      img.onload = () => {
        this._underlayImg = img;
        this.draw();
      };
      img.onerror = () => {
        this._underlayImg = null;
      };
      img.src = url;
    }

    setFilter(name, value) {
      this.filters[name] = Number(value);
      this.applyFilters();
    }

    setFilters(obj) {
      Object.keys(obj || {}).forEach((k) => {
        this.filters[k] = Number(obj[k]);
      });
      this.applyFilters();
    }

    getFilters() {
      return { ...this.filters };
    }

    _passes(props) {
      const f = this.filters;
      const med = Number(props?.med_t || 0);
      const thin = Number(props?.thinness || 0);
      const aspect = Number(props?.aspect || 0);
      const ink = Number(props?.ink_ratio || 0);
      const len = Number(props?.length || props?.perimeter || 0);
      const area = Number(props?.area || 0);
      if (area < 1e-3) {
        return aspect >= f.aspect_min
          && len >= f.min_length
          && med >= f.med_t_min * 0.5;
      }
      return med >= f.med_t_min
        && thin >= f.thinness_min
        && thin <= f.thinness_max
        && aspect >= f.aspect_min
        && ink >= f.ink_ratio_min
        && len >= f.min_length;
    }

    applyFilters() {
      if (!this.raw) return;
      const contours = this.raw.contours || [];
      this.filteredContours = contours.filter((c) => this._passes(c.properties || {}));
      this.draw();
    }

    setLayer(name, on) {
      this.layersOn[name] = !!on;
      this.draw();
    }

    draw() {
      const gl = this.gl;
      if (!gl) return;
      if (this.raw) {
        this._drawRaw();
        return;
      }
      if (!this.scene) return;
      this._drawScene();
    }

    _drawRaw() {
      const gl = this.gl;
      const bg = hexThemeLine();
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);

      const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
      this._resolveX = this._resolveY * aspect;
      gl.uniform2f(this.uResolve, this._resolveX, this._resolveY);
      gl.uniform2f(this.uCenter, this._center[0], this._center[1]);
      gl.uniform1f(this.uZoom, this.zoom);
      gl.uniform2f(this.uPan, this.panX, this.panY);

      const wallArr = [];
      const detailArr = [];
      const all = this.raw.contours || [];
      const kept = new Set(this.filteredContours.map((c) => c.id));
      for (const c of all) {
        const pts = c.points || [];
        if (pts.length < 2) continue;
        const target = kept.has(c.id) ? wallArr : detailArr;
        if (kept.has(c.id) && this.layersOn["01_WALLS"] === false) continue;
        if (!kept.has(c.id) && this.layersOn["06_DETAIL"] === false) continue;
        for (let i = 0; i < pts.length - 1; i++) {
          target.push(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
        }
      }
      // Stairs → dedicated stairs layer (not walls)
      const stairArr = [];
      if (this.layersOn["06_DETAIL"] !== false) {
        for (const stair of this.raw.stairs || []) {
          if (!stair || stair.length < 2) continue;
          for (let i = 0; i < stair.length - 1; i++) {
            stairArr.push(stair[i][0], stair[i][1], stair[i + 1][0], stair[i + 1][1]);
          }
          if (stair.length > 2) {
            const a = stair[stair.length - 1];
            const b = stair[0];
            stairArr.push(a[0], a[1], b[0], b[1]);
          }
        }
      }

      // Axes as simple lines
      const axisArr = [];
      if (this.layersOn["03_AXES"] !== false) {
        for (const a of this.raw.axes || []) {
          if ("x" in a) {
            axisArr.push(a.x, a.y0 ?? 0, a.x, a.y1 ?? 0);
          } else if ("y" in a) {
            axisArr.push(a.x0 ?? 0, a.y, a.x1 ?? 0, a.y);
          }
        }
        for (const c of this.raw.circles || []) {
          const n = 32;
          const cx = c.cx, cy = c.cy, r = c.r || 100;
          for (let i = 0; i < n; i++) {
            const a0 = (i / n) * Math.PI * 2;
            const a1 = ((i + 1) / n) * Math.PI * 2;
            axisArr.push(cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + r * Math.cos(a1), cy + r * Math.sin(a1));
          }
        }
      }

      const dimArr = [];
      if (this.layersOn["04_DIMS_GOST"] !== false) {
        for (const d of this.raw.dimensions || []) {
          if (!d.p1 || !d.p2) continue;
          dimArr.push(d.p1[0], d.p1[1], d.p2[0], d.p2[1]);
        }
      }

      const openArr = [];
      if (this.layersOn["05_OPENINGS"] !== false) {
        for (const op of this.raw.openings || []) {
          if (op.leaf && op.leaf.length >= 2) {
            openArr.push(op.leaf[0][0], op.leaf[0][1], op.leaf[1][0], op.leaf[1][1]);
            continue;
          }
          if (!op.p1 || !op.p2) continue;
          const x1 = op.p1[0], y1 = op.p1[1], x2 = op.p2[0], y2 = op.p2[1];
          const dx = x2 - x1, dy = y2 - y1;
          const len = Math.hypot(dx, dy) || 1;
          const ux = dx / len, uy = dy / len;
          const nx = -uy, ny = ux;
          let half = Number(op.width_mm);
          if (!Number.isFinite(half) || half < 40) half = 100;
          half *= 0.5;
          // Centerline + two side lines (reads as a thin opening, not a fat box)
          openArr.push(x1, y1, x2, y2);
          openArr.push(x1 + nx * half, y1 + ny * half, x2 + nx * half, y2 + ny * half);
          openArr.push(x1 - nx * half, y1 - ny * half, x2 - nx * half, y2 - ny * half);
        }
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.enableVertexAttribArray(this.aPos);
      gl.vertexAttribPointer(this.aPos, 2, gl.FLOAT, false, 0, 0);

      const batches = [
        [detailArr, LAYER_COLORS["06_DETAIL"]],
        [stairArr, LAYER_COLORS["06_STAIRS"]],
        [axisArr, LAYER_COLORS["03_AXES"]],
        [dimArr, LAYER_COLORS["04_DIMS_GOST"]],
        [openArr, LAYER_COLORS["05_OPENINGS"]],
        [wallArr, LAYER_COLORS["01_WALLS"]],
      ];
      for (const [arr, color] of batches) {
        if (!arr.length) continue;
        gl.uniform3fv(this.uColor, color);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.LINES, 0, arr.length / 2);
      }

      const texts = [];
      for (const t of this.raw.texts || []) {
        const kind = String(t.kind || "").toLowerCase();
        const layer = (kind === "label" || kind === "detail")
          ? "06_DETAIL"
          : "04_DIMS_GOST";
        if (this.layersOn[layer] === false) continue;
        texts.push({
          type: "text",
          layer,
          points: [t.pos || [0, 0]],
          text: t.text || "",
          height: t.height || 120,
          rotation: t.rotation || 0,
        });
      }
      this._drawFills([]);
      this._drawUnderlay();
      this._drawTexts(texts);
    }

    _drawScene() {
      const gl = this.gl;
      const bg = hexThemeLine();
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);

      const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
      this._resolveX = this._resolveY * aspect;
      gl.uniform2f(this.uResolve, this._resolveX, this._resolveY);
      gl.uniform2f(this.uCenter, this._center[0], this._center[1]);
      gl.uniform1f(this.uZoom, this.zoom);
      gl.uniform2f(this.uPan, this.panX, this.panY);

      const byLayer = {};
      const fills = [];
      const texts = [];
      for (const e of this.scene.entities || []) {
        if (this.layersOn[e.layer] === false) continue;
        if (e.type === "text") {
          texts.push(e);
          continue;
        }
        if (!byLayer[e.layer]) byLayer[e.layer] = [];
        const pts = e.points || [];
        if (e.type === "line" && pts.length >= 2) {
          byLayer[e.layer].push(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
        } else if (
          (e.type === "polyline" || e.type === "circle" || e.type === "arc")
          && pts.length
        ) {
          for (let i = 0; i < pts.length - 1; i++) {
            byLayer[e.layer].push(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
          }
          if (e.closed && pts.length > 2 && e.type === "polyline") {
            const a = pts[pts.length - 1];
            const b = pts[0];
            byLayer[e.layer].push(a[0], a[1], b[0], b[1]);
          }
          if (e.filled && e.closed && pts.length >= 3) {
            fills.push(e);
          }
        }
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.enableVertexAttribArray(this.aPos);
      gl.vertexAttribPointer(this.aPos, 2, gl.FLOAT, false, 0, 0);

      for (const [layer, arr] of Object.entries(byLayer)) {
        if (!arr.length) continue;
        const color = LAYER_COLORS[layer] || [0.7, 0.75, 0.8];
        gl.uniform3fv(this.uColor, color);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.LINES, 0, arr.length / 2);
      }

      this._drawFills(fills);
      this._drawTexts(texts);
    }

    _drawFills(fills) {
      const ctx = this.tctx;
      if (!ctx) return;
      ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
      for (const e of fills) {
        const pts = e.points || [];
        if (pts.length < 3) continue;
        const color = LAYER_COLORS[e.layer] || [0.7, 0.75, 0.8];
        ctx.beginPath();
        const [sx0, sy0] = this._worldToScreen(pts[0][0], pts[0][1]);
        ctx.moveTo(sx0, sy0);
        for (let i = 1; i < pts.length; i++) {
          const [sx, sy] = this._worldToScreen(pts[i][0], pts[i][1]);
          ctx.lineTo(sx, sy);
        }
        ctx.closePath();
        ctx.fillStyle = `rgba(${Math.round(color[0] * 255)},${Math.round(color[1] * 255)},${Math.round(color[2] * 255)},0.22)`;
        ctx.fill();
      }
    }

    _drawUnderlay() {
      const ctx = this.tctx;
      const img = this._underlayImg;
      if (!ctx || !img || !this.raw || this.underlayOpacity <= 0.01) return;
      const ul = this.raw.underlay || {};
      const b = this.raw.bounds || { min: [0, 0], max: [12000, 8000] };
      const min = ul.min || b.min || [0, 0];
      const max = ul.max || b.max || [12000, 8000];
      // Image Y-down ↔ CAD Y-up: place image top at CAD maxY
      const [sx0, sy0] = this._worldToScreen(min[0], max[1]);
      const [sx1, sy1] = this._worldToScreen(max[0], min[1]);
      const w = sx1 - sx0;
      const h = sy1 - sy0;
      if (Math.abs(w) < 2 || Math.abs(h) < 2) return;
      ctx.save();
      ctx.globalAlpha = this.underlayOpacity;
      ctx.drawImage(img, sx0, sy0, w, h);
      ctx.restore();
    }

    _drawTexts(texts) {
      const ctx = this.tctx;
      if (!ctx) return;
      // Fills already painted in draw(); do not wipe them when drawing labels
      const dark = document.documentElement.getAttribute("data-theme") !== "light";
      for (const e of texts) {
        const [x, y] = e.points[0] || [0, 0];
        const [sx, sy] = this._worldToScreen(x, y);
        if (sx < -80 || sy < -40 || sx > this.textCanvas.width + 80 || sy > this.textCanvas.height + 40) {
          continue;
        }
        // World height → screen px
        const [, sy2] = this._worldToScreen(x, y + (e.height || 100));
        const pxH = Math.max(9, Math.min(36, Math.abs(sy - sy2)));
        const color = LAYER_COLORS[e.layer] || [0.2, 0.2, 0.2];
        const rot = ((e.rotation || 0) * Math.PI) / 180;
        ctx.save();
        ctx.translate(sx, sy);
        if (rot) ctx.rotate(-rot);
        ctx.font = `600 ${pxH}px "Segoe UI", "DejaVu Sans", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.lineWidth = Math.max(2, pxH * 0.12);
        ctx.strokeStyle = dark ? "rgba(0,0,0,0.65)" : "rgba(255,255,255,0.85)";
        ctx.fillStyle = rgbCss(color);
        const label = e.text || "";
        ctx.strokeText(label, 0, 0);
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
    }
  }

  window.DxfWebGLViewer = DxfWebGLViewer;
  window.DXF_LAYER_LABELS = LAYER_LABELS;
})();
