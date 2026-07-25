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
  };

  const LAYER_LABELS = {
    "00_FRAME": "00_Оформление",
    "00_STAMP_TEXT": "00_Текст_Штампа",
    "01_WALLS": "01_Стены",
    "02_PARTITIONS": "02_Перегородки",
    "03_AXES": "03_Оси",
    "04_DIMS_GOST": "04_Размеры",
    "05_OPENINGS": "05_Окна_Двери",
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
      this.canvas.style.display = "block";
      this.canvas.style.width = "100%";
      this.canvas.style.height = "100%";

      this.textCanvas = document.createElement("canvas");
      this.textCanvas.className = "viewer-text-canvas";
      this.textCanvas.style.position = "absolute";
      this.textCanvas.style.left = "0";
      this.textCanvas.style.top = "0";
      this.textCanvas.style.width = "100%";
      this.textCanvas.style.height = "100%";
      this.textCanvas.style.pointerEvents = "none";

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
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this._dragging = false;
      this._last = null;
      this._initGL();
      this._bindInput();
      this._resize();
      window.addEventListener("resize", () => this._resize());
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
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(2, Math.floor(rect.width * dpr));
      const h = Math.max(2, Math.floor(Math.max(rect.height, 320) * dpr));
      for (const c of [this.canvas, this.textCanvas]) {
        c.width = w;
        c.height = h;
        c.style.width = `${rect.width}px`;
        c.style.height = `${Math.max(rect.height, 320)}px`;
      }
      if (this.gl) {
        this.gl.viewport(0, 0, w, h);
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
      // Expose Russian labels for UI toggles
      const labeled = {};
      Object.keys(this.layersOn).forEach((k) => {
        labeled[k] = this.layersOn[k];
      });
      labeled.__labels = LAYER_LABELS;
      return this.layersOn;
    }

    setLayer(name, on) {
      this.layersOn[name] = !!on;
      this.draw();
    }

    draw() {
      const gl = this.gl;
      if (!gl || !this.scene) return;
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
        } else if (e.type === "polyline" && pts.length) {
          for (let i = 0; i < pts.length - 1; i++) {
            byLayer[e.layer].push(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
          }
          if (e.closed && pts.length > 2) {
            const a = pts[pts.length - 1];
            const b = pts[0];
            byLayer[e.layer].push(a[0], a[1], b[0], b[1]);
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

      this._drawTexts(texts);
    }

    _drawTexts(texts) {
      const ctx = this.tctx;
      if (!ctx) return;
      ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
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
        ctx.font = `600 ${pxH}px "Segoe UI", "DejaVu Sans", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.lineWidth = Math.max(2, pxH * 0.12);
        ctx.strokeStyle = dark ? "rgba(0,0,0,0.65)" : "rgba(255,255,255,0.85)";
        ctx.fillStyle = rgbCss(color);
        const label = e.text || "";
        ctx.strokeText(label, sx, sy);
        ctx.fillText(label, sx, sy);
      }
    }
  }

  window.DxfWebGLViewer = DxfWebGLViewer;
  window.DXF_LAYER_LABELS = LAYER_LABELS;
})();
