/* Lightweight WebGL orthographic line viewer for DXF scene JSON */
(() => {
  const LAYER_COLORS = {
    "00_Оформление": [0.85, 0.85, 0.85],
    "00_Текст_Штампа": [0.75, 0.8, 0.9],
    "01_Стены_Капитальные": [0.95, 0.35, 0.3],
    "02_Перегородки": [0.95, 0.8, 0.25],
    "03_Оси": [0.25, 0.85, 0.9],
    "04_Размеры_ГОСТ": [0.35, 0.9, 0.45],
    "05_Окна_Двери": [0.35, 0.55, 1.0],
  };

  function hexThemeLine() {
    const dark = document.documentElement.getAttribute("data-theme") !== "light";
    return dark ? [0.12, 0.16, 0.22] : [0.92, 0.94, 0.96];
  }

  class DxfWebGLViewer {
    constructor(container) {
      this.container = container;
      this.canvas = document.createElement("canvas");
      this.canvas.className = "viewer-canvas";
      this.canvas.setAttribute("tabindex", "0");
      container.innerHTML = "";
      container.appendChild(this.canvas);

      this.gl = this.canvas.getContext("webgl", { antialias: true })
        || this.canvas.getContext("experimental-webgl");
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
        // Convert screen px to world approx via resolve
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
      this.canvas.width = Math.max(2, Math.floor(rect.width * dpr));
      this.canvas.height = Math.max(2, Math.floor(Math.max(rect.height, 320) * dpr));
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${Math.max(rect.height, 320)}px`;
      if (this.gl) {
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.draw();
      }
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
      for (const e of this.scene.entities || []) {
        if (this.layersOn[e.layer] === false) continue;
        if (!byLayer[e.layer]) byLayer[e.layer] = [];
        const pts = e.points || [];
        if (e.type === "line" && pts.length >= 2) {
          byLayer[e.layer].push(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
        } else if ((e.type === "polyline" || e.type === "text") && pts.length) {
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
    }
  }

  window.DxfWebGLViewer = DxfWebGLViewer;
})();
