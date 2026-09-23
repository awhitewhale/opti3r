const q = (selector, parent = document) => parent.querySelector(selector);

// Navigation and scroll reveals.
const menuButton = q('.menu-button');
const nav = q('nav');
menuButton?.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  menuButton.setAttribute('aria-expanded', String(open));
});
nav?.addEventListener('click', (event) => {
  if (event.target.closest('a')) {
    nav.classList.remove('is-open');
    menuButton?.setAttribute('aria-expanded', 'false');
  }
});

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

// Lightweight WebGL PLY viewer; the shipped cloud is binary little-endian
// with xyz, normal xyz, and RGB fields.
class PointCloudViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', { antialias: true, alpha: true });
    this.rotation = { x: -0.28, y: -0.54 };
    this.targetRotation = { ...this.rotation };
    this.zoom = 2.8;
    this.targetZoom = this.zoom;
    this.dragging = false;
    this.lastPointer = null;
    this.autoOrbit = true;
    this.init();
  }

  async init() {
    if (!this.gl) return this.fail('WebGL is unavailable');
    this.program = this.createProgram();
    this.bindEvents();
    this.resize();
    window.addEventListener('resize', () => this.resize());
    try {
      const response = await fetch('assets/cv1173.ply');
      if (!response.ok) throw new Error('Point cloud failed to load');
      this.uploadCloud(await response.arrayBuffer());
      q('#viewer-loading')?.classList.add('is-hidden');
      this.render();
    } catch (error) {
      this.fail('Unable to load the point cloud');
    }
  }

  fail(message) {
    const loading = q('#viewer-loading');
    if (loading) loading.textContent = message;
  }

  createProgram() {
    const gl = this.gl;
    const vertexSource = `
      attribute vec3 aPosition;
      attribute vec3 aColor;
      uniform mat4 uMatrix;
      uniform float uPointSize;
      varying vec3 vColor;
      varying float vDepth;
      void main() {
        vec4 position = uMatrix * vec4(aPosition, 1.0);
        gl_Position = position;
        gl_PointSize = max(1.1, uPointSize / max(0.7, position.w));
        vColor = aColor;
        vDepth = position.z;
      }
    `;
    const fragmentSource = `
      precision mediump float;
      varying vec3 vColor;
      varying float vDepth;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        if (dot(p, p) > 0.25) discard;
        vec3 color = mix(vColor, vec3(0.02, 0.28, 0.40), 0.04);
        gl_FragColor = vec4(color, 0.96);
      }
    `;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  uploadCloud(buffer) {
    const bytes = new Uint8Array(buffer);
    const marker = new TextEncoder().encode('end_header\n');
    let dataOffset = -1;
    outer: for (let i = 0; i < bytes.length - marker.length; i++) {
      for (let j = 0; j < marker.length; j++) if (bytes[i + j] !== marker[j]) continue outer;
      dataOffset = i + marker.length;
      break;
    }
    if (dataOffset < 0) throw new Error('Invalid PLY header');
    const header = new TextDecoder().decode(bytes.slice(0, dataOffset));
    const count = Number(header.match(/element vertex (\d+)/)?.[1] || 0);
    const stride = 27;
    const view = new DataView(buffer, dataOffset);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      const base = i * stride;
      for (let axis = 0; axis < 3; axis++) {
        const value = view.getFloat32(base + axis * 4, true);
        positions[i * 3 + axis] = value;
        min[axis] = Math.min(min[axis], value);
        max[axis] = Math.max(max[axis], value);
      }
      colors[i * 3] = view.getUint8(base + 24) / 255;
      colors[i * 3 + 1] = view.getUint8(base + 25) / 255;
      colors[i * 3 + 2] = view.getUint8(base + 26) / 255;
    }
    const center = min.map((value, axis) => (value + max[axis]) / 2);
    const scale = 1.8 / Math.max(...max.map((value, axis) => value - min[axis]));
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = (positions[i] - center[0]) * scale;
      positions[i + 1] = (positions[i + 1] - center[1]) * scale;
      positions[i + 2] = (positions[i + 2] - center[2]) * scale;
    }
    this.count = count;
    this.positionBuffer = this.makeBuffer(positions);
    this.colorBuffer = this.makeBuffer(colors);
  }

  makeBuffer(data) {
    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
    return buffer;
  }

  bindEvents() {
    const canvas = this.canvas;
    canvas.addEventListener('pointerdown', (event) => {
      this.dragging = true;
      this.autoOrbit = false;
      this.lastPointer = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.dragging) return;
      this.targetRotation.y += (event.clientX - this.lastPointer[0]) * 0.008;
      this.targetRotation.x += (event.clientY - this.lastPointer[1]) * 0.008;
      this.targetRotation.x = Math.max(-1.35, Math.min(1.35, this.targetRotation.x));
      this.lastPointer = [event.clientX, event.clientY];
    });
    canvas.addEventListener('pointerup', () => { this.dragging = false; });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      this.autoOrbit = false;
      this.targetZoom = Math.max(1.5, Math.min(5.2, this.targetZoom + event.deltaY * 0.002));
    }, { passive: false });
    q('#viewer-reset')?.addEventListener('click', () => {
      this.targetRotation = { x: -0.28, y: -0.54 };
      this.targetZoom = 2.8;
      this.autoOrbit = true;
    });
  }

  resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(this.canvas.clientWidth * ratio);
    const height = Math.round(this.canvas.clientHeight * ratio);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.gl?.viewport(0, 0, width, height);
    }
  }

  matrix() {
    const aspect = this.canvas.width / this.canvas.height;
    const fov = 0.82;
    const f = 1 / Math.tan(fov / 2);
    const near = 0.1, far = 100;
    const projection = new Float32Array([
      f / aspect, 0, 0, 0, 0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0,
    ]);
    const cx = Math.cos(this.rotation.x), sx = Math.sin(this.rotation.x);
    const cy = Math.cos(this.rotation.y), sy = Math.sin(this.rotation.y);
    const model = new Float32Array([
      cy, sx * sy, -cx * sy, 0,
      0, cx, sx, 0,
      sy, -sx * cy, cx * cy, 0,
      0, 0, -this.zoom, 1,
    ]);
    const result = new Float32Array(16);
    for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
      result[col * 4 + row] = projection[row] * model[col * 4] + projection[4 + row] * model[col * 4 + 1] + projection[8 + row] * model[col * 4 + 2] + projection[12 + row] * model[col * 4 + 3];
    }
    return result;
  }

  render() {
    if (!this.count) return;
    this.resize();
    if (this.autoOrbit && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) this.targetRotation.y += 0.0015;
    this.rotation.x += (this.targetRotation.x - this.rotation.x) * 0.08;
    this.rotation.y += (this.targetRotation.y - this.rotation.y) * 0.08;
    this.zoom += (this.targetZoom - this.zoom) * 0.08;
    const gl = this.gl;
    gl.clearColor(0.969, 0.976, 0.980, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);
    const bindAttribute = (name, buffer) => {
      const location = gl.getAttribLocation(this.program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
    };
    bindAttribute('aPosition', this.positionBuffer);
    bindAttribute('aColor', this.colorBuffer);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'uMatrix'), false, this.matrix());
    gl.uniform1f(gl.getUniformLocation(this.program, 'uPointSize'), Math.min(window.devicePixelRatio || 1, 2) * 4.2);
    gl.drawArrays(gl.POINTS, 0, this.count);
    requestAnimationFrame(() => this.render());
  }
}

const canvas = q('#point-cloud');
if (canvas) new PointCloudViewer(canvas);
