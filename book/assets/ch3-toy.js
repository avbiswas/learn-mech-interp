/* Toy model of superposition (Elhage et al. 2022), tied weights:
   h = W x ;  x' = ReLU(Wᵀ h + b)   (or linear: x' = Wᵀ h + b)
   loss = mean over batch of Σ_i I_i (x_i − x'_i)²
   Hand-written gradients + Adam. Works in browser and in node. */
(function (root) {
  function makeRng(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
  function gauss(r) { const u = r() || 1e-9, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  function ToyModel(opts) {
    this.n = opts.n; this.m = opts.m;
    this.S = opts.S; this.decay = opts.decay == null ? 1 : opts.decay;
    this.relu = opts.relu !== false;
    this.lr = opts.lr || 0.01; this.batch = opts.batch || 256;
    this.rng = makeRng(opts.seed || 1);
    this.reset();
  }
  ToyModel.prototype.reset = function () {
    const { n, m } = this;
    this.W = new Float64Array(m * n);      // W[k*n + i]
    for (let k = 0; k < m * n; k++) this.W[k] = gauss(this.rng) * Math.sqrt(1 / m) * 0.7;
    this.b = new Float64Array(n);
    this.mW = new Float64Array(m * n); this.vW = new Float64Array(m * n);
    this.mb = new Float64Array(n); this.vb = new Float64Array(n);
    this.t = 0; this.loss = NaN;
    this.I = new Float64Array(n);
    for (let i = 0; i < n; i++) this.I[i] = Math.pow(this.decay, i);
  };
  ToyModel.prototype.setImportance = function (decay) {
    this.decay = decay;
    for (let i = 0; i < this.n; i++) this.I[i] = Math.pow(decay, i);
  };
  ToyModel.prototype.forward = function (x, h, out, pre) {
    const { n, m, W, b } = this;
    for (let k = 0; k < m; k++) { let s = 0; for (let i = 0; i < n; i++) s += W[k * n + i] * x[i]; h[k] = s; }
    for (let i = 0; i < n; i++) {
      let s = b[i]; for (let k = 0; k < m; k++) s += W[k * n + i] * h[k];
      pre[i] = s; out[i] = this.relu ? Math.max(0, s) : s;
    }
  };
  ToyModel.prototype.step = function () {
    const { n, m, W, b, I, batch, rng } = this;
    const gW = new Float64Array(m * n), gb = new Float64Array(n);
    const x = new Float64Array(n), h = new Float64Array(m), out = new Float64Array(n), pre = new Float64Array(n), e = new Float64Array(n), We = new Float64Array(m);
    let L = 0;
    for (let s = 0; s < batch; s++) {
      for (let i = 0; i < n; i++) x[i] = rng() < this.S ? 0 : rng();
      this.forward(x, h, out, pre);
      for (let i = 0; i < n; i++) {
        const d = out[i] - x[i];
        L += I[i] * d * d;
        e[i] = (this.relu && pre[i] <= 0) ? 0 : 2 * I[i] * d;
        gb[i] += e[i];
      }
      for (let k = 0; k < m; k++) { let s2 = 0; for (let i = 0; i < n; i++) s2 += W[k * n + i] * e[i]; We[k] = s2; }
      // dL/dW = (W x) eᵀ + (W e) xᵀ
      for (let k = 0; k < m; k++) for (let i = 0; i < n; i++) gW[k * n + i] += h[k] * e[i] + We[k] * x[i];
    }
    this.loss = L / batch;
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, lr = this.lr;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let k = 0; k < m * n; k++) {
      const g = gW[k] / batch;
      this.mW[k] = b1 * this.mW[k] + (1 - b1) * g; this.vW[k] = b2 * this.vW[k] + (1 - b2) * g * g;
      W[k] -= lr * (this.mW[k] / c1) / (Math.sqrt(this.vW[k] / c2) + eps);
    }
    for (let i = 0; i < n; i++) {
      const g = gb[i] / batch;
      this.mb[i] = b1 * this.mb[i] + (1 - b1) * g; this.vb[i] = b2 * this.vb[i] + (1 - b2) * g * g;
      b[i] -= lr * (this.mb[i] / c1) / (Math.sqrt(this.vb[i] / c2) + eps);
    }
    return this.loss;
  };
  /* column i of W as array of length m */
  ToyModel.prototype.col = function (i) { const c = []; for (let k = 0; k < this.m; k++) c.push(this.W[k * this.n + i]); return c; };
  ToyModel.prototype.gram = function () {
    const { n, m, W } = this, G = [];
    for (let i = 0; i < n; i++) { G.push([]); for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < m; k++) s += W[k * n + i] * W[k * n + j]; G[i].push(s); } }
    return G;
  };
  /* feature dimensionality D_i = ||W_i||² / Σ_j (Ŵ_i·W_j)² */
  ToyModel.prototype.dims = function () {
    const G = this.gram(), D = [];
    for (let i = 0; i < this.n; i++) {
      const ni = G[i][i];
      if (ni < 1e-4) { D.push(0); continue; }
      let s = 0; for (let j = 0; j < this.n; j++) s += (G[i][j] * G[i][j]) / ni;
      D.push(ni / s);
    }
    return D;
  };
  /* expected loss on a fresh eval batch (for picking the best restart) */
  ToyModel.prototype.evalLoss = function (N) {
    const r = makeRng(12345), n = this.n, x = new Float64Array(n), h = new Float64Array(this.m), out = new Float64Array(n), pre = new Float64Array(n);
    let L = 0;
    for (let s = 0; s < N; s++) {
      for (let i = 0; i < n; i++) x[i] = r() < this.S ? 0 : r();
      this.forward(x, h, out, pre);
      for (let i = 0; i < n; i++) L += this.I[i] * (out[i] - x[i]) ** 2;
    }
    return L / N;
  };

  root.ToyModel = ToyModel;
  if (typeof module !== "undefined") module.exports = { ToyModel };
})(typeof window !== "undefined" ? window : globalThis);
